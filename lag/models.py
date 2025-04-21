from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.lax import scan as jscan
from numpy.typing import NDArray
from numpyro import factor

from lag.data import CoalescentData, GenomicData
from lag.utils import choose2


class RtModel(ABC):
    @classmethod
    def fit(
        cls,
        data: GenomicData,
        hyperparameters: dict,
        rng_key,
        mcmc_config: dict[str, Any],
        nuts_config: dict[str, Any] = {},
    ) -> numpyro.infer.MCMC:
        kernel = numpyro.infer.NUTS(cls.model, **nuts_config)
        mcmc = numpyro.infer.MCMC(kernel, **mcmc_config)
        mcmc.run(
            rng_key,
            data=data,
            hyperparameters=hyperparameters,
        )
        return mcmc

    @classmethod
    @abstractmethod
    def model(cls, data: GenomicData, hyperparameters: dict):
        raise NotImplementedError()


class RenewalCoalescentModel(RtModel):
    def __init__(self):
        raise NotImplementedError(
            "RenewalCoalescentModel is not designed to be instantiated."
        )

    @classmethod
    def model(
        cls,
        data: CoalescentData,
        hyperparameters: dict[str, Any],
    ):
        """
        A model inferring Rt from coalescent times using a discrete-time renewal model as the link.
        """

        # Renewal model
        daily_rt = RenewalCoalescentModel.daily_rt(
            hyperparameters["n_weeks"], hyperparameters["renewal_t_max"]
        )
        i0 = RenewalCoalescentModel.i0()
        init_growth_rate = RenewalCoalescentModel.exp_growth_rate()
        daily_incidence = RenewalCoalescentModel.daily_incidence(
            daily_rt,
            hyperparameters["reversed_infectiousness_profile"],
            i0,
            init_growth_rate,
            hyperparameters["init_growth_steps"],
        )
        daily_prevalence = RenewalCoalescentModel.daily_prevalence(
            daily_incidence, hyperparameters["generation_interval"]
        )

        # Coalescent likelihood
        factor(
            "coalescent_likelihood",
            RenewalCoalescentModel.piecewise_constant_log_likelihood(
                data,
                jnp.flip(daily_incidence),
                jnp.flip(
                    RenewalCoalescentModel.approx_squared_prevalence(
                        daily_prevalence
                    )
                ),
            ),
        )

    ########################
    # Coalescent utilities #
    ########################

    @staticmethod
    def approx_squared_prevalence(prevalence):
        prev_diff = jnp.diff(prevalence)
        prev_cubed = jnp.pow(prevalence[:-1], 3.0)
        return jnp.where(
            prev_diff == 0.0,
            prev_cubed,
            (jnp.pow(prevalence[:-1] + jnp.diff(prevalence), 3.0) - prev_cubed)
            / (3.0 * prev_diff),
        )

    @staticmethod
    def approx_coalescent_rate(
        approx_squared_prevalence, force_of_infection, n_active_choose_2
    ):
        return (
            n_active_choose_2 * force_of_infection / approx_squared_prevalence
        )

    @staticmethod
    def grid_helper(
        coalescent_times: NDArray,
        sampling_times: NDArray,
        reversed_infectiousness_profile: NDArray,
        generation_interval: int,
    ):
        """
        Figures out where the tree lives in renewal grid time and the farthest back the grid must go.
        """
        tmrs = sampling_times.min()
        tmrca = coalescent_times.max()

        # The discrete-time intervals into which these events fall
        tree_t_min = np.floor(tmrs).astype(int)
        tree_t_max = np.floor(tmrca).astype(int)
        # You need at least generation_interval days of incidence before t_MRCA to get prevalence at that time
        # You need a burn-in of at least reversed_infectiousness_profile to have the entire renewal history contributing to daily incidence at some time
        tmp = 1 + generation_interval + len(reversed_infectiousness_profile)
        burnin = 50 if 50 > tmp else tmp

        return tree_t_min, tree_t_max, burnin

    @staticmethod
    def piecewise_constant_log_likelihood(
        intervals: CoalescentData,
        force_of_infection,
        approx_squared_prevalence,
    ):
        """
        Computes the likelihood of construct_epi_coalescent_grid(coalescent_times, sampling_times, rate_shift_times)
        given the piecewise constant force of infection and approximated piecewise constant squared prevalence.
        """

        rate = RenewalCoalescentModel.approx_coalescent_rate(
            approx_squared_prevalence[intervals.rate_indexer],
            force_of_infection[intervals.rate_indexer],
            intervals.num_active_choose_2,
        )
        lnl = rate * intervals.dt - jnp.where(
            intervals.ends_in_coalescent_indicator, jnp.log(rate), 0.0
        )
        return lnl.sum()

    @staticmethod
    def preprocess_from_vectors(
        coalescent_times: NDArray,
        sampling_times: NDArray,
        reversed_infectiousness_profile: NDArray,
        generation_interval: int,
    ) -> tuple[CoalescentData, dict[str, Any]]:
        """
        Convenience function to obtain formatted data and hyperparameters for use in RenewalCoalescentModel.model
        """
        tree_t_min, tree_t_max, init_growth_steps = (
            RenewalCoalescentModel.grid_helper(
                coalescent_times,
                sampling_times,
                reversed_infectiousness_profile,
                generation_interval,
            )
        )
        n_weeks = np.ceil((tree_t_max + init_growth_steps) / 7).astype(int)
        renewal_t_max = tree_t_max + init_growth_steps

        rate_grid = np.arange(tree_t_min + 1, tree_t_max + 1)
        coal_data = CoalescentData(coalescent_times, sampling_times, rate_grid)

        return CoalescentData.likelihood_relevant_intervals(coal_data), {
            "init_growth_steps": init_growth_steps,
            "n_weeks": n_weeks,
            "renewal_t_max": renewal_t_max,
        }

    @staticmethod
    @np.errstate(divide="ignore")
    def simulate_approx_coalescent_times(
        sampling_times: NDArray,
        rate_shift_times: NDArray,
        force_of_infection: NDArray,
        approx_squared_prevalence: NDArray,
        rng=np.random.default_rng(),
    ) -> CoalescentData:
        """
        Simulates coalescent times given sampling times, rate shit times,
        the piecewise constant force of infection, and the piecewise constant prevalence.
        """
        assert (
            rate_shift_times.shape[0] == force_of_infection.shape[0] - 1
        ), f"There are {rate_shift_times.shape[0]} rate shift times, expected {rate_shift_times.shape[0] + 1} `force_of_infection` and `prevalence entries`."
        assert (
            force_of_infection.shape[0] == approx_squared_prevalence.shape[0]
        ), f"Provided force_of_infection is length {force_of_infection.shape[0]} while provided prevalence is length {approx_squared_prevalence.shape[0]}"
        rate_times = np.concat(
            (
                np.sort(rate_shift_times),
                [np.inf],
            )
        )
        samp_times = np.concat(
            (
                np.sort(sampling_times),
                [np.inf],
            )
        )

        n_coal = sampling_times.shape[0] - 1
        time = 0.0
        rate_idx = 0
        sample_idx = 0
        n_active = 0
        coalescent_times = []
        rate_inv = 1.0 / RenewalCoalescentModel.approx_coalescent_rate(
            approx_squared_prevalence[rate_idx],
            force_of_infection[rate_idx],
            choose2(n_active),
        )
        while len(coalescent_times) < n_coal:
            wt = rng.exponential(rate_inv)

            if time + wt > samp_times[sample_idx]:
                time = samp_times[sample_idx]
                sample_idx += 1
                n_active += 1
            elif time + wt > rate_times[rate_idx]:
                time = rate_times[rate_idx]
                rate_idx += 1
            else:
                time += wt
                coalescent_times.append(time)
                n_active -= 1
            rate_inv = 1.0 / RenewalCoalescentModel.approx_coalescent_rate(
                approx_squared_prevalence[rate_idx],
                force_of_infection[rate_idx],
                choose2(n_active),
            )

        return CoalescentData(
            np.array(coalescent_times),
            sampling_times,
            rate_shift_times,
            intervals=None,
        )

    #################
    # Renewal model #
    #################

    @staticmethod
    def daily_incidence(
        rt,
        reversed_infectiousness_profile,
        i0,
        init_growth_rate,
        init_growth_steps,
    ):
        """
        Generates daily incident infection time series from daily Rt time series (rt) and reversed infectiousness profile PMF (g).
        Initializes by starting with i0 incident infections and exponential growth for init_growth_steps
        steps at rate init_growth_rate.
        Note: returns the whole time series including initialization/burnin.
        """
        carry_len = len(reversed_infectiousness_profile + 1)
        init = i0 * jnp.exp(jnp.arange(init_growth_steps) * init_growth_rate)
        scan_init = init[-carry_len:]

        def scan_renewal(prev_incident, r):
            incident = (
                r
                * (prev_incident[-1:] * reversed_infectiousness_profile).sum()
            )
            return jnp.concat(
                (prev_incident[1:], jnp.array([incident]))
            ), incident

        _, post_init = jscan(scan_renewal, scan_init, xs=rt)
        return jnp.concat((init, post_init))

    @staticmethod
    def daily_prevalence(incidence, generation_interval):
        """
        Assuming all infections last for duration generation_interval, get daily prevalence from daily incidence
        """
        gen_vec = np.repeat(1.0, generation_interval)
        return jnp.convolve(incidence, gen_vec, mode="valid")

    @staticmethod
    def daily_rt(n_weeks, n_days):
        """
        A GMRF on the weekly scale, reported as daily scale.
        """
        z_log_rt = numpyro.sample(
            "z_log_rt",
            dist.Normal(),
            sample_shape=(n_weeks,),
        )
        # somewhat arbitrary attempt to say differences shouldn't be massive
        sigma_log_rt = numpyro.sample(
            "sigma_log_rt", dist.Exponential(rate=n_weeks)
        )
        weekly_rt = numpyro.deterministic(
            "weekly_rt", jnp.exp(jnp.cumsum(z_log_rt) * sigma_log_rt)
        )

        return jnp.repeat(weekly_rt, 7)[:n_days]

    @staticmethod
    def i0():
        """
        Almost a 1/x prior on the initial number infected.
        """
        return numpyro.sample("i0", dist.Pareto(scale=1e-3, alpha=1e-3))

    @staticmethod
    def exp_growth_rate():
        """
        A prior on relatively constant dynamics in the burin phase of the renewal model.
        """
        return numpyro.sample("exp_growth_rate", dist.Normal(0.0, 0.01))
