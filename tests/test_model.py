import jax
import numpy as np

import lag.data as data
from lag.models import RenewalCoalescentModel


def test_lnl_runs():
    samps = np.arange(5, 10) + 0.5
    coals = samps[1:] + 0.25
    rate_grid = np.arange(19)
    intervals = data.CoalescentData(coals, samps, rate_grid)
    _ = RenewalCoalescentModel.log_likelihood(
        intervals, np.arange(1, 21), np.arange(1, 21)
    )


def test_offset_same_loglik():
    samps = np.arange(5, 10) + 0.5
    coals = samps[1:] + 0.25

    rate_grid = np.arange(19)
    foi = np.arange(1, 21)
    prevalence = np.arange(1, 21)

    coal_data = data.CoalescentData(coals, samps, rate_grid)
    print(coal_data.intervals)

    lnl_no_offset = float(
        RenewalCoalescentModel.log_likelihood(coal_data, foi, prevalence)
    )

    nondeterministic = data.CoalescentData.likelihood_components(coal_data)
    assert nondeterministic.dt.shape[0] < coal_data.dt.shape[0]
    lnl_offset = float(
        RenewalCoalescentModel.log_likelihood(
            nondeterministic, foi, prevalence
        )
    )

    assert lnl_no_offset == lnl_offset


def test_model_runs():
    samps = np.arange(14, 28)
    coals = samps[1:] + 1.5

    reversed_infectiousness_profile = np.ones(5) / 5.0
    generation_interval = 5
    intervals, par = RenewalCoalescentModel.preprocess_from_vectors(
        coals, samps, reversed_infectiousness_profile, generation_interval
    )

    rng_key = jax.random.key(0)

    RenewalCoalescentModel.fit(
        data=intervals,
        hyperparameters=par
        | {
            "reversed_infectiousness_profile": reversed_infectiousness_profile,
            "generation_interval": generation_interval,
        },
        rng_key=rng_key,
        mcmc_config={"num_warmup": 1, "num_samples": 1},
    )


def test_sim_runs():
    _ = RenewalCoalescentModel.simulate_coalescent_times(
        sampling_times=np.arange(5, 10) + 0.5,
        rate_shift_times=np.arange(19),
        force_of_infection=np.array([1] * 20),
        prevalence=np.arange(1, 21),
        rng=np.random.default_rng(0),
    )
