import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.lax import scan as jscan


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
            r * (prev_incident[-1:] * reversed_infectiousness_profile).sum()
        )
        return jnp.concat((prev_incident[1:], jnp.array([incident]))), incident

    _, post_init = jscan(scan_renewal, scan_init, xs=rt)
    return jnp.concat((init, post_init))


def daily_prevalence(incidence, generation_interval):
    """
    Assuming all infections last for duration generation_interval, get daily prevalence from daily incidence
    """
    gen_vec = np.repeat(1.0, generation_interval)
    return jnp.convolve(incidence, gen_vec, mode="valid")


#####
# Priors
#####


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


def i0():
    """
    Almost a 1/x prior on the initial number infected.
    """
    return numpyro.sample("i0", dist.Pareto(scale=1e-3, alpha=1e-3))


def exp_growth_rate():
    """
    A prior on relatively constant dynamics in the burin phase of the renewal model.
    """
    return numpyro.sample("exp_growth_rate", dist.Normal(0.0, 0.01))
