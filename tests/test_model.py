import jax
import numpy as np
import numpyro

import lag


def test_model_runs():
    samps = np.arange(14, 28)
    coals = samps[1:] + 1.5

    rev_inf_prof = np.ones(5) / 5.0
    gen_int = 5
    intervals, init_growth_steps, n_weeks, renewal_t_max = lag.preprocess(
        coals, samps, rev_inf_prof, gen_int
    )

    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)

    kernel = numpyro.infer.NUTS(lag.renewal_coalescent_model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=1, num_samples=1)
    mcmc.run(
        rng_key_,
        intervals=intervals,
        init_growth_steps=init_growth_steps,
        n_weeks=n_weeks,
        renewal_t_max=renewal_t_max,
        rev_inf_prof=rev_inf_prof,
        gen_int=gen_int,
    )
