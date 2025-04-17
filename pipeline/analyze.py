import json

import arviz as az
import jax
import numpy as np
import numpyro

from lag.models import RenewalCoalescentModel
from pipeline.utils import parser, read_config

if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    config["bayes"]["nuts"]["init_strategy"] = getattr(
        numpyro.infer, "init_to_sample"
    )

    numpyro.set_host_device_count(config["bayes"]["cores"])

    with open(args.infile[0], "r") as file:
        coal_dict = json.load(file)

    rev_inf_prof = np.flip(config["renewal"]["infectious_profile"])
    gen_int = len(rev_inf_prof)

    true_weekly_rt = np.flip(np.loadtxt(args.infile[1]))

    intervals, par = RenewalCoalescentModel.preprocess_from_vectors(
        np.array(coal_dict["coalescent_times"]),
        np.array(coal_dict["sampling_times"]),
        rev_inf_prof,
        gen_int,
    )

    rng_key = jax.random.key(0)

    mcmc = RenewalCoalescentModel.fit(
        data=intervals,
        hyperparameters=par
        | {
            "reversed_infectiousness_profile": rev_inf_prof,
            "generation_interval": gen_int,
        },
        rng_key=rng_key,
        mcmc_config=config["bayes"]["mcmc"],
        nuts_config=config["bayes"]["nuts"],
    )

    inference = az.from_numpyro(
        mcmc,
        prior=None,
        posterior_predictive=None,
    )

    summary = az.summary(inference)

    posterior_weekly_rt = az.extract(
        inference, "posterior", var_names=["weekly_rt"]
    )
    mse_weekly_rt = np.pow(
        posterior_weekly_rt.T - true_weekly_rt[: posterior_weekly_rt.shape[0]],
        2,
    ).mean(axis=0)
    alpha = config["bayes"]["ci_alpha"]
    results = {
        "convergence": {
            "min_ess": float(summary["ess_bulk"].min()),
            "max_psrf": float(summary["r_hat"].max()),
        },
        "rt_est": {
            "lower": np.quantile(
                posterior_weekly_rt, alpha / 2.0, axis=1
            ).tolist(),
            "point": np.median(posterior_weekly_rt, axis=1).tolist(),
            "upper": np.quantile(
                posterior_weekly_rt, 1.0 - alpha / 2.0, axis=1
            ).tolist(),
        },
        "rt_error": mse_weekly_rt.to_numpy().tolist(),
    }

    with open(
        args.outfile,
        "w",
    ) as outfile:
        json.dump(results, outfile)
