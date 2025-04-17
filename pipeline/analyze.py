import json

import arviz as az
import jax
import numpy as np
import numpyro
import polars as pl

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

    posterior_weekly_rt = (
        pl.from_pandas(
            az.extract(
                inference, "posterior", var_names=["weekly_rt"]
            ).to_pandas()
        )
        .with_columns(week=pl.int_range(pl.len()))
        .unpivot(index="week", variable_name="chain_sample", value_name="Rt")
        .with_columns(
            chain=pl.col("chain_sample")
            .str.extract(r"(\d+), (\d+)", 1)
            .cast(pl.Int64),
            sample=pl.col("chain_sample")
            .str.extract(r"(\d+), (\d+)", 2)
            .cast(pl.Int64),
        )
        .drop("chain_sample")
    )

    posterior_weekly_rt.write_parquet(args.outfile[0])

    convergence = summary[["ess_bulk", "ess_tail", "r_hat"]].to_dict()
    if not config["bayes"]["convergence_report_all"]:
        convergence = {
            "ess_bulk": min(list(convergence["ess_bulk"].values())),
            "ess_tail": min(list(convergence["ess_tail"].values())),
            "r_hat": max(list(convergence["r_hat"].values())),
        }

    with open(
        args.outfile[1],
        "w",
    ) as outfile:
        json.dump(convergence, outfile)
