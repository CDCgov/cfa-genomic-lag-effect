import json

import numpy as np

from lag.models import RenewalCoalescentModel
from pipeline.utils import LagSampler, construct_seed, simulate_sampling_times

lag_scale = float(snakemake.wildcards.scaling_factor)  # type: ignore  # noqa: F821

seed = construct_seed(
    snakemake.params.seed,  # type: ignore  # noqa: F821
    snakemake.wildcards.scenario,  # type: ignore  # noqa: F821
    snakemake.wildcards.i0,  # type: ignore  # noqa: F821
    snakemake.wildcards.scaling_factor,  # type: ignore  # noqa: F821
    snakemake.wildcards.rep,  # type: ignore  # noqa: F821
)
rng = np.random.default_rng(seed)

rate_shift_times = np.arange(1, snakemake.params.n_weeks * 7)  # type: ignore  # noqa: F821
n_days = rate_shift_times.shape[0] + 1

backwards_incidence = np.flip(np.loadtxt(snakemake.input[0]))[:n_days]  # type: ignore  # noqa: F821
backwards_prevalence = np.flip(np.loadtxt(snakemake.input[1]))[:n_days]  # type: ignore  # noqa: F821

with open(snakemake.input[2], "r") as file:  # type: ignore  # noqa: F821
    lag_params = json.load(file)

lag_sampler = LagSampler(lag_params)

samp_times = simulate_sampling_times(
    snakemake.params.weekday_effect,  # type: ignore  # noqa: F821
    snakemake.params.n_sampled_weeks,  # type: ignore  # noqa: F821
    snakemake.params.n_samples,  # type: ignore  # noqa: F821
    rng,
)
unlagged_data = RenewalCoalescentModel.simulate_coalescent_times(
    samp_times,
    rate_shift_times,
    backwards_incidence,
    backwards_prevalence,
)
lags = lag_sampler.draw(snakemake.params.n_samples, rng)  # type: ignore  # noqa: F821
data = unlagged_data.as_of(as_of=0.0, lags=lags)

with open(
    snakemake.output[0],  # type: ignore  # noqa: F821
    "w",
) as outfile:
    json.dump(data.serialize(), outfile)
