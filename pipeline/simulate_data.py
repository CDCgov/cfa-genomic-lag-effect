import json

import numpy as np

from lag.models import RenewalCoalescentModel
from pipeline.fit_lag import BHSQI
from pipeline.utils import construct_seed, parser, read_config


def simulate_sampling_times(
    weekday_effect, n_sampled_weeks, n_samples, rng: np.random.Generator
) -> np.typing.NDArray:
    """
    Samples come in uniformly during a day, at different rates per day of week, over a range of weeks before present day.
    """
    n_days = n_sampled_weeks * 7

    probs = np.tile(weekday_effect, n_sampled_weeks)
    probs = probs / probs.sum()

    backwards_times = n_days - (
        rng.choice(n_days, size=n_samples, replace=True)
        + rng.uniform(0.0, 1.0, size=n_samples)
    )

    return backwards_times


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    lag_scale = float(args.scaling_factor)

    seed = construct_seed(
        config["seed"],
        args.scenario,
        args.i0,
        args.scaling_factor,
        args.rep,
    )
    rng = np.random.default_rng(seed)

    with open(args.infile[0], "r") as infile:
        infections = json.load(infile)
        incidence = np.array(infections["incidence"])
        prevalence = np.array(infections["prevalence"])

    n_days = prevalence.shape[0] - 1
    rate_shift_times = np.arange(1, n_days)

    backwards_incidence = np.flip(incidence)[:n_days]
    backwards_prevalence = np.flip(prevalence)[:n_days]

    with open(args.infile[1], "r") as file:
        lag_params = json.load(file)

    lag_dist = BHSQI(**{k: np.array(v) for k, v in lag_params.items()})

    samp_times = simulate_sampling_times(
        config["simulations"]["sampling"]["weekday_effect"],
        config["simulations"]["sampling"]["n_sampled_weeks"],
        config["simulations"]["sampling"]["n_samples"],
        rng,
    )
    unlagged_data = RenewalCoalescentModel.simulate_coalescent_times(
        samp_times,
        rate_shift_times,
        backwards_incidence,
        backwards_prevalence,
    )
    lags = lag_dist.draw(
        size=config["simulations"]["sampling"]["n_samples"],
        scale=lag_scale,
        rng=rng,
    )
    data = unlagged_data.as_of(as_of=0.0, lags=lags)

    with open(
        args.outfile,
        "w",
    ) as outfile:
        json.dump(data.serialize(), outfile)
