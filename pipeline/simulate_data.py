import json

import numpy as np

from lag.models import RenewalCoalescentModel
from pipeline.fit_lags import BHSQI
from pipeline.utils import construct_seed, parser, read_config


class LagSampler:
    """
    Wrapper for BHSQI that allows 0 lag.
    """

    def __init__(self, list_params):
        if list_params == {}:
            self.bhsqi = None
        else:
            self.bhsqi = BHSQI(
                np.array(list_params["knots"]), np.array(list_params["coef"])
            )

    def draw(self, size: int, rng: np.random.Generator) -> np.typing.NDArray:
        if self.bhsqi is None:
            return np.array([0.0] * size)
        else:
            return self.bhsqi.draw(size=size, rng=rng)


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

    n_weeks = (
        config["simulations"]["n_init_weeks"]
        + config["simulations"]["n_change_weeks"]
    )
    rate_shift_times = np.arange(1, n_weeks * 7)
    n_days = rate_shift_times.shape[0] + 1

    backwards_incidence = np.flip(np.loadtxt(args.infile[0]))[:n_days]
    backwards_prevalence = np.flip(np.loadtxt(args.infile[1]))[:n_days]

    with open(args.infile[2], "r") as file:
        lag_params = json.load(file)

    lag_sampler = LagSampler(lag_params)

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
    lags = lag_sampler.draw(
        config["simulations"]["sampling"]["n_samples"], rng
    )
    data = unlagged_data.as_of(as_of=0.0, lags=lags)

    with open(
        args.outfile,
        "w",
    ) as outfile:
        json.dump(data.serialize(), outfile)
