import argparse
import json

import numpy as np
from numpy.typing import NDArray

from lag.models import RenewalCoalescentModel
from pipeline.bhsqi import BHSQI


class LagSampler:
    def __init__(self, list_params):
        if list_params == {}:
            self.bhsqi = None
        else:
            self.bhsqi = BHSQI(
                np.array(params["knots"]), np.array(params["coef"])
            )

    def draw(self, size: int, rng: np.random.Generator) -> NDArray:
        if self.bhsqi is None:
            return np.array([0.0] * size)
        else:
            return self.bhsqi.draw(size=size, rng=rng)


def simulate_sampling_times(
    weekday_effect, n_sampled_weeks, n_samples, rng: np.random.Generator
) -> NDArray:
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
    parser = argparse.ArgumentParser(
        description="Simulate and store 3 Rt trends over time."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file",
    )

    with open(parser.parse_args().config, "r") as file:
        config = json.load(file)

    rng = np.random.default_rng(config["seed"])
    n_samples = config["simulations"]["sampling"]["n_samples"]

    rate_shift_times = np.arange(1, config["simulations"]["n_weeks"] * 7)
    n_days = rate_shift_times.shape[0] + 1

    for scenario in config["simulations"]["rt_scenarios"]:
        weekly_rt = np.loadtxt(f"pipeline/out/rt/{scenario}.txt")
        rt = np.repeat(weekly_rt, 7)

        for i0 in config["simulations"]["i0"]:
            with open(
                f"pipeline/out/infections/incidence_{scenario}_{i0}.txt", "r"
            ) as file:
                backwards_incidence = np.flip(np.loadtxt(file))[:n_days]

            with open(
                f"pipeline/out/infections/prevalence_{scenario}_{i0}.txt", "r"
            ) as file:
                backwards_prevalence = np.flip(np.loadtxt(file))[:n_days]

            for lag_scale in config["empirical_lag"]["scaling_factors"]:
                with open(f"pipeline/out/lags/{lag_scale}.json", "r") as file:
                    params = json.load(file)

                lag_sampler = LagSampler(params)

                for i in range(config["simulations"]["n_rep"]):
                    samp_times = simulate_sampling_times(
                        config["simulations"]["sampling"]["weekday_effect"],
                        config["simulations"]["sampling"]["n_sampled_weeks"],
                        n_samples,
                        rng,
                    )
                    unlagged_data = (
                        RenewalCoalescentModel.simulate_coalescent_times(
                            samp_times,
                            rate_shift_times,
                            backwards_incidence,
                            backwards_prevalence,
                        )
                    )
                    lags = lag_sampler.draw(n_samples, rng)
                    data = unlagged_data.as_of(as_of=0.0, lags=lags)

                    print("+")
                    with open(
                        f"pipeline/out/coalescent/{scenario}_{i0}_{lag_scale}_{i}.json",
                        "w",
                    ) as outfile:
                        json.dump(data.serialize(), outfile)
