import argparse
import json

import numpy as np
from numpy.typing import NDArray


def ar1(
    mu: NDArray,
    sd: float,
    ac: float,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Draw from an AR1 process with mean vector mu,
    standard deviation sd, and autocorrelation ac.
    """
    n = mu.shape[0]
    z = rng.normal(loc=0.0, scale=1.0, size=n) * sd
    x = np.zeros(n)
    x[0] = z[0]
    for i in range(1, n):
        x[i] = x[i - 1] * ac + z[i]
    return np.exp(np.log(mu) + x)


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

    n_weeks = config["simulations"]["n_weeks"]
    r_low = config["simulations"]["r_low"]
    r_high = config["simulations"]["r_high"]
    r_const = config["simulations"]["r_const"]
    rt_scenarios = {
        "decreasing": np.linspace(r_high, r_low, n_weeks),
        "constant": np.array([r_const] * n_weeks),
        "increasing": np.linspace(r_low, r_high, n_weeks),
    }

    sd = config["simulations"]["r_sd"]
    ac = config["simulations"]["r_ac"]
    rng = np.random.default_rng(config["seed"])
    for scenario, mean in rt_scenarios.items():
        with open(f"pipeline/out/rt/{scenario}.txt", "w") as outfile:
            outfile.write("\n".join([str(rt) for rt in ar1(mean, sd, ac)]))
