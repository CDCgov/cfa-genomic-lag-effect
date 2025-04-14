import json

import matplotlib.pyplot as plt
import numpy as np

from pipeline.fit_lag import BHSQI, get_empirical_lag
from pipeline.utils import construct_seed, parser, read_config

if __name__ == "__main__":
    args = parser.parse_args()

    config = read_config(args.config)

    with open(args.infile[0], "r") as file:
        lag_params = json.load(file)

    empirical_lags = get_empirical_lag(config)
    approx_lag_dist = BHSQI(
        knots=np.array(lag_params["knots"]), coef=np.array(lag_params["coef"])
    )
    lag_scales = config["empirical_lag"]["scaling_factors"]

    q_vec = np.arange(1.0, 2499.0) / 2500.0

    scale = float(args.scaling_factor)
    rng = np.random.default_rng(
        construct_seed(
            config["seed"],
            scenario=None,
            i0=None,
            scaling_factor=args.scaling_factor,
            rep=None,
        )
    )

    approx_samples = approx_lag_dist.draw(size=100000, scale=scale, rng=rng)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # Approximate PDF (meaningful range)
    t_pdf = np.linspace(
        0.0, np.quantile(empirical_lags * scale, np.array([0.95])), 2500
    )
    axs[0, 0].plot(t_pdf, approx_lag_dist.pdf(t_pdf, scale=scale))
    axs[0, 0].set_ylabel("PDF")

    # Approximate CDF (whole range)
    t_cdf = np.linspace(0.0, approx_lag_dist.xmax, 2500)
    axs[1, 0].plot(t_cdf, approx_lag_dist.cdf(t_cdf, scale=scale))
    axs[1, 0].set_ylabel("CDF")
    axs[1, 0].set_xlabel("Time (in days)")

    # QQ plot of approximation vs empirical distribution
    approx_quants = np.array(
        [approx_lag_dist.quantile_function(p, scale=scale) for p in q_vec]
    )
    observed_quants = np.quantile(empirical_lags * scale, q_vec)
    axs[0, 1].plot(observed_quants, approx_quants)
    axs[0, 1].plot(
        observed_quants, observed_quants, ls="--"
    )  # a 1:1 line for comparison
    axs[0, 1].set_ylabel("Approximated quantiles")
    axs[0, 1].set_xlabel("Empirical quantiles")

    # Samples look like approximation they should be draws from
    approx_sample_quants = np.quantile(approx_samples, q_vec)

    axs[1, 1].plot(approx_quants, approx_sample_quants)
    axs[1, 1].plot(
        approx_quants, approx_quants, ls="--"
    )  # a 1:1 line for comparison
    axs[1, 1].set_ylabel("Approximated quantiles")
    axs[1, 1].set_xlabel("Approximation sample quantiles")

    plt.savefig(args.outfile)
