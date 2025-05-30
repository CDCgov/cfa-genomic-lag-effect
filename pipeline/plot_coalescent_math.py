import json

import matplotlib.pyplot as plt
import numpy as np

from lag.models import RenewalCoalescentModel
from pipeline.utils import parser, read_config


def stepwise_coal_rate(tau, pieces):
    return pieces[np.floor(tau).astype(int)]


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    gen_int = len(config["renewal"]["infectious_profile"])

    with open(args.infile[0], "r") as infile:
        infections = json.load(infile)
        incidence = np.array(infections["incidence"])
        prevalence = np.array(infections["prevalence"])

    n_days = prevalence.shape[0] - 2

    incidence = np.flip(incidence[:-1])[: prevalence.shape[0]]
    prevalence = np.flip(prevalence)

    fine_time = np.linspace(0.0, n_days, 2500)

    unit_coal_rate = RenewalCoalescentModel.approx_coalescent_rate(
        prevalence,
        incidence,
        np.repeat(1, incidence.shape[0]),
    )

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    axs[0].scatter(np.arange(n_days), incidence[:n_days], marker=".")
    axs[0].set_ylabel("Incidence")

    axs[1].plot(np.arange(n_days), prevalence[:n_days])
    axs[1].set_ylabel("Prevalence")

    axs[2].plot(
        fine_time, stepwise_coal_rate(fine_time, unit_coal_rate), marker="."
    )
    axs[2].set_ylabel("Unit coalescent rate")

    axs[2].set_xlabel("Time before present (in days)")

    plt.savefig(args.outfile[0])
