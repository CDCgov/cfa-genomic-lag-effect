import json
import os

import matplotlib.pyplot as plt
import numpy as np

from pipeline.utils import parser, read_config


def count_events(n_days, key, scenario, i0, scaling_factor, n_rep):
    grid = np.arange(n_days)
    counts = np.zeros((n_rep, n_days - 1))
    for rep in range(n_rep):
        with open(
            os.path.join(
                "pipeline",
                "output",
                "coalescent",
                f"{scenario}_{i0}_{scaling_factor}_{rep}.json",
            ),
            "r",
        ) as infile:
            data = json.load(infile)[key]
        counts[rep, :] = np.histogram(data, bins=grid)[0]

    return counts.sum(axis=0) / n_rep


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    n_rep = config["simulations"]["n_rep"]

    with open(args.infile[0], "r") as infile:
        infections = json.load(infile)
        incidence = np.array(infections["incidence"])
        prevalence = np.array(infections["prevalence"])

    incidence = np.flip(incidence)[: prevalence.shape[0]]
    prevalence = np.flip(prevalence)

    n_days = prevalence.shape[0] - 2

    coalescent_counts = count_events(
        n_days,
        "coalescent_times",
        args.scenario,
        args.i0,
        args.scaling_factor,
        n_rep,
    )
    sampling_counts = count_events(
        n_days,
        "sampling_times",
        args.scenario,
        args.i0,
        args.scaling_factor,
        n_rep,
    )

    fig, axs = plt.subplots(4, 1, figsize=(6, 12))

    axs[0].scatter(np.arange(n_days), incidence[:n_days], marker=".")
    axs[0].set_ylabel("Incidence")

    axs[1].plot(np.arange(n_days), prevalence[:n_days])
    axs[1].set_ylabel("Prevalence")

    axs[2].scatter(np.arange(n_days - 1), coalescent_counts, marker=".")
    axs[2].set_ylabel("Mean daily coalescents")

    axs[3].scatter(np.arange(n_days - 1), sampling_counts, marker=".")
    axs[3].set_ylabel("Mean daily samples")

    axs[3].set_xlabel("Time before present (in days)")

    plt.savefig(args.outfile[0])
