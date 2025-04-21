import json

import matplotlib.pyplot as plt
import numpy as np

from lag.models import RenewalCoalescentModel
from pipeline.utils import parser, read_config


def continuous_prevalence(tau, prevalence):
    floor_tau = np.floor(tau).astype(int)
    ceil_tau = np.where(
        tau - floor_tau == 0.0, floor_tau + 1, np.ceil(tau)
    ).astype(int)
    delta_p = prevalence[ceil_tau] - prevalence[floor_tau]
    return prevalence[floor_tau] + (tau - floor_tau) * delta_p


def approx_sq_prevalence(tau, prevalence):
    floor_tau = np.floor(tau).astype(int)
    return RenewalCoalescentModel.approx_squared_prevalence(prevalence)[
        floor_tau
    ]


def force_of_infection(tau, incidence):
    return incidence[np.floor(tau).astype(int)]


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    gen_int = len(config["renewal"]["infectious_profile"])

    with open(args.infile[0], "r") as infile:
        infections = json.load(infile)
        incidence = np.array(infections["incidence"])
        prevalence = np.array(infections["prevalence"])

    n_days = prevalence.shape[0] - 2

    incidence = np.flip(incidence)[: prevalence.shape[0]]
    prevalence = np.flip(prevalence)

    fine_time = np.linspace(0.0, n_days, 2500)
    approx_sq_p = approx_sq_prevalence(fine_time, prevalence)
    true_sq_p = continuous_prevalence(fine_time, prevalence) ** 2.0
    error = (approx_sq_p - true_sq_p) / true_sq_p

    unit_coal_rate = RenewalCoalescentModel.approx_coalescent_rate(
        RenewalCoalescentModel.approx_squared_prevalence(prevalence),
        incidence[:-1],
        np.repeat(1, incidence.shape[0]),
    )

    fig, axs = plt.subplots(4, 1, figsize=(6, 12))

    axs[0].scatter(np.arange(n_days), incidence[:n_days], marker=".")
    axs[0].set_ylabel("Incidence")

    axs[1].plot(np.arange(n_days), prevalence[:n_days])
    axs[1].set_ylabel("Prevalence")

    axs[2].scatter(fine_time, error, marker=".")
    axs[2].set_ylabel("Relative error of squared prevalence")

    axs[3].scatter(
        np.arange(unit_coal_rate.shape[0]), unit_coal_rate, marker="."
    )
    axs[3].set_ylabel("Unit coalescent rate")

    axs[3].set_xlabel("Time before present (in days)")

    plt.savefig(args.outfile[0])
