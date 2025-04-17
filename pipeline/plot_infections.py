import json

import matplotlib.pyplot as plt
import numpy as np

from pipeline.utils import parser

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.infile[0], "r") as infile:
        infections = json.load(infile)
        incidence = np.array(infections["incidence"])
        prevalence = np.array(infections["prevalence"])

    n_days = prevalence.shape[0]
    incidence = incidence[-n_days:]
    time = np.arange(n_days)

    fig, axs = plt.subplots(2, 1)
    axs[0].scatter(time, incidence, marker=".")
    axs[0].set_ylabel("Incidence")
    axs[1].plot(time, prevalence)
    axs[1].set_xlabel("Time (in days)")
    axs[1].set_ylabel("Prevalence")

    plt.savefig(args.outfile)
