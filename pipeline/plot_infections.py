import matplotlib.pyplot as plt
import numpy as np

from pipeline.utils import parser, read_config

if __name__ == "__main__":
    args = parser.parse_args()

    config = read_config(args.config)

    n_days = 7 * (
        config["simulations"]["n_init_weeks"]
        + config["simulations"]["n_change_weeks"]
    )

    incidence = np.loadtxt(args.infile[0])[-n_days:]
    prevalence = np.loadtxt(args.infile[1])[-n_days:]
    time = np.arange(n_days)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(time, incidence)
    axs[0].set_ylabel("Incidence")
    axs[1].plot(time, prevalence)
    axs[1].set_xlabel("Time (in days)")
    axs[1].set_ylabel("Prevalence")

    plt.savefig(args.outfile)
