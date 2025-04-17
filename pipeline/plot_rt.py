import matplotlib.pyplot as plt
import numpy as np

from pipeline.utils import parser

if __name__ == "__main__":
    args = parser.parse_args()

    weekly_rt = np.loadtxt(args.infile[0])
    rt = np.repeat(weekly_rt, 7)
    time = np.arange(rt.shape[0])

    plt.plot(time, rt)
    plt.xlabel("Time (in days)")
    plt.ylabel("Rt")

    plt.savefig(args.outfile[0])
