import numpy as np

from lag.models import RenewalCoalescentModel
from pipeline.utils import parser, read_config

if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    gen_int = len(config["renewal"]["infectious_profile"])

    incidence = np.loadtxt(args.infile[0])

    prevalence = RenewalCoalescentModel.daily_prevalence(incidence, gen_int)

    with open(
        args.outfile,
        "w",
    ) as outfile:
        outfile.write("\n".join([str(p) for p in prevalence]))
