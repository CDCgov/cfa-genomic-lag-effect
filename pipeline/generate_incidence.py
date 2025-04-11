import numpy as np

from lag.models import RenewalCoalescentModel
from pipeline.utils import parser, read_config

if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    rev_inf_prof = np.flip(config["renewal"]["infectious_profile"])
    gen_int = len(rev_inf_prof)

    weekly_rt = np.loadtxt(args.infile[0])
    rt = np.repeat(weekly_rt, 7)

    scenario = args.scenario
    i0 = args.i0

    incidence = RenewalCoalescentModel.daily_incidence(
        rt,
        rev_inf_prof,
        int(i0),
        config["renewal"]["init_growth_rate"],
        config["renewal"]["init_growth_steps"],
    )

    with open(args.outfile, "w") as outfile:
        outfile.write("\n".join([str(i) for i in incidence]))
