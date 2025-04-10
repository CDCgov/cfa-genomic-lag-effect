import argparse
import json

import numpy as np

from lag.models import RenewalCoalescentModel

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

    rev_inf_prof = np.flip(config["renewal"]["infectious_profile"])
    gen_int = len(rev_inf_prof)

    for scenario in config["simulations"]["rt_scenarios"]:
        weekly_rt = np.loadtxt(f"pipeline/out/rt/{scenario}.txt")
        rt = np.repeat(weekly_rt, 7)

        for i0 in config["simulations"]["i0"]:
            incidence = RenewalCoalescentModel.daily_incidence(
                rt,
                rev_inf_prof,
                i0,
                config["renewal"]["init_growth_rate"],
                config["renewal"]["init_growth_steps"],
            )
            prevalence = RenewalCoalescentModel.daily_prevalence(
                incidence, gen_int
            )

            with open(
                f"pipeline/out/infections/incidence_{scenario}_{i0}.txt", "w"
            ) as outfile:
                outfile.write("\n".join([str(i) for i in incidence]))

            with open(
                f"pipeline/out/infections/prevalence_{scenario}_{i0}.txt", "w"
            ) as outfile:
                outfile.write("\n".join([str(p) for p in prevalence]))
