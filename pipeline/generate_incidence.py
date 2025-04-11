import numpy as np

from lag.models import RenewalCoalescentModel

rev_inf_prof = np.flip(snakemake.params.infectious_profile)  # noqa: F821 # type: ignore
gen_int = len(rev_inf_prof)

weekly_rt = np.loadtxt(snakemake.input[0])  # noqa: F821 # type: ignore
rt = np.repeat(weekly_rt, 7)

scenario = snakemake.wildcards.scenario  # noqa: F821 # type: ignore
i0 = snakemake.wildcards.i0  # noqa: F821 # type: ignore

incidence = RenewalCoalescentModel.daily_incidence(
    rt,
    rev_inf_prof,
    int(i0),
    snakemake.params.init_growth_rate,  # noqa: F821 # type: ignore
    snakemake.params.init_growth_steps,  # noqa: F821 # type: ignore
)

with open(snakemake.output[0], "w") as outfile:  # noqa: F821 # type: ignore
    outfile.write("\n".join([str(i) for i in incidence]))
