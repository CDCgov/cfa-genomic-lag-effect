import numpy as np

from lag.models import RenewalCoalescentModel

gen_int = len(snakemake.params.infectious_profile)  # noqa: F821 # type: ignore

incidence = np.loadtxt(snakemake.input[0])  # noqa: F821 # type: ignore

prevalence = RenewalCoalescentModel.daily_prevalence(incidence, gen_int)

with open(
    snakemake.output[0],  # noqa: F821 # type: ignore
    "w",
) as outfile:
    outfile.write("\n".join([str(p) for p in prevalence]))
