import numpy as np

from pipeline.utils import generate_rt_scenario

rt_scenarios = {
    "decreasing": (
        snakemake.params.r_med,  # type: ignore  # noqa: F821
        snakemake.params.r_low,  # type: ignore  # noqa: F821
    ),
    "constant": (
        snakemake.params.r_med,  # type: ignore  # noqa: F821
        snakemake.params.r_med,  # type: ignore  # noqa: F821
    ),
    "increasing": (
        snakemake.params.r_med,  # type: ignore  # noqa: F821
        snakemake.params.r_high,  # type: ignore  # noqa: F821
    ),
}

rng = np.random.default_rng(snakemake.params.seed)  # type: ignore  # noqa: F821
for scenario, rtup in rt_scenarios.items():
    with open(f"pipeline/out/rt/{scenario}.txt", "w") as outfile:
        outfile.write(
            "\n".join(
                [
                    str(rt)
                    for rt in generate_rt_scenario(
                        rtup[0],
                        rtup[1],
                        snakemake.params.n_init_weeks,  # type: ignore  # noqa: F821
                        snakemake.params.n_change_weeks,  # type: ignore  # noqa: F821
                        snakemake.params.sd,  # type: ignore  # noqa: F821
                        snakemake.params.ac,  # type: ignore  # noqa: F821
                        rng,
                    )  # type: ignore  # noqa: F821
                ]
            )
        )
