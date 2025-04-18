import itertools
import json
import os

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pipeline.utils import parser, read_config


def mse_component(est: pl.Expr, true: pl.Expr) -> pl.Expr:
    return (est - true) ** 2


def identity_component(est: pl.Expr, _: pl.Expr) -> pl.Expr:
    return est


def expr_mean(x: pl.Expr) -> pl.Expr:
    return x.mean()


def expr_median(x: pl.Expr) -> pl.Expr:
    return x.quantile(0.5)


def get_viridis_colors(x):
    values = np.array(x)
    norm = matplotlib.colors.Normalize(values.min(), values.max())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(norm(value)) for value in values]
    hex_colors = [
        matplotlib.colors.rgb2hex(
            color[:3],
        )
        for color in colors
    ]
    return hex_colors


# TODO: should this be stairs, not step?
def plot_median_rt(df: pl.DataFrame, alpha_hex="66"):
    rt_est = "Posterior median Rt"
    scales = df["scaling_factor"].unique().sort()
    colors = [col + alpha_hex for col in get_viridis_colors(scales)]
    color_map = {sf: col for sf, col in zip(scales, colors)}
    _, axs = plt.subplots(3, 1, figsize=(9, 6))
    for ax, scenario in zip(axs, df["scenario"].unique().sort()):
        scenario_df = df.filter(pl.col("scenario") == scenario)
        for scaling_factor in df["scaling_factor"].unique().sort():
            for rep in df["rep"].unique().sort():
                week_rt = (
                    scenario_df.filter(pl.col("rep") == rep)
                    .select(["week", rt_est])
                    .unique()
                    .sort("week")
                )
                ax.step(
                    week_rt["week"].to_numpy(),
                    week_rt[rt_est].to_numpy(),
                    color_map[scaling_factor],
                )
        week_rt = scenario_df.select(["week", "true_Rt"]).unique().sort("week")
        ax.step(
            week_rt["week"].to_numpy(), week_rt["true_Rt"].to_numpy(), "black"
        )
    plt.savefig(os.path.join("pipeline", "output", "rt_est.png"))


def plot_mse_rt(df: pl.DataFrame, alpha_hex="66"):
    error = "Posterior MSE Rt"
    scales = df["scaling_factor"].unique().sort()
    colors = [col + alpha_hex for col in get_viridis_colors(scales)]
    color_map = {sf: col for sf, col in zip(scales, colors)}
    _, axs = plt.subplots(3, 1, figsize=(9, 6))
    for ax, scenario in zip(axs, df["scenario"].unique().sort()):
        scenario_df = df.filter(pl.col("scenario") == scenario)
        for scaling_factor in df["scaling_factor"].unique().sort():
            for rep in df["rep"].unique().sort():
                week_rt = (
                    scenario_df.filter(pl.col("rep") == rep)
                    .select(["week", error])
                    .unique()
                    .sort("week")
                )
                ax.step(
                    week_rt["week"].to_numpy(),
                    week_rt[error].to_numpy(),
                    color_map[scaling_factor],
                )
    plt.savefig(os.path.join("pipeline", "output", "rt_error.png"))


posterior_summaries = {
    "comp_funs": [identity_component, mse_component],
    "agg_funs": [expr_median, expr_mean],
    "names": ["Posterior median Rt", "Posterior MSE Rt"],
    "plot_funs": [plot_median_rt, plot_mse_rt],
}


def true_rt_df(scenario):
    rt_fwd = np.loadtxt(
        os.path.join("pipeline", "input", "rt", f"{scenario}.txt")
    )
    return pl.DataFrame(
        {"week": np.arange(rt_fwd.shape[0]), "true_Rt": np.flip(rt_fwd)}
    )


def summarize_replicate_rt(
    df,
    true_rt,
    comp_funs=posterior_summaries["comp_funs"],
    agg_funs=posterior_summaries["agg_funs"],
    names=posterior_summaries["names"],
    marginalize=["chain", "sample"],
):
    df = df.join(true_rt, on=["week"], validate="m:1")
    assert set(df.columns) == {"week", "chain", "sample", "Rt", "true_Rt"}
    df = df.with_columns(
        [
            comp_funs[i](pl.col("Rt"), pl.col("true_Rt")).alias(names[i])
            for i in range(len(comp_funs))
        ]
    )

    groups = {"week", "chain", "sample"}.difference(marginalize)
    if groups:
        df = df.group_by(groups)
        df = df.agg(
            [agg_funs[i](pl.col(names[i])) for i in range(len(comp_funs))]
        )
    else:
        df.select(
            [agg_funs[i](pl.col(names[i])) for i in range(len(comp_funs))]
        )

    return df


def aggregate_rt(scenario, i0_vec, scaling_factor_vec, n_rep):
    true_rt = true_rt_df(scenario)

    return pl.concat(
        [
            pl.concat(
                [
                    summarize_replicate_rt(
                        pl.read_parquet(
                            os.path.join(
                                "pipeline",
                                "output",
                                "analysis",
                                f"rt_{scenario}_{i0}_{scaling_factor}_{rep}.parquet",
                            )
                        ),
                        true_rt,
                    ).with_columns(rep=pl.lit(rep))
                    for rep in range(n_rep)
                ]
            ).with_columns(
                i0=pl.lit(int(i0)),
                scaling_factor=pl.lit(float(scaling_factor)),
            )
            for i0, scaling_factor in itertools.product(
                i0_vec, scaling_factor_vec
            )
        ]
    )


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    with open(
        os.path.join("pipeline", "output", "rt", "hash.json"), "r"
    ) as file:
        scenarios = json.load(file).keys()

    df = pl.concat(
        [
            aggregate_rt(
                scenario,
                config["simulations"]["i0"],
                config["empirical_lag"]["scaling_factors"],
                config["simulations"]["n_rep"],
            )
            .with_columns(scenario=pl.lit(scenario))
            .join(true_rt_df(scenario), on=["week"], validate="m:1")
            for scenario in scenarios
        ]
    )

    df.write_parquet(os.path.join("pipeline", "output", "results.parquet"))

    for fun in posterior_summaries["plot_funs"]:
        fun(df)
