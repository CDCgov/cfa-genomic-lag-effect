import json

import polars as pl

from pipeline.utils import BHSQI

date_l = pl.date(
    snakemake.params.date_lower["y"],  # type: ignore  # noqa: F821
    snakemake.params.date_lower["m"],  # type: ignore  # noqa: F821
    snakemake.params.date_lower["d"],  # type: ignore  # noqa: F821
)
date_u = pl.date(
    snakemake.params.date_upper["y"],  # type: ignore  # noqa: F821
    snakemake.params.date_upper["m"],  # type: ignore  # noqa: F821
    snakemake.params.date_upper["d"],  # type: ignore  # noqa: F821
)

df = (
    pl.scan_csv(snakemake.params.nextstrain_path, separator="\t")  # type: ignore  # noqa: F821
    .cast({"date": pl.Date, "date_submitted": pl.Date}, strict=False)
    .filter(
        pl.col("date").is_not_null(),
        pl.col("date") >= date_l,
        pl.col("date") < date_u,
        pl.col("date_submitted").is_not_null(),
        country="USA",
        host="Homo sapiens",
    )
    .with_columns(
        lag=(pl.col("date_submitted") - pl.col("date")).dt.total_days()
    )
    .collect()
)

low = (
    snakemake.params.lag_low  # type: ignore  # noqa: F821
    if snakemake.params.lag_low is not None  # type: ignore  # noqa: F821
    else df["lag"].min() - 1  # type: ignore
)
high = (
    snakemake.params.lag_high  # type: ignore  # noqa: F821
    if snakemake.params.lag_high is not None  # type: ignore  # noqa: F821
    else df["lag"].max() + 1  # type: ignore
)
lags = df["lag"].to_numpy()

scale = float(snakemake.wildcards.scaling_factor)  # type: ignore  # noqa: F821
print(f"Fitting lag scaled to {scale}")
if scale > 0.0:
    knots, coef = BHSQI.bshqi(
        samples=lags * scale,
        n_steps=500,
        low=low,  # type: ignore
        high=high,  # type: ignore
    )

    params = {
        "knots": knots.tolist(),
        "coef": coef.tolist(),
    }
else:
    params = {}

with open(f"pipeline/out/lags/{scale}.json", "w") as outfile:
    json.dump(params, outfile)
