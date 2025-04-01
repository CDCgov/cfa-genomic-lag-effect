import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import scripts.bhsqi

with open("scripts/config.json", "r") as file:
    config = json.load(file)

ns_path = config["nextstrain_path"]

df = (
    pl.scan_csv(ns_path, separator="\t")
    .cast({"date": pl.Date, "date_submitted": pl.Date}, strict=False)
    .filter(
        pl.col("date").is_not_null(),
        pl.col("date") >= pl.date(2020, 1, 1),
        pl.col("date") < pl.date(2025, 1, 1),
        pl.col("date_submitted").is_not_null(),
        country="USA",
        host="Homo sapiens",
    )
    .with_columns(
        lag=(pl.col("date_submitted") - pl.col("date")).dt.total_days()
    )
    .collect()
)

df["lag"].describe()

approx = scripts.bhsqi.BHSQI(
    samples=df["lag"].to_numpy(),
    n_steps=500,
    low=0.0,
    high=df["lag"].max() + 1,  # type: ignore
)

# Approximate PDF
t = np.arange(1723)
plt.plot(t, approx.pdf(t))
plt.show()

# Approximate CDF
t = np.arange(1723)
plt.plot(t, approx.cdf(t))
plt.show()

# Samples
approx_samples = approx.draw(10000)

approx_quants = np.quantile(approx_samples, np.arange(1, 100) / 100)
observed_quants = np.quantile(df["lag"].to_numpy(), np.arange(1, 100) / 100)

plt.plot(observed_quants, approx_quants)
plt.plot(observed_quants, observed_quants)
plt.show()
