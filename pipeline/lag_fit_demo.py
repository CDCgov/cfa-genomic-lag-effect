import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pipeline.bhsqi

with open("pipeline/config.json", "r") as file:
    config = json.load(file)

ns_path = config["empirical_lag"]["nextstrain_path"]

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

approx = pipeline.bhsqi.BHSQI.from_samples(
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

# QQ plot of approximation vs empirical distribution
approx_quants = np.vectorize(approx.quantile_function)(np.arange(1, 100) / 100)
observed_quants = np.quantile(df["lag"].to_numpy(), np.arange(1, 100) / 100)

plt.plot(observed_quants, approx_quants)
plt.plot(
    observed_quants, observed_quants, ls="--"
)  # a 1:1 line for comparison
plt.show()

# Samples look like approximation they should be draws from
approx_samples = approx.draw(100000)

plt.hist(approx_samples, bins=500, density=True)
plt.plot(t, approx.pdf(t))
plt.show()

approx_sample_quants = np.quantile(approx_samples, np.arange(1, 100) / 100)

plt.plot(approx_quants, approx_sample_quants)
plt.plot(approx_quants, approx_quants, ls="--")  # a 1:1 line for comparison
plt.show()
