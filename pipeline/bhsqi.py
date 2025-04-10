import argparse
import json
from typing import Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import brentq


class BHSQI:
    """
    A class for spline interpolation of probability densities.

    The most useful features of the resulting object are:
    .pdf(), which evaluates the probability density function
    .cdf(), which evaluates the cumulative density function
    .draw(), which samples from the approximated distribution
    """

    def __init__(self, knots: NDArray, coef: NDArray):
        self.knots = knots
        self.xmin = self.knots[2]
        self.xmax = self.knots[-3]

        self.coef = coef

        self._pdf = BSpline(self.knots, self.coef, 2, extrapolate=False)
        assert (self._pdf(knots) != np.nan).all()

        self._cdf = np.vectorize(lambda x: self._pdf.integrate(self.xmin, x))
        self.cdf_points = self.knots[1:-1]
        self.pmax = self._cdf(self.xmax)
        self.cdf_precompute = self.cdf(self.cdf_points)

    @classmethod
    def from_samples(
        cls,
        samples: NDArray,
        n_steps: int,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        knots, coef = BHSQI.bshqi(samples, n_steps, low, high)
        return cls(knots, coef)

    @staticmethod
    def bshqi(
        samples: NDArray,
        n_steps: int,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        """
        Fits the interpolation

        For more, see https://www.sciencedirect.com/science/article/pii/S0377042724003807
        """
        n = samples.shape[0]
        a = low if low is not None else samples.min()
        b = high if high is not None else samples.max()

        h = (b - a) / n_steps

        # Grid, per Tamborrino et al.
        # Defined between equations 2 and 3
        pi = np.linspace(a, b, n_steps + 1)

        assert np.isclose(pi[1] - pi[0], h, atol=0.0), print(
            f"Grid size is off by {h - (pi[1] - pi[0])}"
        )

        # Pad out grid one step each direction
        pi = np.concat((np.array([pi[0] - h]), pi, np.array([pi[-1] + h])))

        # Defined in Lemma 2.2, give or take offsetting differences between paper and SciPy
        c_grid = pi[:-1] + h / 2

        @np.vectorize
        def naive_kernel(x):
            return ((samples >= x - h / 2.0) & (samples <= x + h / 2)).sum()

        # Lemma 2.2
        coef = 1.0 / (n * h) * naive_kernel(c_grid)

        knots = np.concat(
            (
                [pi[0]],
                pi,
                [pi[-1]],
            )
        )

        return (
            knots,
            coef,
        )

    def cdf(self, x) -> NDArray:
        return np.where(
            x <= self.xmin, 0.0, np.where(x >= self.xmax, 1.0, self._cdf(x))
        )

    def draw(
        self, size: int, rng: np.random.Generator = np.random.default_rng()
    ) -> NDArray:
        u = rng.uniform(low=0.0, high=self.pmax, size=size)
        return np.array([self.quantile_function(uu) for uu in u])

    def pdf(self, x) -> NDArray:
        return np.where(
            x <= self.xmin, 0.0, np.where(x >= self.xmax, 0.0, self._pdf(x))
        )

    def quantile_function(self, p: float) -> float:
        interval = np.argwhere(self.cdf_precompute <= p).max()
        if p == self.cdf_precompute[interval]:
            return float(self.cdf_points[interval])
        remainder = p - self.cdf_precompute[interval]
        return brentq(
            lambda x: self._pdf.integrate(self.cdf_points[interval], x)
            - remainder,
            self.cdf_points[interval],
            self.cdf_points[interval + 1],
        )  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit BHSQI approximations to scaled sample lags and save parameters to JSON."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file",
    )

    with open(parser.parse_args().config, "r") as file:
        config = json.load(file)

    ns_path = config["empirical_lag"]["nextstrain_path"]

    date_l = pl.date(
        config["empirical_lag"]["date_lower"]["y"],
        config["empirical_lag"]["date_lower"]["m"],
        config["empirical_lag"]["date_lower"]["d"],
    )
    date_u = pl.date(
        config["empirical_lag"]["date_upper"]["y"],
        config["empirical_lag"]["date_upper"]["m"],
        config["empirical_lag"]["date_upper"]["d"],
    )

    df = (
        pl.scan_csv(ns_path, separator="\t")
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
        config["empirical_lag"]["low"]
        if config["empirical_lag"]["low"] is not None
        else df["lag"].min() - 1
    )  # type: ignore
    high = (
        config["empirical_lag"]["high"]
        if config["empirical_lag"]["high"] is not None
        else df["lag"].max() + 1
    )  # type: ignore
    lags = df["lag"].to_numpy()

    for scale in config["empirical_lag"]["scaling_factors"]:
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
