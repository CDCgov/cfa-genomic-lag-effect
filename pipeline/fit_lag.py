import json
from typing import Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import brentq

from pipeline.utils import parser, read_config


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

        self.location = 0.0
        self.scale = 1.0

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

    def cdf(self, x, location=0.0, scale=1.0) -> NDArray:
        return np.where(
            LocationScale.reverse(x, location, scale) <= self.xmin,
            0.0,
            np.where(
                LocationScale.reverse(x, location, scale) >= self.xmax,
                1.0,
                self._cdf(LocationScale.reverse(x, location, scale)),
            ),
        )

    def draw(
        self,
        size: int,
        location=0.0,
        scale=1.0,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> NDArray:
        u = rng.uniform(low=0.0, high=self.pmax, size=size)
        return np.array(
            [
                self.quantile_function(uu, location=location, scale=scale)
                for uu in u
            ]
        )

    def pdf(self, x, location=0.0, scale=1.0) -> NDArray:
        if scale == 0.0:
            return np.where(x == location, np.inf, 0.0)
        else:
            return np.where(
                LocationScale.reverse(x, location, scale) <= self.xmin,
                0.0,
                np.where(
                    LocationScale.reverse(x, location, scale) >= self.xmax,
                    0.0,
                    self._pdf(LocationScale.reverse(x, location, scale)),
                ),
            )

    def quantile_function(self, p: float, location=0.0, scale=1.0) -> float:
        interval = np.argwhere(self.cdf_precompute <= p).max()
        if p == self.cdf_precompute[interval]:
            return float(self.cdf_points[interval])
        remainder = p - self.cdf_precompute[interval]
        unscaled = brentq(
            lambda x: self._pdf.integrate(self.cdf_points[interval], x)
            - remainder,
            self.cdf_points[interval],
            self.cdf_points[interval + 1],
        )
        return LocationScale.forward(unscaled, location, scale)  # type: ignore


class LocationScale:
    """
    A class for location-scale transformations.

    Forward: g(x) = location + scale * x
    Reverse: g'(x) = (x - location) / scale

    https://en.wikipedia.org/wiki/Location-scale_family
    """

    @staticmethod
    def forward(x, location, scale) -> NDArray:
        if scale == 0.0:
            return np.zeros(np.shape(x))
        else:
            return location + scale * x

    @staticmethod
    def reverse(x, location, scale) -> NDArray:
        if scale == 0.0:
            return np.where(x == location, 0.0, np.inf)
        else:
            return (x - location) / scale


def get_empirical_lag(config) -> NDArray:
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
        pl.scan_csv("pipeline/input/metadata.tsv", separator="\t")
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
    return df["lag"].to_numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    config = read_config(args.config)

    lags = get_empirical_lag(config)

    low = (
        config["empirical_lag"]["low"]
        if config["empirical_lag"]["low"] is not None
        else lags.min() - 1
    )
    high = (
        config["empirical_lag"]["high"]
        if config["empirical_lag"]["high"] is not None
        else lags.max() + 1
    )

    assert low <= lags.min()
    assert high >= lags.min()

    knots, coef = BHSQI.bshqi(
        samples=lags,
        n_steps=500,
        low=low,
        high=high,
    )

    params = {
        "knots": knots.tolist(),
        "coef": coef.tolist(),
    }

    with open(args.outfile, "w") as outfile:
        json.dump(params, outfile)
