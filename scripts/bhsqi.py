from typing import Optional

import numpy as np
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

    def __init__(
        self,
        samples: NDArray,
        n_steps: int,
        low: Optional[float] = None,
        high: Optional[float] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.rng = rng
        self.knots, self.coef = BHSQI.bshqi(samples, n_steps, low, high)
        self._pdf = BSpline(self.knots, self.coef, 2, extrapolate=False)
        self._cdf = self._pdf.antiderivative(1)
        self.cdf_points = self.knots[1:-1]
        self.xmin = self.cdf_points[0]
        self.xmax = self.cdf_points[-1]
        self.cdf_precompute = np.concat(
            (
                np.array([0.0]),
                self._cdf(self.cdf_points[1:-1]),
                np.array([1.0]),
            )
        )

    @staticmethod
    def bshqi(
        samples: NDArray,
        n_steps: int,
        low: Optional[float] = None,
        high: Optional[float] = None,
    ):
        """
        Fit the interpolation, see: https://www.sciencedirect.com/science/article/pii/S0377042724003807
        """
        n = samples.shape[0]
        a = low if low is not None else samples.min()
        b = high if high is not None else samples.max()

        h = (b - a) / n_steps

        # Defined between equations 2 and 3
        pi = np.linspace(a, b, n_steps + 1)

        assert np.isclose(pi[1] - pi[0], h, atol=0.0), print(
            f"Grid size is off by {h - (pi[1] - pi[0])}"
        )

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

    def draw(self, size: int) -> NDArray:
        u = self.rng.uniform(low=0.0, high=1.0, size=size)
        return np.array([self.quantile_function(uu) for uu in u])

    def pdf(self, x) -> NDArray:
        return np.where(
            x <= self.xmin, 0.0, np.where(x >= self.xmax, 0.0, self._pdf(x))
        )

    def quantile_function(self, p: float) -> float:
        interval = np.argwhere(self.cdf_precompute <= p).max()
        if p == self.cdf_precompute[interval]:
            return float(self.cdf_points[interval])
        return brentq(
            lambda x: self.cdf(x) - p,
            self.cdf_points[interval],
            self.cdf_points[interval + 1],
        )  # type: ignore
