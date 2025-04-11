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


class LagSampler:
    """
    Wrapper for BHSQI that allows 0 lag.
    """

    def __init__(self, list_params):
        if list_params == {}:
            self.bhsqi = None
        else:
            self.bhsqi = BHSQI(
                np.array(list_params["knots"]), np.array(list_params["coef"])
            )

    def draw(self, size: int, rng: np.random.Generator) -> NDArray:
        if self.bhsqi is None:
            return np.array([0.0] * size)
        else:
            return self.bhsqi.draw(size=size, rng=rng)


def ar1(
    mu: NDArray,
    sd: float,
    ac: float,
    rng: np.random.Generator,
):
    """
    Draw from an AR1 process with mean vector mu,
    standard deviation sd, and autocorrelation ac.
    """
    n = mu.shape[0]
    z = rng.normal(loc=0.0, scale=1.0, size=n) * sd
    x = np.zeros(n)
    x[0] = z[0]
    for i in range(1, n):
        x[i] = x[i - 1] * ac + z[i]
    return np.exp(np.log(mu) + x)


def construct_seed(
    root_seed: int, scenario: str, i0: str, scaling_factor: str, rep: str
):
    scenario_to_int = {
        "decreasing": "0",
        "constant": "1",
        "increasing": "2",
    }
    i0_to_int = {
        "1000": "0",
        "2000": "1",
        "4000": "2",
    }
    scale_to_int = {
        "0.0": "0",
        "0.25": "1",
        "0.5": "2",
        "0.75": "3",
        "1.0": "4",
    }
    return int(
        str(root_seed)
        + scenario_to_int[scenario]
        + i0_to_int[i0]
        + scale_to_int[scaling_factor]
        + rep
    )


def generate_rt_scenario(
    r_init: float,
    r_final: float,
    init_weeks: int,
    change_weeks: int,
    sd: float,
    ac: float,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Generates an Rt time series from an AR1 process where for `init_weeks` Rt
    has median `r_init`, then the median changes towards `r_final` over the
    course of `change_weeks`.
    """
    mean = np.concat(
        (
            np.array([r_init] * init_weeks),
            np.linspace(r_init, r_final, change_weeks),
        )
    )

    return ar1(mean, sd, ac, rng)


def simulate_sampling_times(
    weekday_effect, n_sampled_weeks, n_samples, rng: np.random.Generator
) -> NDArray:
    """
    Samples come in uniformly during a day, at different rates per day of week, over a range of weeks before present day.
    """
    n_days = n_sampled_weeks * 7

    probs = np.tile(weekday_effect, n_sampled_weeks)
    probs = probs / probs.sum()

    backwards_times = n_days - (
        rng.choice(n_days, size=n_samples, replace=True)
        + rng.uniform(0.0, 1.0, size=n_samples)
    )

    return backwards_times
