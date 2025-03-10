import math

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from numpyro import factor

choose2 = np.vectorize(lambda x: math.comb(x, 2))
"""
numpy-vectorized choose(n, 2)
"""


class CoalescentIntervals:
    """
    A class for intervals in a coalescent model.
    """

    def __init__(
        self, coal_times: NDArray, samp_times: NDArray, rate_times: NDArray
    ):
        CoalescentIntervals.assert_valid_coal_times(coal_times, samp_times)
        self.intervals = CoalescentIntervals.construct_coalescent_intervals(
            coal_times, samp_times, rate_times
        )

    @staticmethod
    def assert_valid_coal_times(
        coal_times: NDArray, samp_times: NDArray
    ) -> None:
        """
        To produce a valid tree the kth-ranked (1 being the smallest)
        coalescent time must have k+1 sampling times more recent than it.
        """
        assert len(coal_times.shape) == 1
        assert len(samp_times.shape) == 1
        assert coal_times.shape[0] == samp_times.shape[0] - 1

        srt_ct = np.sort(coal_times)
        srt_st = np.sort(samp_times)

        for ct, st in zip(srt_ct, srt_st[1:]):
            assert ct >= st

    @staticmethod
    def construct_coalescent_intervals(
        coal_times: NDArray, samp_times: NDArray, rate_times: NDArray
    ):
        """
        Matrix with information required to compute the coalescent likelihood of the provided
        coalescent and sampling times given a rate function on the specified grid.

        `rate_times` should be only the times of changes, one less than the number of rate intervals.

        Returns matrix of intervals between events such that
        [:, 0] is the duration of the interval
        [:, 1] is choose(# active lineages, 2) during the interval
        [:, 2] indicates if the interval ends in a coalescent event
        [:, 3] indicates the index for the rate function in that interval
        """

        # A matrix with columns:
        #     0: time of event
        #     1: change in # active lineages at event
        #     2: indicator for coalescent event specifically
        #     3: change in rate interval index at event
        event_times = np.concatenate(
            (
                np.column_stack(
                    (
                        coal_times,
                        np.repeat(-1, coal_times.shape),
                        np.repeat(1, coal_times.shape),
                        np.repeat(0, coal_times.shape),
                    )
                ),
                np.column_stack(
                    (
                        samp_times,
                        np.repeat(1, samp_times.shape),
                        np.repeat(0, samp_times.shape),
                        np.repeat(0, samp_times.shape),
                    )
                ),
                np.column_stack(
                    (
                        rate_times,
                        np.repeat(0, rate_times.shape),
                        np.repeat(0, rate_times.shape),
                        np.repeat(1, rate_times.shape),
                    )
                ),
            ),
        )
        # When ties are present, puts coalescent times before rate shifts, sampling times after
        key = np.lexsort(
            (
                event_times[:, 1],
                event_times[:, 0],
            )
        )
        event_times = event_times[key, :]

        dt = np.diff(np.concat((np.zeros(1), event_times[:, 0])))
        # Active lineages during interval, not counting any changes at event at end of interval
        n_active = np.cumsum(
            np.concat(
                (
                    np.zeros(1),
                    event_times[:, 1],
                )
            )
        ).astype(int)[:-1]
        nc2 = choose2(n_active)
        # Rate during interval, not counting any changes at event at end of interval
        rate_index = np.cumsum(np.concat((np.zeros(1), event_times[:, 3])))[
            :-1
        ]

        intervals = np.column_stack(
            (
                dt,
                nc2,
                event_times[:, 2],
                rate_index,
            )
        )

        return intervals

    def remove_deterministic_intervals(self) -> None:
        """
        Removes intervals that don't impact the likelihood from output of `construct_coalescent_intervals`.

        Intervals where choose(# active lineages, 2) have rate 0 and thus with probability 1 nothing happens.
        """
        self.intervals = self.intervals[
            np.argwhere(self.intervals[:, 1] > 0).T[0], :
        ]

    def dt(self) -> NDArray:
        """
        Get duration of all intervals.
        """
        return self.intervals[:, 0]

    def ends_in_coalescent_indicator(self) -> NDArray:
        """
        Get indicator for whether each interval ends in a coalescent event.
        """
        return self.intervals[:, 2]

    def num_active_choose_2(self) -> NDArray:
        """
        choose(# active lineages, 2) for all intervals.
        """
        return self.intervals[:, 1]

    def rate_indexer(self) -> NDArray:
        """
        The element of the piecewise-constant rate function to use in each interval.
        """
        return self.intervals[:, 3].astype(int)


def episodic_epi_coalescent_loglik(
    intervals: CoalescentIntervals, foi, prevalence
):
    """
    Computes the likelihood of construct_epi_coalescent_grid(coal_times, samp_times, rate_times)
    given the piecewise constant force of infection and piecewise constant prevalence.
    """

    rate = (
        intervals.num_active_choose_2()
        * 2.0
        * foi[intervals.rate_indexer()]
        / prevalence[intervals.rate_indexer()]
    )
    lnl = rate * intervals.dt() - jnp.where(
        intervals.ends_in_coalescent_indicator(), jnp.log(rate), 0.0
    )
    return lnl.sum()


def episodic_epi_coalescent_factor(
    intervals: CoalescentIntervals, foi, prevalence
):
    """
    Computes the likelihood of construct_epi_coalescent_grid(coal_times, samp_times, rate_times)
    given the piecewise constant force of infection and piecewise constant prevalence.
    """
    factor(
        "coalescent_likelihood",
        episodic_epi_coalescent_loglik(intervals, foi, prevalence),
    )


def construct_epi_coalescent_sim_grid(
    samp_times: NDArray, rate_times: NDArray
):
    """
    As construct_epi_coalescent_grid(), but without coalescent times and with a terminal
    fake rate-shift at infinity.
    """
    assert len(samp_times.shape) == 1
    assert len(rate_times.shape) == 1

    all_times = np.concatenate(
        (
            np.column_stack(
                (
                    samp_times,
                    np.repeat(1, samp_times.shape),
                )
            ),
            np.column_stack(
                (
                    np.concat((rate_times, np.array([np.inf]))),
                    np.repeat(0, rate_times.shape[0] + 1),
                )
            ),
        ),
    )
    key = np.lexsort(
        (
            all_times[:, 1],
            all_times[:, 0],
        )
    )
    return all_times[key, :]


@np.errstate(divide="ignore")
def sim_episodic_epi_coalescent(
    times_types: NDArray,
    foi: NDArray,
    prevalence: NDArray,
    rng=np.random.default_rng(),
) -> NDArray:
    """
    Simulates coalescent times given construct_epi_coalescent_sim_grid(samp_times, rate_times),
    the piecewise constant force of infection, and the piecewise constant prevalence.
    """
    n_real_rate_shifts = (
        (times_types[:, 1] == 0) & (times_types[:, 0] < np.inf)
    ).sum()
    assert (
        n_real_rate_shifts == foi.shape[0] - 1
    ), f"There appear to be {n_real_rate_shifts} rate shift times in `times_types`, expected {n_real_rate_shifts + 1} foi and prevalence entries."
    assert (
        foi.shape[0] == prevalence.shape[0]
    ), f"Provided foi is length {foi.shape[0]} while provided prevalence is length {prevalence.shape[0]}"

    n_samps = (times_types[:, 1] == 1).sum()
    n_coal = n_samps - 1
    coal_times = np.zeros(n_coal)
    n_active = 0
    rate_idx = 0

    print(times_types)
    time = 0.0
    coal_idx = 0
    for i in range(times_types.shape[0]):
        rate_inv = prevalence[rate_idx] / (
            choose2(n_active) * 2.0 * foi[rate_idx]
        )
        while coal_idx < n_coal:
            wt = rng.exponential(rate_inv)
            if time + wt < times_types[i, 0]:
                time += wt
                coal_times[coal_idx] = time
                coal_idx += 1
                n_active -= 1
                rate_inv = prevalence[rate_idx] / (
                    choose2(n_active) * 2.0 * foi[rate_idx]
                )
            else:
                time = times_types[i, 0]
                break

        if times_types[i, 1] == 0:
            if times_types[i, 0] < np.inf:
                rate_idx += 1
        elif times_types[i, 1] == 1:
            n_active += 1
        else:
            raise ValueError()

    return coal_times
