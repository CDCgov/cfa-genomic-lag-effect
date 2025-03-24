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
        self,
        coalescent_times: NDArray,
        sampling_times: NDArray,
        rate_shift_times: NDArray,
    ):
        CoalescentIntervals.assert_valid_coalescent_times(
            coalescent_times, sampling_times
        )
        self.intervals = CoalescentIntervals.construct_coalescent_intervals(
            coalescent_times, sampling_times, rate_shift_times
        )

    @staticmethod
    def assert_valid_coalescent_times(
        coalescent_times: NDArray, sampling_times: NDArray
    ) -> None:
        """
        To produce a valid tree the kth-ranked (1 being the smallest)
        coalescent time must have k+1 sampling times more recent than it.
        """
        assert len(coalescent_times.shape) == 1
        assert len(sampling_times.shape) == 1
        assert coalescent_times.shape[0] == sampling_times.shape[0] - 1

        srt_ct = np.sort(coalescent_times)
        srt_st = np.sort(sampling_times)

        for ct, st in zip(srt_ct, srt_st[1:]):
            assert ct >= st

    @staticmethod
    def construct_coalescent_intervals(
        coalescent_times: NDArray,
        sampling_times: NDArray,
        rate_shift_times: NDArray,
    ):
        """
        Matrix with information required to compute the coalescent likelihood of the provided
        coalescent and sampling times given a rate function on the specified grid.

        `rate_shift_times` should be only the times of changes, one less than the number of rate intervals.

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
                        coalescent_times,
                        np.repeat(-1, coalescent_times.shape),
                        np.repeat(1, coalescent_times.shape),
                        np.repeat(0, coalescent_times.shape),
                    )
                ),
                np.column_stack(
                    (
                        sampling_times,
                        np.repeat(1, sampling_times.shape),
                        np.repeat(0, sampling_times.shape),
                        np.repeat(0, sampling_times.shape),
                    )
                ),
                np.column_stack(
                    (
                        rate_shift_times,
                        np.repeat(0, rate_shift_times.shape),
                        np.repeat(0, rate_shift_times.shape),
                        np.repeat(1, rate_shift_times.shape),
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
    intervals: CoalescentIntervals, force_of_infection, prevalence
):
    """
    Computes the likelihood of construct_epi_coalescent_grid(coalescent_times, sampling_times, rate_shift_times)
    given the piecewise constant force of infection and piecewise constant prevalence.
    """

    rate = (
        intervals.num_active_choose_2()
        * 2.0
        * force_of_infection[intervals.rate_indexer()]
        / prevalence[intervals.rate_indexer()]
    )
    lnl = rate * intervals.dt() - jnp.where(
        intervals.ends_in_coalescent_indicator(), jnp.log(rate), 0.0
    )
    return lnl.sum()


def episodic_epi_coalescent_factor(
    intervals: CoalescentIntervals, force_of_infection, prevalence
):
    """
    Computes the likelihood of construct_epi_coalescent_grid(coalescent_times, sampling_times, rate_shift_times)
    given the piecewise constant force of infection and piecewise constant prevalence.
    """
    factor(
        "coalescent_likelihood",
        episodic_epi_coalescent_loglik(
            intervals, force_of_infection, prevalence
        ),
    )


def construct_epi_coalescent_sim_grid(
    sampling_times: NDArray, rate_shift_times: NDArray
):
    """
    As construct_epi_coalescent_grid(), but without coalescent times and with a terminal
    fake rate-shift at infinity.
    """
    assert len(sampling_times.shape) == 1
    assert len(rate_shift_times.shape) == 1

    all_times = np.concatenate(
        (
            np.column_stack(
                (
                    sampling_times,
                    np.repeat(1, sampling_times.shape),
                )
            ),
            np.column_stack(
                (
                    np.concat((rate_shift_times, np.array([np.inf]))),
                    np.repeat(0, rate_shift_times.shape[0] + 1),
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


def inv_coalescent_rate(
    prevalence: float, n_active: int, force_of_infection: float
):
    """
    1/(coalescent rate)
    """
    return prevalence / (choose2(n_active) * 2.0 * force_of_infection)


@np.errstate(divide="ignore")
def sim_episodic_epi_coalescent(
    times_types: NDArray,
    force_of_infection: NDArray,
    prevalence: NDArray,
    rng=np.random.default_rng(),
) -> NDArray:
    """
    Simulates coalescent times given construct_epi_coalescent_sim_grid(sampling_times, rate_shift_times),
    the piecewise constant force of infection, and the piecewise constant prevalence.
    """
    n_real_rate_shifts = (
        (times_types[:, 1] == 0) & (times_types[:, 0] < np.inf)
    ).sum()
    assert (
        n_real_rate_shifts == force_of_infection.shape[0] - 1
    ), f"There appear to be {n_real_rate_shifts} rate shift times in `times_types`, expected {n_real_rate_shifts + 1} force_of_infection and prevalence entries."
    assert (
        force_of_infection.shape[0] == prevalence.shape[0]
    ), f"Provided force_of_infection is length {force_of_infection.shape[0]} while provided prevalence is length {prevalence.shape[0]}"

    n_samps = (times_types[:, 1] == 1).sum()
    n_coal = n_samps - 1
    coalescent_times = np.zeros(n_coal)
    n_active = 0
    rate_idx = 0

    print(times_types)
    time = 0.0
    coal_idx = 0
    for i in range(times_types.shape[0]):
        rate_inv = inv_coalescent_rate(
            prevalence[rate_idx], n_active, force_of_infection[rate_idx]
        )
        while coal_idx < n_coal:
            wt = rng.exponential(rate_inv)
            if time + wt < times_types[i, 0]:
                time += wt
                coalescent_times[coal_idx] = time
                coal_idx += 1
                n_active -= 1
                rate_inv = inv_coalescent_rate(
                    prevalence[rate_idx],
                    n_active,
                    force_of_infection[rate_idx],
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

    return coalescent_times
