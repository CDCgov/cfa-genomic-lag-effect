from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from lag.utils import choose2


class GenomicData(ABC):
    """
    Abstract class for genomic data.
    """

    @abstractmethod
    def validate(self):
        raise NotImplementedError()


class CoalescentData(GenomicData):
    """
    A class for intervals in a coalescent model.
    """

    def __init__(
        self,
        coalescent_times: Optional[NDArray] = None,
        sampling_times: Optional[NDArray] = None,
        rate_shift_times: Optional[NDArray] = None,
        intervals: Optional[NDArray] = None,
        likelihood_only: bool = False,
    ):
        if intervals is None:
            assert coalescent_times is not None
            assert sampling_times is not None
            assert rate_shift_times is not None
            self.intervals = CoalescentData.construct_coalescent_intervals(
                coalescent_times, sampling_times, rate_shift_times
            )

            assert np.isclose(
                np.sort(coalescent_times),
                np.sort(self.coalescent_times),
                atol=0.0,
            ).all()
            assert np.isclose(
                np.sort(sampling_times), np.sort(self.sampling_times), atol=0.0
            ).all()
        else:
            self.intervals = intervals
        self.validate(likelihood_only)

    def validate(self, likelihood_only: bool = False) -> None:
        if not likelihood_only:
            self.assert_valid_coalescent_times(
                self.coalescent_times, self.sampling_times
            )
        assert self.intervals.shape[1] == 4
        self.assert_col_specs()

    def assert_col_specs(self):
        # Nonnegative durations
        assert (self.dt >= 0).all()
        # Nonnegative integer choose(n_active, 2)
        assert (self.num_active_choose_2 >= 0).all()
        assert (
            (self.num_active_choose_2.astype(int) - self.num_active_choose_2)
            == 0.0
        ).all()
        # Change in active lineage counts is +1, +0, or -1
        assert set(self.da).issubset(set([-1, 0, 1]))
        # Nonnegative indices for rate function vector
        assert (self.rate_indexer >= 0).all()
        assert (
            (self.rate_indexer.astype(int) - self.rate_indexer) == 0.0
        ).all()

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
        assert (
            coalescent_times.shape[0] == sampling_times.shape[0] - 1
        ), f"There are {coalescent_times.shape[0]} coalescent times but {sampling_times.shape[0]} sampling times."

        srt_ct = np.sort(coalescent_times)
        srt_st = np.sort(sampling_times)

        for ct, st in zip(srt_ct, srt_st[1:]):
            assert (
                ct >= st
            ), f"Invalid coalescent/sampling time pair: {srt_ct}/{srt_st}"

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
        [:, 2] indicates change in number of active lineages at the end of the interval
        [:, 3] indicates the index for the rate function in that interval
        """

        events = [
            (coalescent_times, [-1, 1, 0]),
            (sampling_times, [1, 0, 0]),
            (rate_shift_times, [0, 0, 1]),
        ]

        event_times = np.concatenate(
            [
                np.column_stack(
                    [times, *[np.repeat(n, times.shape) for n in values]]
                )
                for times, values in events
            ]
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
                event_times[:, 1],
                rate_index,
            )
        )

        return intervals

    @classmethod
    def likelihood_relevant_intervals(cls, data: Self) -> Self:
        """
        Get only the intervals which do not occur with probability 1.

        This is any interval that has choose(n_active, 2) >= 1, as these are intervals where
        coalescence is possible, whether or not it occurs.
        """
        return cls(
            intervals=data.intervals[
                np.argwhere(data.num_active_choose_2 > 0).T[0], :
            ],
            likelihood_only=True,
        )

    ##################
    # Data accessors #
    ##################
    @property
    def coalescent_times(self) -> NDArray:
        """
        The times of the coalescent events
        """
        return np.cumsum(self.dt)[
            np.where(self.ends_in_coalescent_indicator == 1)
        ]

    @property
    def da(self) -> NDArray:
        """
        Change in number of active lineages (at interval end).
        """
        return self.intervals[:, 2]

    @property
    def dt(self) -> NDArray:
        """
        Change in time from interval start to end (i.e. duration).
        """
        return self.intervals[:, 0]

    @property
    def ends_in_coalescent_indicator(self) -> NDArray:
        """
        Indicator for whether each interval ends in a coalescent event.
        """
        return (self.da == -1).astype(int)

    @property
    def ends_in_sampling_indicator(self) -> NDArray:
        """
        Indicator for whether each interval ends in a sampling event.
        """
        return (self.da == 1).astype(int)

    @property
    def num_active_choose_2(self) -> NDArray:
        """
        choose(# active lineages, 2) for all intervals.
        """
        return self.intervals[:, 1]

    @property
    def rate_indexer(self) -> NDArray:
        """
        The element of the piecewise-constant rate function to use in each interval.
        """
        return self.intervals[:, 3].astype(int)

    @property
    def sampling_times(self) -> NDArray:
        """
        The times of the sampling events
        """
        return np.cumsum(self.dt)[
            np.where(self.ends_in_sampling_indicator == 1)
        ]
