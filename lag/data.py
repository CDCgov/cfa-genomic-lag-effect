from abc import ABC, abstractmethod
from typing import Optional

import dendropy
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from lag.utils import choose2, is_nondecreasing


class GenomicData(ABC):
    """
    Abstract class for genomic data.
    """

    @abstractmethod
    def validate(self):
        raise NotImplementedError()


class LaggableGenomicData(ABC):
    """
    GenomicData that can tell you what was known as of a particular time.
    """

    @abstractmethod
    def as_of(self, as_of: float, lags: NDArray, **kwargs) -> Self:
        """
        What would the data has looked as of time `as_of`?

        Parameters
        ----------
        as_of : float
            The `as_of` time, measured in time units before the present.
            The present is taken to be time 0.

        lags: NDArray
            The time-ordered lags from event date to report date.
            Ordering means that lags[i] is applied to the ith-smallest sampling time.

        Returns
        -------
        Self
            The GenomicData as it would have looked at the time.

        """
        raise NotImplementedError()


class CoalescentData(LaggableGenomicData):
    """
    A class for intervals in a coalescent model.
    """

    def __init__(
        self,
        coalescent_times: Optional[NDArray] = None,
        sampling_times: Optional[NDArray] = None,
        rate_shift_times: Optional[NDArray] = None,
        rate_indices: Optional[NDArray] = None,
        intervals: Optional[NDArray] = None,
        likelihood_only: bool = False,
    ):
        """
        CoalescentData constructor.

        Parameters
        -------
        coalescent_times: Optional[NDArray]
            The times of coalescent events.
            If provided, must also provide `sampling_times` and `rate_shift_times`.
        sampling_times: Optional[NDArray]
            The times of sampling events.
            If provided, must also provide `coalescent_times` and `rate_shift_times`.
        rate_shift_times: Optional[NDArray]
            The times at which the piecewise constant coalescent rate function changes.
            If provided, must also provide `coalescent_times` and `sampling_times`.
        rate_indices: Optional[NDArray]
            If providing rate_shift_times,
        intervals: Optional[NDArray]
            Allows construction of a new
        likelihood_only: bool
        """
        self.tree = None
        if intervals is None:
            assert coalescent_times is not None
            assert sampling_times is not None
            assert rate_shift_times is not None
            self.intervals = CoalescentData.construct_coalescent_intervals(
                coalescent_times,
                sampling_times,
                rate_shift_times,
                rate_indices,
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

    def as_of(self, as_of: float, lags: NDArray, **kwargs) -> Self:
        assert as_of >= 0.0
        assert (
            lags.shape == self.sampling_times.shape
        ), "Provided lags don't match sampling times"
        rng = kwargs.get("rng", np.random.default_rng())
        tree, time_map = self.random_topology(rng)
        remove_samps = [
            f"s_{i}"
            for i in range(lags.shape[0])
            if time_map[f"s_{i}"] + lags[i] >= as_of
        ]
        tree.filter_leaf_nodes(
            lambda x: x.label in remove_samps, suppress_unifurcations=True
        )

        samp_times = [time_map[node.label] for node in tree.leaf_node_iter()]
        coal_times = [
            time_map[node.label] for node in tree.preorder_internal_node_iter()
        ]
        all_rate_indices = self.rate_indexer
        return type(self)(
            coalescent_times=np.array(coal_times),
            sampling_times=np.array(samp_times),
            rate_shift_times=self.rate_shift_times,
            rate_indices=np.arange(
                all_rate_indices.min(), all_rate_indices.max() + 1
            ),
        )

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
        # Nonnegative nondecreasing indices for rate function vector
        assert (self.rate_indexer >= 0).all()
        assert (
            (self.rate_indexer.astype(int) - self.rate_indexer) == 0.0
        ).all()
        assert is_nondecreasing(self.rate_indexer)

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
        rate_indices: Optional[NDArray],
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
        if rate_indices is not None:
            rate_index = rate_indices[rate_index]

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

    def random_topology(
        self, rng: np.random.Generator
    ) -> tuple[dendropy.Tree, dict[str, float]]:
        """
        Generates a random topology compatible with coalescent and sampling times.

        At each coalescent event, the pair of active lineages to coalesce is chosen at random.
        This implicitly assumes a one-population/panmictic ("standard") coalescent, rather than a structured coalescent.
        """
        times = {}

        active = []
        time = 0.0
        sidx = 0
        cidx = 0

        dt = self.dt
        is_coalescent = self.ends_in_coalescent_indicator
        is_sampling = self.ends_in_sampling_indicator
        for i in range(dt.shape[0] - 1):
            # print(f"+++ Iterating, active = {active}")
            time += dt[i]
            if is_coalescent[i]:
                parent = dendropy.Node(label=f"c_{cidx}")
                times[f"c_{cidx}"] = time
                chosen = rng.choice(len(active), 2, replace=False).astype(int)
                # print(f"++++++ chose to merge {chosen}")
                for i in sorted(chosen, reverse=True):
                    # print(f"+++++++++ working with {i} ({active[i]})")
                    child = active.pop(i)
                    # print(f"+++++++++ active = {active}")
                    parent.add_child(child)
                    child.parent_node = parent
                    # For visualization purposes
                    child.edge_length = time - times[child.label]
                active.append(parent)
                # print(f"parent {parent} has children {parent.child_nodes}")
                cidx += 1
            elif is_sampling[i]:
                active.append(dendropy.Node(label=f"s_{sidx}"))
                times[f"s_{sidx}"] = time
                sidx += 1

        tree = dendropy.Tree()
        tree.seed_node = parent

        return (
            tree,
            times,
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
    def rate_shift_times(self) -> NDArray:
        """
        The times at which the piecewise constant rate function changes.
        """
        return np.cumsum(self.dt)[np.where(self.da == 0)]

    @property
    def sampling_times(self) -> NDArray:
        """
        The times of the sampling events
        """
        return np.cumsum(self.dt)[
            np.where(self.ends_in_sampling_indicator == 1)
        ]
