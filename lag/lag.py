import dendropy
import numpy as np
from numpy.typing import NDArray

from lag.data import CoalescentData


class LaggableTimes:
    def __init__(
        self,
        coal_times: NDArray,
        samp_times: NDArray,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.tree, self.times = LaggableTimes.topo_from_times(
            coal_times, samp_times, rng
        )
        self.sample_order = np.argsort(samp_times, stable=True)
        self._nsamps = self.sample_order.shape[0]

    @staticmethod
    def topo_from_times(
        coal_times: NDArray, samp_times: NDArray, rng: np.random.Generator
    ) -> tuple[dendropy.Tree, dict[str, float]]:
        """
        Generates a random topology compatible with the times.

        At each coalescent event, the pair of active lineages to coalesce is chosen at random.
        This implicitly assumes a one-population/panmictic ("standard") coalescent, rather than a structured coalescent.
        """
        times = {}
        intervals = CoalescentData(
            coal_times, samp_times, np.array([np.inf])
        )
        # print(f"My intervals are\n {intervals.intervals}")
        active = []
        time = 0.0
        sidx = 0
        cidx = 0
        for i in range(intervals.intervals.shape[0] - 1):
            # print(f"+++ Iterating, active = {active}")
            time += intervals.intervals[i, 0]
            if intervals.intervals[i, 2] == 1:
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
            else:
                active.append(dendropy.Node(label=f"s_{sidx}"))
                times[f"s_{sidx}"] = time
                sidx += 1

        tree = dendropy.Tree()
        tree.seed_node = parent

        return (
            tree,
            times,
        )

    def apply_lags(
        self, lags: NDArray, as_of: float
    ) -> tuple[NDArray, NDArray]:
        """
        Returns the coalescent and sampling events observed prior to `as_of` given `lags`.

        The `lags` are taken to be in the same order as the sampling times.
        """
        assert len(lags.shape) == 1
        assert (
            lags.shape[0] == self._nsamps
        ), f"Number of provided lags {len(lags)} does not match number of samples {self._nsamps}"
        lagged = dendropy.Tree(self.tree)
        remove_samps = [
            f"s_{i}"
            for i in range(self._nsamps)
            if self.times[f"s_{i}"] + lags[i] >= as_of
        ]
        lagged.filter_leaf_nodes(lambda x: x.label in remove_samps)

        return (
            np.array(
                [
                    self.times[node.label]
                    for node in lagged.postorder_node_iter()
                ]
            ),
            np.array(
                [self.times[node.label] for node in lagged.leaf_node_iter()]
            ),
        )
