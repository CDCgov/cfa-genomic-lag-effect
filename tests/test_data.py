import numpy as np
import pytest

import lag.data as data


def test_validate_times():
    samps = np.arange(9)
    coals = samps[1:]
    bad_samps = samps + 10.0

    # Should pass
    data.CoalescentData.assert_valid_coalescent_times(coals, samps)

    # Should fail
    with pytest.raises(Exception) as e_info:
        data.CoalescentData.assert_valid_coalescent_times(coals, bad_samps)
    assert isinstance(e_info.value, AssertionError)


def test_grid():
    """
    The expected durations, active lineage counts, and indices can be obtained
    by drawing out either tree which matches the provided times.
    """
    samps = np.array([1, 1, 2, 6])
    coals = np.array([3, 5, 8])
    rate_grid = np.array([4])

    intervals = data.CoalescentData(coals, samps, rate_grid)

    n_active = np.array([0, 1, 2, 3, 2, 2, 1, 2])
    expected_dt = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    expected_nc2 = data.choose2(n_active)
    expected_indicator = np.array([0, 0, 0, 1, 0, 1, 0, 1])
    expected_index = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    assert np.all(intervals.dt == expected_dt)
    assert np.all(intervals.num_active_choose_2 == expected_nc2)
    assert np.all(intervals.ends_in_coalescent_indicator == expected_indicator)
    assert np.all(intervals.rate_indexer == expected_index)
