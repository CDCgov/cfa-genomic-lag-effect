import numpy as np
import pytest

import lag.data as data


@pytest.fixture
def four_tip():
    r"""
    There is only one tree possible for these times:
    /----- s_3
    +
    |                          /------ s_2
    \--------------------------+
                               |                          /------ s_1
                               \--------------------------+
                                                          \-------------------- s_0
    time
    5.5    5                   3.5     3                  1.5     1             0
    """
    samps = np.array([0, 1, 3, 5])
    coals = np.array([1.5, 3.5, 5.5])
    rst = np.array([0.5, 2.25, 2.5, 2.75])
    intervals = data.CoalescentData(
        sampling_times=samps, coalescent_times=coals, rate_shift_times=rst
    )

    return intervals


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


def test_no_lag(four_tip):
    lagged = four_tip.as_of(
        as_of=0.0, lags=np.array([0.0] * 4), rng=np.random.default_rng(0)
    )
    assert (lagged.coalescent_times == four_tip.coalescent_times).all()
    assert (lagged.sampling_times == four_tip.sampling_times).all()


def test_too_much_lag(four_tip):
    with pytest.raises(Exception) as e_info:
        _ = four_tip.as_of(
            as_of=10.0, lags=np.array([0.0] * 4), rng=np.random.default_rng(0)
        )
    assert isinstance(e_info.value, ValueError)


def test_lag(four_tip):
    lag_0_as_of_0_5 = four_tip.as_of(
        as_of=0.5, lags=np.array([0.0] * 4), rng=np.random.default_rng(0)
    )
    assert (lag_0_as_of_0_5.coalescent_times == np.array([3.5, 5.5])).all()
    assert (lag_0_as_of_0_5.sampling_times == np.array([1, 3, 5])).all()

    lag_0_as_of_1_1 = four_tip.as_of(
        as_of=1.1, lags=np.array([0.0] * 4), rng=np.random.default_rng(0)
    )
    assert (lag_0_as_of_1_1.coalescent_times == np.array([5.5])).all()
    assert (lag_0_as_of_1_1.sampling_times == np.array([3, 5])).all()

    as_of_0_lag_1_1 = four_tip.as_of(
        as_of=0.0, lags=np.array([1.1] * 4), rng=np.random.default_rng(0)
    )
    assert (
        lag_0_as_of_1_1.coalescent_times == as_of_0_lag_1_1.coalescent_times
    ).all()
    assert (
        lag_0_as_of_1_1.sampling_times == as_of_0_lag_1_1.sampling_times
    ).all()

    varying_lag = four_tip.as_of(
        as_of=0.0,
        lags=np.array([0.0, 1.5, 0.25, 0.5]),
        rng=np.random.default_rng(0),
    )
    assert (varying_lag.coalescent_times == np.array([3.5, 5.5])).all()
    assert (varying_lag.sampling_times == np.array([0, 3, 5])).all()
