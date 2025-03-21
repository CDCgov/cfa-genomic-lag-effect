import numpy as np
import pytest

import lag.coalescent as coalescent


def test_validate_times():
    samps = np.arange(9)
    coals = samps[1:]
    bad_samps = samps + 10.0

    # Should pass
    coalescent.CoalescentIntervals.assert_valid_coal_times(coals, samps)

    # Should fail
    with pytest.raises(Exception) as e_info:
        coalescent.CoalescentIntervals.assert_valid_coal_times(
            coals, bad_samps
        )
    assert isinstance(e_info.value, AssertionError)


def test_sim_runs():
    samps = np.arange(5, 10) + 0.5
    rate_grid = np.arange(19)
    grid = coalescent.construct_epi_coalescent_sim_grid(samps, rate_grid)
    rng = np.random.default_rng(0)
    coals = coalescent.sim_episodic_epi_coalescent(
        grid, np.arange(1, 21), np.arange(1, 21), rng
    )
    coalescent.CoalescentIntervals.assert_valid_coal_times(coals, samps)


def test_grid():
    """
    The expected durations, active lineage counts, and indices can be obtained
    by drawing out either tree which matches the provided times.
    """
    samps = np.array([1, 1, 2, 6])
    coals = np.array([3, 5, 8])
    rate_grid = np.array([4])

    intervals = coalescent.CoalescentIntervals(coals, samps, rate_grid)

    n_active = np.array(
        [
            0,
            1,
            2,
            3,
            2,
            2,
            1,
            2,
        ]
    )
    expected_dt = np.array(
        [
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            2.0,
        ]
    )
    expected_nc2 = coalescent.choose2(n_active)
    expected_indicator = np.array(
        [
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )
    expected_index = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    assert np.all(intervals.dt() == expected_dt)
    assert np.all(intervals.num_active_choose_2() == expected_nc2)
    assert np.all(
        intervals.ends_in_coalescent_indicator() == expected_indicator
    )
    assert np.all(intervals.rate_indexer() == expected_index)


def test_lnl_runs():
    samps = np.arange(5, 10) + 0.5
    coals = samps[1:] + 0.25
    rate_grid = np.arange(19)
    intervals = coalescent.CoalescentIntervals(coals, samps, rate_grid)
    _ = coalescent.episodic_epi_coalescent_loglik(
        intervals, np.arange(1, 21), np.arange(1, 21)
    )


def test_offset_same_loglik():
    samps = np.arange(5, 10) + 0.5
    coals = samps[1:] + 0.25

    rate_grid = np.arange(19)
    foi = np.arange(1, 21)
    prevalence = np.arange(1, 21)

    intervals = coalescent.CoalescentIntervals(coals, samps, rate_grid)

    lnl_no_offset = float(
        coalescent.episodic_epi_coalescent_loglik(intervals, foi, prevalence)
    )

    intervals.remove_deterministic_intervals()
    lnl_offset = float(
        coalescent.episodic_epi_coalescent_loglik(intervals, foi, prevalence)
    )

    assert lnl_no_offset == lnl_offset
