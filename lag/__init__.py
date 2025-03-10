import numpy as np

import lag.coalescent as coalescent
import lag.renewal as renewal


def grid_helper(coal_times, samp_times, rev_inf_prof, gen_int):
    """
    Figures out where the tree lives in renewal grid time and the farthest back the grid must go.
    """
    tmrs = samp_times.min()
    tmrca = coal_times.max()

    # The discrete-time intervals into which these events fall
    tree_t_min = np.floor(tmrs).astype(int)
    tree_t_max = np.floor(tmrca).astype(int)
    # You need at least gen_int days of incidence before t_MRCA to get prevalence at that time
    # You need a burn-in of at least rev_inf_prof to have the entire renewal history contributing to daily incidence at some time
    tmp = 1 + gen_int + len(rev_inf_prof)
    burnin = 50 if 50 > tmp else tmp

    return tree_t_min, tree_t_max, burnin


def preprocess(coal_times, samp_times, rev_inf_prof, gen_int):
    """
    Convenience function to obtain formatted data for use in `renewal_coalescent_model`
    """
    tree_t_min, tree_t_max, init_growth_steps = grid_helper(
        coal_times, samp_times, rev_inf_prof, gen_int
    )
    n_weeks = np.ceil((tree_t_max + init_growth_steps) / 7).astype(int)
    renewal_t_max = tree_t_max + init_growth_steps

    rate_grid = np.arange(tree_t_min + 1, tree_t_max + 1)
    intervals = coalescent.CoalescentIntervals(
        coal_times, samp_times, rate_grid
    )
    intervals.remove_deterministic_intervals()

    return intervals, init_growth_steps, n_weeks, renewal_t_max


def renewal_coalescent_model(
    intervals: coalescent.CoalescentIntervals,
    init_growth_steps,
    n_weeks,
    renewal_t_max,
    rev_inf_prof,
    gen_int,
):
    """
    A model inferring Rt from coalescent times using a discrete-time renewal model as the link.
    """

    # Renewal model
    daily_rt = renewal.daily_rt(n_weeks, renewal_t_max)
    i0 = renewal.i0()
    init_growth_rate = renewal.exp_growth_rate()
    daily_incidence = renewal.daily_incidence(
        daily_rt, rev_inf_prof, i0, init_growth_rate, init_growth_steps
    )
    daily_prevalence = renewal.daily_prevalence(daily_incidence, gen_int)

    # Coalescent likelihood
    coalescent.episodic_epi_coalescent_factor(
        intervals, daily_incidence, daily_prevalence
    )
