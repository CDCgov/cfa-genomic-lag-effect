import numpy as np

configfile: "pipeline/config.json"

rule all:
    input:
        expand(
            "pipeline/out/coalescent/{scenario}_{i0}_{scaling_factor}_{rep}.json",
            scenario=config["simulations"]["rt_scenarios"],
            i0=config["simulations"]["i0"],
            scaling_factor=config["empirical_lag"]["scaling_factors"],
            rep=np.arange(config["simulations"]["n_rep"])
        )

rule simulate_rt:
    output:
        "pipeline/out/rt/{scenario}.txt",

    params:
        n_init_weeks=config["simulations"]["n_init_weeks"],
        n_change_weeks=config["simulations"]["n_change_weeks"],
        r_low=config["simulations"]["r_low"],
        r_med=config["simulations"]["r_med"],
        r_high=config["simulations"]["r_high"],
        sd=config["simulations"]["r_sd"],
        ac=config["simulations"]["r_ac"],
        seed=config["seed"],

    script:
        "pipeline/simulate_rt.py"

rule generate_incidence:
    output:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt"

    input:
        "pipeline/out/rt/{scenario}.txt"

    params:
        init_growth_rate=config["renewal"]["init_growth_rate"],
        init_growth_steps=config["renewal"]["init_growth_steps"],
        infectious_profile=config["renewal"]["infectious_profile"],

    script:
        "pipeline/generate_incidence.py"

rule generate_prevalence:
    output:
            "pipeline/out/infections/prevalence_{scenario}_{i0}.txt",

    input:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt"

    params:
        infectious_profile=config["renewal"]["infectious_profile"]

    script:
        "pipeline/generate_prevalence.py"

rule fit_lags:
    output:
        "pipeline/out/lags/{scaling_factor}.json",

    params:
        nextstrain_path = config["empirical_lag"]["nextstrain_path"],
        date_lower = config["empirical_lag"]["date_lower"],
        date_upper = config["empirical_lag"]["date_upper"],
        lag_low = config["empirical_lag"]["low"],
        lag_high = config["empirical_lag"]["high"],

    script:
        "pipeline/fit_lags.py"

rule simulate_data:
    input:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt",
        "pipeline/out/infections/prevalence_{scenario}_{i0}.txt",
        "pipeline/out/lags/{scaling_factor}.json",

    output:
        "pipeline/out/coalescent/{scenario}_{i0}_{scaling_factor}_{rep}.json"

    params:
        seed=config["seed"],
        n_weeks=config["simulations"]["n_init_weeks"] + config["simulations"]["n_change_weeks"],
        weekday_effect=config["simulations"]["sampling"]["weekday_effect"],
        n_sampled_weeks=config["simulations"]["sampling"]["n_sampled_weeks"],
        n_samples=config["simulations"]["sampling"]["n_samples"],

    script:
        "pipeline/simulate_data.py"
