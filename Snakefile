import numpy as np

configfile: "pipeline/config.json"

rule all:
    input:
        expand(
            "pipeline/out/coalescent/{scenarios}_{i0}_{scaling_factors}_{rep}.json",
            scenarios=config["simulations"]["rt_scenarios"],
            i0=config["simulations"]["i0"],
            scaling_factors=config["empirical_lag"]["scaling_factors"],
            rep=np.arange(config["simulations"]["n_rep"])
        )

rule fit_lags:
    output:
        "pipeline/out/lags/{scaling_factors}.json",

    shell:
        "python3 -m pipeline.bhsqi --config pipeline/config.json"

rule simulate_rt:
    output:
        "pipeline/out/rt/{scenarios}.txt",

    shell:
        "python3 -m pipeline.simulate_rt --config pipeline/config.json"

rule generate_infections:
    input:
        "pipeline/out/rt/{scenarios}.txt",

    output:
        "pipeline/out/infections/incidence_{scenarios}_{i0}.txt"
        # This should probably be two steps since declaring these both as outputs is not working
        # "pipeline/out/infections/prevalence_{scenarios}_{i0}.txt"

    shell:
        "python3 -m pipeline.generate_infections --config pipeline/config.json"

rule simulate_data:
    input:
        "pipeline/out/lags/{scaling_factors}.json",
        "pipeline/out/infections/incidence_{scenarios}_{i0}.txt"
        # "pipeline/out/infections/prevalence_{scenarios}_{i0}.txt"

    output:
        "pipeline/out/coalescent/{scenarios}_{i0}_{scaling_factors}_{rep}.json"

    shell:
        "python3 -m pipeline.simulate_data --config pipeline/config.json"
