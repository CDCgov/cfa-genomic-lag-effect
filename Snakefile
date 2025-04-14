import numpy as np

configfile: "pipeline/config.json"

rule all:
    input:
        expand(
            "pipeline/out/analysis/{scenario}_{i0}_{scaling_factor}_{rep}.json",
            scenario=config["simulations"]["rt_scenarios"],
            i0=config["simulations"]["i0"],
            scaling_factor=config["empirical_lag"]["scaling_factors"],
            rep=np.arange(config["simulations"]["n_rep"])
        )

rule diagnostics:
    input:
        expand(
            "pipeline/out/infections/{scenario}_{i0}.png",
            scenario=config["simulations"]["rt_scenarios"],
            i0=config["simulations"]["i0"],
        ),
        expand(
            "pipeline/out/rt/{scenario}.png",
            scenario=config["simulations"]["rt_scenarios"],
        ),
        expand(
            "pipeline/out/lag/{scaling_factor}.png",
            scaling_factor=config["empirical_lag"]["scaling_factors"],
        ),


rule simulate_rt:
    output:
        "pipeline/out/rt/{scenario}.txt",

    shell:
        "python3 -m pipeline.simulate_rt --config pipeline/config.json --scenario {wildcards.scenario} --outfile {output}"

rule plot_rt_diagnostic:
    input:
        "pipeline/out/rt/{scenario}.txt"

    output:
        "pipeline/out/rt/{scenario}.png",

    shell:
        "python3 -m pipeline.plot_rt --config pipeline/config.json --scenario {wildcards.scenario} --infile {input} --outfile {output}"

rule generate_incidence:
    input:
        "pipeline/out/rt/{scenario}.txt"

    output:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt"

    shell:
        "python3 -m pipeline.generate_incidence --config pipeline/config.json --scenario {wildcards.scenario} --i0 {wildcards.i0} --infile {input} --outfile {output}"

rule generate_prevalence:
    input:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt"

    output:
        "pipeline/out/infections/prevalence_{scenario}_{i0}.txt"

    shell:
        "python3 -m pipeline.generate_prevalence --config pipeline/config.json --infile {input} --outfile {output}"

rule plot_infection_diagnostics:
    input:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt",
        "pipeline/out/infections/prevalence_{scenario}_{i0}.txt",

    output:
        "pipeline/out/infections/{scenario}_{i0}.png"

    shell:
        "python3 -m pipeline.plot_infections --config pipeline/config.json --infile {input} --outfile {output}"

rule fit_lag:
    output:
        "pipeline/out/lag/fit.json",

    shell:
        "python3 -m pipeline.fit_lag --config pipeline/config.json --outfile {output}"

rule plot_lag_diagnostic:
    input:
        "pipeline/out/lag/fit.json",

    output:
        "pipeline/out/lag/{scaling_factor}.png",

    shell:
        "python3 -m pipeline.plot_lag --config pipeline/config.json --scaling_factor {wildcards.scaling_factor} --infile {input} --outfile {output}"

rule simulate_data:
    input:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt",
        "pipeline/out/infections/prevalence_{scenario}_{i0}.txt",
        "pipeline/out/lag/fit.json",

    output:
        "pipeline/out/coalescent/{scenario}_{i0}_{scaling_factor}_{rep}.json"

    shell:
        "python3 -m pipeline.simulate_data --config pipeline/config.json --scenario {wildcards.scenario} --i0 {wildcards.i0} --scaling_factor {wildcards.scaling_factor} --rep {wildcards.rep} --infile {input} --outfile {output}"

rule analyze:
    input:
        "pipeline/out/coalescent/{scenario}_{i0}_{scaling_factor}_{rep}.json",
        "pipeline/out/rt/{scenario}.txt",

    output:
        "pipeline/out/analysis/{scenario}_{i0}_{scaling_factor}_{rep}.json"

    shell:
        "python3 -m pipeline.analyze --config pipeline/config.json --infile {input} --outfile {output}"
