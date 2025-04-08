import json
import os

import numpy as np

configfile: "pipeline/config.json"

scenario_list = glob_wildcards("pipeline/input/rt/{scenario}.txt").scenario

rule all:
    input:
        expand(
            "pipeline/output/analysis/rt_{scenario}_{i0}_{scaling_factor}_{rep}.parquet",
            scenario=scenario_list,
            i0=config["simulations"]["i0"],
            scaling_factor=config["empirical_lag"]["scaling_factors"],
            rep=np.arange(config["simulations"]["n_rep"])
        )

rule diagnostics:
    input:
        expand(
            "pipeline/output/coalescent/{scenario}_{i0}.png",
            scenario=scenario_list,
            i0=config["simulations"]["i0"],
        ),
        expand(
            "pipeline/out/infections/{scenario}_{i0}.png",
            scenario=scenario_list,
            i0=config["simulations"]["i0"],
        ),
        expand(
            "pipeline/output/rt/{scenario}.png",
            scenario=scenario_list,
        ),
        expand(
            "pipeline/output/lag/{scaling_factor}.png",
            scaling_factor=config["empirical_lag"]["scaling_factors"],
        ),

rule hash_scenarios:
    output:
        "pipeline/output/rt/hash.json"

    run:
        hash_dict = {
            file.split(".")[0] : i
            for i,file in enumerate(os.listdir("pipeline/input/rt"))
            if file != ".placeholder"
        }
        with open(
            output[0],
            "w",
        ) as outfile:
            json.dump(hash_dict, outfile)

rule plot_rt_diagnostic:
    input:
        "pipeline/input/rt/{scenario}.txt"

    output:
        "pipeline/output/rt/{scenario}.png",

    shell:
        "python3 -m pipeline.plot_rt --config pipeline/config.json --scenario {wildcards.scenario} --infile {input} --outfile {output}"

rule generate_infections:
    input:
        "pipeline/input/rt/{scenario}.txt"

    output:
        "pipeline/output/infections/{scenario}_{i0}.json"

    shell:
        "python3 -m pipeline.generate_infections --config pipeline/config.json --scenario {wildcards.scenario} --i0 {wildcards.i0} --infile {input} --outfile {output}"

rule plot_infection_diagnostics:
    input:
        "pipeline/output/infections/{scenario}_{i0}.json"

    output:
        "pipeline/output/infections/{scenario}_{i0}.png"

    shell:
        "python3 -m pipeline.plot_infections --config pipeline/config.json --infile {input} --outfile {output}"

rule plot_coalescent_math_diagnostics:
    input:
        "pipeline/out/infections/incidence_{scenario}_{i0}.txt",
        "pipeline/out/infections/prevalence_{scenario}_{i0}.txt",

    output:
        "pipeline/out/coalescent/{scenario}_{i0}.png"

    shell:
        "python3 -m pipeline.plot_coalescent_math --config pipeline/config.json --infile {input} --outfile {output}"

rule fit_lag:
    output:
        "pipeline/output/lag/fit.json",

    shell:
        "python3 -m pipeline.fit_lag --config pipeline/config.json --outfile {output}"

rule plot_lag_diagnostic:
    input:
        "pipeline/output/lag/fit.json",
        "pipeline/output/rt/hash.json",

    output:
        "pipeline/output/lag/{scaling_factor}.png",

    shell:
        "python3 -m pipeline.plot_lag --config pipeline/config.json --scaling_factor {wildcards.scaling_factor} --infile {input} --outfile {output}"

rule simulate_data:
    input:
        "pipeline/output/infections/{scenario}_{i0}.json",
        "pipeline/output/lag/fit.json",
        "pipeline/output/rt/hash.json",

    output:
        "pipeline/output/coalescent/{scenario}_{i0}_{scaling_factor}_{rep}.json"

    shell:
        "python3 -m pipeline.simulate_data --config pipeline/config.json --scenario {wildcards.scenario} --i0 {wildcards.i0} --scaling_factor {wildcards.scaling_factor} --rep {wildcards.rep} --infile {input} --outfile {output}"

rule analyze:
    input:
        "pipeline/output/coalescent/{scenario}_{i0}_{scaling_factor}_{rep}.json",

    output:
        "pipeline/output/analysis/rt_{scenario}_{i0}_{scaling_factor}_{rep}.parquet",
        "pipeline/output/analysis/convergence_{scenario}_{i0}_{scaling_factor}_{rep}.json",

    shell:
        "python3 -m pipeline.analyze --config pipeline/config.json --infile {input} --outfile {output}"

rule summarize:
    input:
        directory("pipeline/output/analysis"),
        "pipeline/output/rt/hash.json",

    output:
        "pipeline/output/results.parquet",
        "pipeline/output/rt_est.png",
        "pipeline/output/rt_error.png",

    shell:
        "python3 -m pipeline.summarize --config pipeline/config.json"
