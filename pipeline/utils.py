import argparse
import json
import os
from typing import Optional


def construct_seed(
    root_seed: int,
    config: dict,
    scenario: Optional[str],
    i0: Optional[str],
    scaling_factor: Optional[str],
    rep: Optional[str],
):
    """
    Allows for consistent and unique seeds across parameter grid
    """
    scenario_hash_file = "pipeline/output/rt/hash.json"
    assert os.path.isfile(
        scenario_hash_file
    ), f"Cannot construct seed without {scenario_hash_file}"
    with open(scenario_hash_file, "r") as file:
        scenario_hash = json.load(file)

    scenario_to_int = {None: "0"} | {
        k: str(v + 1) for k, v in scenario_hash.items()
    }

    i0_to_int = {None: "0"} | {str(x) : str(i) for i,x in enumerate(config["simulations"]["i0"])}
    scale_to_int = {None: "0"} | {str(x) : str(i) for i,x in enumerate(config["empirical_lag"]["scaling_factors"])}
    rep_str = "0" if rep is None else str(int(rep) + 1)
    return int(
        str(root_seed)
        + scenario_to_int[scenario]  # type: ignore
        + i0_to_int[i0]
        + scale_to_int[scaling_factor]
        + rep_str
    )


def read_config(path):
    with open(path, "r") as file:
        config = json.load(file)
    return config


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    help="Path to JSON configuration file",
)

parser.add_argument(
    "--scenario",
    type=str,
    help="The Rt scenario.",
)

parser.add_argument(
    "--i0",
    type=str,
    help="Starting incident infections for initializing renewal model.",
)

parser.add_argument(
    "--scaling_factor",
    type=str,
    help="Scale the empirical lag distribution up or down by a factor.",
)

parser.add_argument(
    "--rep",
    type=str,
    help="Replicate index for multiple simulations/analyses at a given parameter combination.",
)

parser.add_argument(
    "--infile",
    type=str,
    help="Input file(s).",
    nargs="*",
)

parser.add_argument(
    "--outfile",
    type=str,
    help="Output file(s).",
    nargs="*",
)
