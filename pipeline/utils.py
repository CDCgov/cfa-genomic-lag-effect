import argparse
import json
from typing import Optional


def construct_seed(
    root_seed: int,
    scenario: Optional[str],
    i0: Optional[str],
    scaling_factor: Optional[str],
    rep: Optional[str],
):
    """
    Allows for consistent and unique seeds across parameter grid
    """
    scenario_to_int = {
        None: "0",
        "decreasing": "1",
        "notrend": "2",
        "increasing": "3",
    }
    i0_to_int = {
        None: "0",
        "1000": "1",
        "2000": "2",
        "4000": "3",
    }
    scale_to_int = {
        None: "0",
        "0.0": "1",
        "0.25": "2",
        "0.5": "3",
        "0.75": "4",
        "1.0": "5",
    }
    rep_str = "0" if rep is None else str(int(rep) + 1)
    return int(
        str(root_seed)
        + scenario_to_int[scenario]
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
    help="Output file.",
)
