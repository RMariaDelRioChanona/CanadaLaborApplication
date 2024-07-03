import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from data_bridge import generic_loader
from data_bridge.bridge import DataBridge

from . import labor_abm as lbm

FILE_PATH = Path(__file__).parent
PROJECT_PATH = FILE_PATH.parent
DATA_PATH = PROJECT_PATH / "data"

REGIONS = [
    "Alberta.a",
    "British Columbia.a",
    "Manitoba.a",
    "New Brunswick.a",
    "Newfoundland and Labrador.a",
    "Nova Scotia.a",
    "Ontario.a",
    "Ontario.b",
    "Prince Edward Island.a",
    "Quebec.a",
    "Quebec.b",
    "Saskatchewan.a",
    "National",
]

GDP_SEASONAL_BAROMETER = DATA_PATH / "u_v_gdp_seasonal_barometer.csv"
DEFAULT_MODEL_PARAMS = FILE_PATH / "model-params.yaml"


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default=None,
        help="Path to the scenario file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./labour_model_results.csv",
        help="Path to the output file.",
    )
    parser.add_argument(
        "-- region",
        type=str,
        default="National",
        help="Region to run the model for.",
    )

    parser.add_argument(
        "--parameters",
        type=str,
        default=str(DEFAULT_MODEL_PARAMS),
        help="Model parameters file.",
    )

    return parser


def run_model(
    scenario_filename: str | Path,
    region: str = "National",
    burn_in: int = 1_000,
    model_parameters: str | Path = DEFAULT_MODEL_PARAMS,
) -> dict:
    """
    Instantiates and runs the model.

    Args:
        scenario_filename (str | Path): Path to the scenario file.
        region (str): Region to run the model for.
        burn_in (int): Number of burn-in steps.
        model_parameters (str | Path): Model parameters file.
    """

    # Generate model inputs
    databridge = DataBridge.from_standard_files(scenario_file=scenario_filename)
    model_inputs = databridge.generate_model_inputs(region, burn_in=burn_in, smooth=6)

    # Set seed for reproducibility
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load parameters
    with open(model_parameters, "r") as f:
        model_params = yaml.safe_load(f)

    # Overwrite some parameters  # TODO: remove this
    # gamma_u = 0.12
    # gamma_v = 0.12

    # Generate network and scenario data

    # Initialize labor ABM  # TODO: is this the right model ??
    lab_abm = lbm.LabourABM(seed=seed, **model_params, **model_inputs)

    # Run the model
    _ = lab_abm.run_model()

    # Aggregate data
    total_unemployment = lab_abm.unemployment.sum(axis=0)
    total_vacancies = lab_abm.vacancies.sum(axis=0)
    total_employment = lab_abm.employment.sum(axis=0)
    total_demand = lab_abm.d_dagger.sum(axis=0)
    d_dagger = lab_abm.d_dagger
    D_dagger = d_dagger.sum(axis=0)

    # Save results

    results = dict()
    results["configs"] = dict(
        scenario_filename=scenario_filename,
        region=region,
        burn_in=burn_in,
        seed=seed,
        model_params=model_params,
    )

    results["simuation-results"] = dict(
        total_unemployment=total_unemployment,
        total_vacancies=total_vacancies,
        total_employment=total_employment,
        total_demand=total_demand,
        d_dagger=d_dagger,
        D_dagger=D_dagger,
    )

    return results


if __name__ == "__main__":

    # Create the parser
    parser = setup_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Run the model
    results = run_model(scenario_filename=args.scenario, region=args.region, model_parameters=args.parameters)

    # Save the results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
