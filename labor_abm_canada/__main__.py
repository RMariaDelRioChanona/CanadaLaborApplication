import copy
import json
from argparse import ArgumentParser
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch

from labor_abm_canada.model import labor_abm as lbm
from labor_abm_canada.configuration.configuration import LaborSettings, ModelConfiguration
from labor_abm_canada.data_bridge.bridge import DataBridge

FILE_PATH = Path(__file__).parent
PROJECT_PATH = FILE_PATH.parent
DATA_PATH = PROJECT_PATH / "data"


class Regions(StrEnum):
    Alberta_A = "Alberta.a"
    British_Columbia_A = "British Columbia.a"
    Manitoba_A = "Manitoba.a"
    New_Brunswick_A = "New Brunswick.a"
    Newfoundland_and_Labrador_A = "Newfoundland and Labrador.a"
    Nova_Scotia_A = "Nova Scotia.a"
    Ontario_A = "Ontario.a"
    Ontario_B = "Ontario.b"
    Prince_Edward_Island_A = "Prince Edward Island.a"
    Quebec_A = "Quebec.a"
    Quebec_B = "Quebec.b"
    Saskatchewan_A = "Saskatchewan.a"
    National = "National"


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
        default="./labour_model_results.json",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--region",
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

    parser.add_argument('--aggregate', action='store_true', help='If flagged, returns aggregate data')

    return parser


def run_model(
    scenario_filename: str | Path,
    region: Regions = "National",
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
    print("Generating model inputs...")
    databridge = DataBridge.from_standard_files(scenario_file=scenario_filename)
    model_inputs = databridge.generate_model_inputs(region, burn_in=burn_in, smooth=6)

    # Set seed for reproducibility
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    labor_settings = LaborSettings.from_yaml(model_parameters)

    # Overwrite some parameters  # TODO: remove this
    # gamma_u = 0.12
    # gamma_v = 0.12

    # Generate network and scenario data
    model_required_inputs = copy.deepcopy(model_inputs)
    del model_required_inputs["L"]
    del model_required_inputs["time_indices"]

    model_configuration = ModelConfiguration(
        labor=labor_settings, t_max=model_inputs["t_max"], n=model_inputs["n_occupations"]
    )

    # Initialize labor ABM  # TODO: is this the right model ??
    lab_abm = lbm.LabourABM.default_create(
        model_configuration=model_configuration,
        transition_matrix=model_required_inputs["adjacency_matrix"],
        initial_employment=model_required_inputs["initial_employment"],
        initial_unemployment=model_required_inputs["initial_unemployment"],
        initial_vacancies=model_required_inputs["initial_vacancies"],
        wages=model_required_inputs["wages"],
        demand_scenario=model_required_inputs["d_dagger"],
    )

    # Run the model
    print("Running the model...")
    lab_abm.run_model()

    # Aggregate data
    total_unemployment = lab_abm.unemployment.sum(dim=0)
    total_vacancies = lab_abm.vacancies.sum(dim=0)
    total_employment = lab_abm.employment.sum(dim=0)
    total_demand = lab_abm.demand_scenario.sum(dim=0)
    d_dagger = lab_abm.demand_scenario
    D_dagger = d_dagger.sum(dim=0)

    # Save results
    results = dict()
    results["configs"] = dict(
        scenario_filename=scenario_filename,
        region=region,
        burn_in=burn_in,
        seed=seed,
        model_params=model_configuration.dict(),
    )

    results["simuation-results"] = dict(
        total_unemployment=total_unemployment.numpy().tolist(),
        total_vacancies=total_vacancies.numpy().tolist(),
        total_employment=total_employment.numpy().tolist(),
        total_demand=total_demand.numpy().tolist(),
        d_dagger=d_dagger.numpy().tolist(),
        D_dagger=D_dagger.numpy().tolist(),
    )

    print("Saving results...")
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
