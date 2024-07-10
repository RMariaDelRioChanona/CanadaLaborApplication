import copy
from pathlib import Path

import numpy as np
import torch

from labour_abm_canada.configuration.configuration import LaborSettings, ModelConfiguration
from labour_abm_canada.data_bridge.bridge import DataBridge
from labour_abm_canada.model import labour_abm as lbm
from labour_abm_canada.regions.regions import Regions

DEFAULT_MODEL_PARAMS = Path(__file__).parent / "model-params.yaml"


def run_model(
    scenario_filename: str | Path,
    region: Regions = "National",
    burn_in: int = 1_000,
    model_parameters: str | Path = DEFAULT_MODEL_PARAMS,
    seed: int = 123,
) -> dict:
    """
    Instantiates and runs the model.

    Args:
        scenario_filename (str | Path): Path to the scenario file.
        region (str): Region to run the model for.
        burn_in (int): Number of burn-in steps.
        model_parameters (str | Path): Model parameters file.
        seed (int): Seed for random number generation.
    """

    # Generate model inputs
    print("Generating model inputs...")
    databridge = DataBridge.from_standard_files(scenario_file=scenario_filename)
    model_inputs = databridge.generate_model_inputs(region, burn_in=burn_in, smooth=6)

    # Set seed for reproducibility
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

    results["simulation-results"] = dict(
        total_unemployment=total_unemployment.numpy().tolist(),
        total_vacancies=total_vacancies.numpy().tolist(),
        total_employment=total_employment.numpy().tolist(),
        total_demand=total_demand.numpy().tolist(),
        d_dagger=d_dagger.numpy().tolist(),
        D_dagger=D_dagger.numpy().tolist(),
    )

    print("Saving results...")
    return results
