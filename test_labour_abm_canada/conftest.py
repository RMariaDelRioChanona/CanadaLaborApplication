from pathlib import Path

import pytest
import torch
import yaml

from labour_abm_canada.configuration.configuration import ModelConfiguration
from labour_abm_canada.model.labour_abm import LabourABM


@pytest.fixture()
def default_config_path():
    return Path(__file__).parent / "test_data" / "default_config.yaml"


@pytest.fixture()
def default_scenario_path():
    return Path(__file__).parent / "test_data" / "net_new_cap_w_jobs.csv"


@pytest.fixture()
def default_runner_config():
    return Path(__file__).parent / "test_data" / "model_params.yaml"


@pytest.fixture()
def default_config(default_config_path):
    with open(default_config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ModelConfiguration(**config_dict)


@pytest.fixture()
def twonode_model(default_config):

    t_max = default_config.t_max

    twonode_transition = torch.tensor([[0.8, 0.2], [0.3, 0.7]])

    initial_employment = torch.tensor([1200, 800])
    initial_unemployment = 0.05 * initial_employment
    initial_vacancies = 0.05 * initial_employment

    initial_employment = 0.95 * initial_employment

    sum_e_u = initial_employment + initial_unemployment
    d_dagger = sum_e_u.unsqueeze(1).repeat(1, t_max)
    wages = torch.rand([1, 1])

    return LabourABM.default_create(
        model_configuration=default_config,
        transition_matrix=twonode_transition,
        initial_employment=initial_employment,
        initial_unemployment=initial_unemployment,
        initial_vacancies=initial_vacancies,
        demand_scenario=d_dagger,
        wages=wages,
    )
