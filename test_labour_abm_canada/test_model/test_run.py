import pytest
import torch

from labour_abm_canada.model.labour_abm import LabourABM
from labour_abm_canada.regions.regions import Regions
from labour_abm_canada.runner import run_model


class TestModel:
    def test_run(self, twonode_model: LabourABM):
        twonode_model.run_model()

        aggregates = twonode_model.aggregates

        assert torch.all(aggregates.total_unemployment >= 0)
        assert torch.all(aggregates.total_vacancies >= 0)
        assert torch.all(aggregates.total_employment >= 0)
        assert torch.all(aggregates.total_demand >= 0)


class TestRunner:

    @pytest.mark.parametrize("region", Regions)
    def test_run(self, default_scenario_path, default_runner_config, region: Regions):
        run_model(scenario_filename=default_scenario_path, region=region, burn_in=0)
        assert True
