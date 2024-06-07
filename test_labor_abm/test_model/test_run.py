import torch

from labor_abm import LaborABM


class TestModel:
    def test_run(self, twonode_model: LaborABM):
        twonode_model.run_model()

        aggregates = twonode_model.aggregates

        assert torch.all(aggregates.total_unemployment >= 0)
        assert torch.all(aggregates.total_vacancies >= 0)
        assert torch.all(aggregates.total_employment >= 0)
        assert torch.all(aggregates.total_demand >= 0)
