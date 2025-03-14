from dataclasses import dataclass

import torch


@dataclass
class Aggregates:
    total_unemployment: torch.Tensor
    total_vacancies: torch.Tensor
    total_employment: torch.Tensor
    total_demand: torch.Tensor

    total_spontaneous_separations: torch.Tensor
    total_state_separations: torch.Tensor
    total_spontaneous_vacancies: torch.Tensor
    total_state_vacancies: torch.Tensor

    job_to_job_transitions: torch.Tensor
    unemployment_to_job_transitions: torch.Tensor
