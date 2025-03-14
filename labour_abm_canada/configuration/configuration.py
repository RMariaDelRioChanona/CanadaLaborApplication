from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, PositiveInt


class LaborSettings(BaseModel):
    """
    Parameters for the labor market model.

    Parameters
    ----------
    separation_rate : float
        Rate at which employees separate from their jobs. Must be between 0 and 1.
    opening_rate : float
        Rate at which jobs open up. Must be between 0 and 1.
    adaptation_rate_u : float
        Rate at which unemployed workers adapt to job search. Must be between 0 and 1.
    adaptation_rate_v : float
        Rate at which employed workers adapt to job search. Must be between 0 and 1.
    otjob_search_prob : float
        Probability of on-the-job search. Must be between 0 and 1.
    n_applications_emp : int
        Number of applications employed workers make.
    n_applications_unemp : int
        Number of applications unemployed workers make.
    """

    separation_rate: float = Field(ge=0, le=1)
    opening_rate: float = Field(ge=0, le=1)
    adaptation_rate_u: float = Field(ge=0, le=1)
    adaptation_rate_v: float = Field(ge=0, le=1)

    otjob_search_prob: float = Field(ge=0, le=1)

    n_applications_emp: PositiveInt
    n_applications_unemp: PositiveInt

    @classmethod
    def from_yaml(
        cls,
        fname: Path | str,
        default_otj_search_prob=0.01,
        default_n_applications_unemp=10,
        default_n_applications_emp=1,
    ):
        with open(fname, "r") as f:
            model_params = yaml.safe_load(f)
        return cls(
            separation_rate=model_params["separation_rate"],
            opening_rate=model_params["opening_rate"],
            adaptation_rate_u=model_params["adaptation_rate_u"],
            adaptation_rate_v=model_params["adaptation_rate_v"],
            otjob_search_prob=model_params.get("otjob_search_prob", default_otj_search_prob),
            n_applications_emp=model_params.get("n_applications_emp", default_n_applications_emp),
            n_applications_unemp=model_params.get("n_applications_unemp", default_n_applications_unemp),
        )


class ModelConfiguration(BaseModel):
    """
    Configuration for the labor market model.

    Parameters
    ----------
    labor : LaborSettings
        Labor market model parameters.
    t_max : PositiveInt
        Number of time steps to run the model for.
    n : PositiveInt
        Number of occupations.
    seed : PositiveInt, optional
        Seed for random number generation
    """

    labor: LaborSettings
    t_max: PositiveInt
    n: PositiveInt
    seed: Optional[PositiveInt] = None
