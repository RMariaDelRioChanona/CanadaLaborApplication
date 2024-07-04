from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, PositiveInt


class LaborSettings(BaseModel):
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
    labor: LaborSettings
    t_max: PositiveInt
    n: PositiveInt
    seed: Optional[PositiveInt] = None
