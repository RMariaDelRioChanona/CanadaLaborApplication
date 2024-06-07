from typing import Optional

from pydantic import BaseModel, PositiveInt, Field


class LaborSettings(BaseModel):
    separation_rate: float = Field(ge=0, le=1)
    opening_rate: float = Field(ge=0, le=1)
    adaptation_rate_u: float = Field(ge=0, le=1)
    adaptation_rate_v: float = Field(ge=0, le=1)

    otjob_search_prob: float = Field(ge=0, le=1)

    n_applications_emp: PositiveInt
    n_applications_unemp: PositiveInt


class ModelConfiguration(BaseModel):
    labor: LaborSettings
    t_max: PositiveInt
    n: PositiveInt
    seed: Optional[PositiveInt] = None
