"""Pydantic request and response schemas."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from app.ml.preprocessing import FEATURES

class MobileSpecifications(BaseModel):
    model_config = ConfigDict(extra="forbid")
    battery_power: int = Field(ge=1, le=10000)
    blue: int = Field(ge=0, le=1)
    clock_speed: float = Field(ge=0, le=10)
    dual_sim: int = Field(ge=0, le=1)
    fc: int = Field(ge=0, le=100)
    four_g: int = Field(ge=0, le=1)
    int_memory: int = Field(ge=1, le=1024)
    m_dep: float = Field(ge=0, le=10)
    mobile_wt: int = Field(ge=1, le=1000)
    n_cores: int = Field(ge=1, le=32)
    pc: int = Field(ge=0, le=100)
    px_height: int = Field(ge=0, le=10000)
    px_width: int = Field(ge=0, le=10000)
    ram: int = Field(ge=1, le=100000)
    sc_h: int = Field(ge=1, le=100)
    sc_w: int = Field(ge=0, le=100)
    talk_time: int = Field(ge=0, le=100)
    three_g: int = Field(ge=0, le=1)
    touch_screen: int = Field(ge=0, le=1)
    wifi: int = Field(ge=0, le=1)

class PredictionRequest(BaseModel):
    mobile: MobileSpecifications
    model: Literal["Random Forest", "Logistic Regression", "Neural Network", "Compare All"] = "Random Forest"

class PredictionResponse(BaseModel):
    results: list[dict]
