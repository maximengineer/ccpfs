"""Pydantic request/response models for the CCPFS API."""

from pydantic import BaseModel, Field


# --- Requests ---

class ScheduleRequest(BaseModel):
    capacity: dict[str, int] = Field(
        default_factory=lambda: {
            "cardiology": 15,
            "neurology": 10,
            "surgery": 15,
            "general_medicine": 25,
        },
        description="Base daily slot capacity per specialty (scaled to cohort size at runtime).",
    )


# --- Responses ---

class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    patients: int


class ModelInfo(BaseModel):
    name: str
    c_index: float
    ibs: float
    params: dict | None = None


class PolicyResult(BaseModel):
    name: str
    capacity_aware: bool
    feasible: bool
    avg_cost: float
    vs_uniform_pct: float
    catch_rate: float
    ebf_rate: float


class CohortSummary(BaseModel):
    total_episodes: int
    test_episodes: int
    event_rate: float
    specialties: dict[str, float]


class PipelineResults(BaseModel):
    cohort: CohortSummary
    models: list[ModelInfo]
    policies: list[PolicyResult]


class PatientCurve(BaseModel):
    patient_index: int
    survival_curve: list[float]
    assigned_day: int
    specialty: str
    specialty_index: int
    event_indicator: bool
    time_to_event: float
    cost: float


class ScheduleMetrics(BaseModel):
    avg_cost: float
    catch_rate: float
    ebf_rate: float
    vs_uniform_pct: float
    elapsed_ms: float


class ScheduleResponse(BaseModel):
    day_histogram: list[int]
    by_specialty: dict[str, list[int]]
    metrics: ScheduleMetrics
