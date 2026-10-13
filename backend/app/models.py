"""Validated inputs and outputs for a research *baseline* process planner."""
from typing import Literal
from pydantic import BaseModel, Field, model_validator

Kind = Literal["planar_face", "hole", "pocket", "slot"]
Machine = Literal["cnc_mill", "drill_press"]
Tool = Literal["face_mill", "spot_drill", "twist_drill", "end_mill", "slot_mill"]


class Feature(BaseModel):
    id: str = Field(min_length=1, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    kind: Kind
    quantity: int = Field(default=1, ge=1, le=100)
    diameter_mm: float = Field(default=8, gt=0, le=1000)
    depth_mm: float = Field(default=12, gt=0, le=1000)
    length_mm: float = Field(default=40, gt=0, le=5000)
    width_mm: float = Field(default=20, gt=0, le=5000)


class PlanRequest(BaseModel):
    part_name: str = Field(default="Demo bracket", min_length=1, max_length=100)
    material: Literal["aluminum", "steel"] = "aluminum"
    features: list[Feature] = Field(min_length=1, max_length=30)
    available_machines: list[Machine] = Field(default_factory=lambda: ["cnc_mill", "drill_press"])
    available_tools: list[Tool] = Field(default_factory=lambda: ["face_mill", "spot_drill", "twist_drill", "end_mill", "slot_mill"])

    @model_validator(mode="after")
    def validate_constraints(self):
        ids = [f.id for f in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError("Feature IDs must be unique")
        if not self.available_machines:
            raise ValueError("At least one available machine is required")
        if len(self.available_machines) != len(set(self.available_machines)):
            raise ValueError("Duplicate machine names are not allowed")
        if len(self.available_tools) != len(set(self.available_tools)):
            raise ValueError("Duplicate tools are not allowed")
        return self


class Operation(BaseModel):
    id: str
    feature_id: str
    action: str
    machine: Machine
    tool: Tool
    process_minutes: float
    predecessors: list[str]
    note: str


class RouteSummary(BaseModel):
    process_minutes: float
    changeover_minutes: float
    total_estimated_minutes: float
    tool_changes: int
    machine_changes: int
    operation_count: int
    method: str


class Route(BaseModel):
    operations: list[Operation]
    summary: RouteSummary


class PlanResponse(BaseModel):
    part_name: str
    material: str
    baseline: Route
    optimized: Route
    improvement_percent: float
    scope_note: str
