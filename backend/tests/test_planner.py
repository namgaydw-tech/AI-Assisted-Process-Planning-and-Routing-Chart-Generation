import pytest
from pydantic import ValidationError
from app.models import PlanRequest
from app.planner import plan


def demo(**kwargs):
    body = {
        "part_name": "Test bracket", "material": "aluminum",
        "features": [
            {"id": "f1", "kind": "hole", "diameter_mm": 8, "depth_mm": 16},
            {"id": "f2", "kind": "pocket", "length_mm": 30, "width_mm": 20, "depth_mm": 6},
        ],
    }
    body.update(kwargs)
    return PlanRequest(**body)


def test_four_operations_and_valid_precedence():
    out = plan(demo())
    assert len(out.optimized.operations) == 4
    index = {o.id: i for i, o in enumerate(out.optimized.operations)}
    assert all(index[p] < index[o.id] for o in out.optimized.operations for p in o.predecessors)


def test_exact_algorithm_not_worse():
    out = plan(demo())
    assert out.optimized.summary.total_estimated_minutes <= out.baseline.summary.total_estimated_minutes
    assert out.improvement_percent >= 0


def test_deterministic():
    assert plan(demo()).model_dump() == plan(demo()).model_dump()


def test_absent_mill_rejects_pocket():
    with pytest.raises(ValueError, match="machine"):
        plan(demo(available_machines=["drill_press"]))


def test_missing_tool_rejects():
    with pytest.raises(ValueError, match="Missing tool"):
        plan(demo(available_tools=["spot_drill", "twist_drill"]))


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        demo(features=[{"id": "x", "kind": "hole"}, {"id": "x", "kind": "slot"}])


def test_negative_dimension_rejected():
    with pytest.raises(ValidationError):
        demo(features=[{"id": "x", "kind": "hole", "diameter_mm": -3}])


def test_greedy_large_instance():
    many = [{"id": f"h{i}", "kind": "hole"} for i in range(8)]
    output = plan(demo(features=many))
    assert "greedy" in output.optimized.summary.method
    assert len(output.optimized.operations) == 16
