from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["training_status"] == "no trained ML model"


def test_plan_endpoint():
    resp = client.post("/api/plan", json={"features": [{"id": "hole1", "kind": "hole"}]})
    assert resp.status_code == 200
    assert resp.json()["optimized"]["summary"]["operation_count"] == 2


def test_invalid_machine():
    resp = client.post("/api/plan", json={"features": [{"id": "a", "kind": "pocket"}], "available_machines": ["drill_press"]})
    assert resp.status_code == 422


def test_invalid_step_extension():
    resp = client.post("/api/analyze-step", files={"file": ("bad.txt", b"foo", "text/plain")})
    assert resp.status_code == 422
