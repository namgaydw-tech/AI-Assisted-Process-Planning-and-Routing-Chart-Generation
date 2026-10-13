"""PS 04 research baseline FastAPI server."""
from pathlib import Path
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.models import PlanRequest, PlanResponse
from app.planner import plan
from app.cad import analyze_step

app = FastAPI(title="PS04 Manufacturing Process Planner", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0", "training_status": "no trained ML model"}


@app.get("/api/catalog")
def catalog():
    return {
        "feature_types": ["planar_face", "hole", "pocket", "slot"],
        "materials": ["aluminum", "steel"],
        "machines": ["cnc_mill", "drill_press"],
        "tools": ["face_mill", "spot_drill", "twist_drill", "end_mill", "slot_mill"],
        "note": "Rules and time estimates are demonstration assumptions, not calibrated CAM recommendations.",
    }


@app.post("/api/plan", response_model=PlanResponse)
def create_plan(payload: PlanRequest):
    try:
        return plan(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/analyze-step")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in {".step", ".stp"}:
        raise HTTPException(422, "Upload a STEP file (.step or .stp)")
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "STEP files must be 10 MB or smaller")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ("uploaded" + ext)
            path.write_bytes(data)
            return analyze_step(path)
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
