"""Optional STEP geometry summary. Does NOT infer machining features.

CadQuery is optional: install requirements-cad-optional.txt for this endpoint.
"""
from pathlib import Path


def analyze_step(path: Path) -> dict:
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError(
            "STEP reader not installed. Optional: pip install -r requirements-cad-optional.txt "
            "in a supported Python 3.11/3.12 environment. Manual-feature planning works without it."
        ) from exc
    try:
        workplane = cq.importers.importStep(str(path))
        solids = workplane.solids().vals()
        if not solids:
            raise ValueError("STEP did not contain a readable solid")
        compound = cq.Compound.makeCompound(solids)
        bb = compound.BoundingBox()
        faces = compound.Faces()
        cylinder_count = sum(1 for face in faces if face.geomType() == "CYLINDER")
        return {
            "source": path.name,
            "solid_count": len(solids),
            "face_count": len(faces),
            "cylindrical_face_count": cylinder_count,
            "bounding_box_mm": {
                "x": round(bb.xlen, 3), "y": round(bb.ylen, 3), "z": round(bb.zlen, 3),
            },
            "volume_mm3": round(compound.Volume(), 3),
            "classification_note": (
                "Geometry metadata only: cylindrical faces are NOT automatically classified "
                "as holes; enter verified manufacturing features manually for route planning."
            ),
        }
    except Exception as exc:
        raise ValueError(f"Could not interpret STEP solids: {exc}") from exc
