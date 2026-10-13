"""Extract label-verified MFCAD face geometry features with OCP (CadQuery OpenCascade).

Run from the PS04_Manufacturing_AI repository root. No pickle execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickletools
import random
from collections import defaultdict
from pathlib import Path

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box

N_CLASSES = 16
FEATURE_NAMES = [
    'rectangular_through_slot', 'triangular_through_slot',
    'rectangular_passage', 'triangular_passage', '6sides_passage',
    'rectangular_through_step', '2sides_through_step', 'slanted_through_step',
    'rectangular_blind_step', 'triangular_blind_step',
    'rectangular_blind_slot', 'rectangular_pocket', 'triangular_pocket',
    '6sides_pocket', 'chamfer', 'stock',
]
SAFE_PICKLE_OPS = {
    'PROTO', 'EMPTY_LIST', 'BINPUT', 'LONG_BINPUT', 'MARK',
    'BININT1', 'BININT2', 'BININT', 'APPENDS', 'APPEND', 'STOP',
    'MEMOIZE', 'FRAME'
}
COLS = [
    'part_id', 'face_id', 'label', 'surface_type', 'log_area', 'area_ratio',
    'center_x', 'center_y', 'center_z', 'bbox_dx', 'bbox_dy', 'bbox_dz',
    'normal_x', 'normal_y', 'normal_z', 'edge_count', 'neighbor_count',
]


def read_face_labels(path: Path) -> list[int]:
    """Decode just a primitive list-of-integers protocol. No pickle.loads()."""
    ops = list(pickletools.genops(path.read_bytes()))
    unknown = sorted({op.name for op, _, _ in ops if op.name not in SAFE_PICKLE_OPS})
    if unknown:
        raise ValueError(f'Unexpected pickle opcodes: {unknown}')
    if not ops or ops[-1][0].name != 'STOP' or not any(op.name == 'EMPTY_LIST' for op, _, _ in ops):
        raise ValueError('Expected primitive list pickle')
    labels = [int(x) for op, x, _ in ops if op.name in {'BININT1', 'BININT2', 'BININT'}]
    if not labels or any(not 0 <= x < N_CLASSES for x in labels):
        raise ValueError('Missing or invalid face labels')
    return labels


def coords_bbox(shape):
    bbox = Bnd_Box()
    BRepBndLib.Add_s(shape, bbox, False)
    minimum = bbox.CornerMin()
    maximum = bbox.CornerMax()
    x1, y1, z1 = minimum.X(), minimum.Y(), minimum.Z()
    x2, y2, z2 = maximum.X(), maximum.Y(), maximum.Z()
    return (x1, y1, z1), (x2, y2, z2)


def extract_part(step_path: Path) -> list[dict]:
    label_path = step_path.with_suffix('.face_truth')
    if not label_path.is_file():
        raise ValueError('Missing .face_truth file')
    labels = read_face_labels(label_path)
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone or reader.TransferRoots() < 1:
        raise ValueError('STEP file could not be read/transferred')
    shape = reader.OneShape()
    low, high = coords_bbox(shape)
    dims = [max(hi - lo, 1e-9) for lo, hi in zip(low, high)]
    diagonal = max(math.sqrt(sum(d*d for d in dims)), 1e-9)
    global_bbox_area = max(2 * (dims[0]*dims[1] + dims[0]*dims[2] + dims[1]*dims[2]), 1e-9)
    transfer = reader.WS().TransferReader()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    raw = []
    edge_to_faces = defaultdict(set)
    seen_ids = set()

    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        entity = transfer.EntityFromShapeResult(face, 1)
        if entity is None:
            raise ValueError('Unmapped STEP face')
        face_name = entity.Name().ToCString().strip()
        if not face_name.isdecimal():
            raise ValueError(f'Unnamed/non-numeric face ID: {face_name!r}')
        face_id = int(face_name)
        if face_id in seen_ids:
            raise ValueError(f'Duplicate face ID: {face_id}')
        seen_ids.add(face_id)
        if face_id < 0 or face_id >= len(labels):
            raise ValueError(f'Face ID {face_id} outside annotation length {len(labels)}')

        prop = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, prop)
        area = float(prop.Mass())
        center = prop.CentreOfMass().Coord()
        low_face, high_face = coords_bbox(face)
        adaptor = BRepAdaptor_Surface(face)
        kind = str(adaptor.GetType()).split('.')[-1].replace('GeomAbs_', '')
        normal = (0., 0., 0.)
        if kind == 'Plane':
            normal = adaptor.Plane().Axis().Direction().Coord()
            if face.Orientation().__str__().endswith('REVERSED'):
                normal = tuple(-v for v in normal)

        edges = set()
        e = TopExp_Explorer(face, TopAbs_EDGE)
        while e.More():
            edge = TopoDS.Edge(e.Current())
            key = hash(edge)
            edges.add(key)
            edge_to_faces[key].add(face_id)
            e.Next()

        rec = {
            'part_id': step_path.stem,
            'face_id': face_id,
            'label': labels[face_id],
            'surface_type': kind,
            'log_area': math.log1p(max(area, 0)),
            'area_ratio': area / global_bbox_area,
            'center_x': (center[0] - low[0]) / dims[0],
            'center_y': (center[1] - low[1]) / dims[1],
            'center_z': (center[2] - low[2]) / dims[2],
            'bbox_dx': (high_face[0] - low_face[0]) / diagonal,
            'bbox_dy': (high_face[1] - low_face[1]) / diagonal,
            'bbox_dz': (high_face[2] - low_face[2]) / diagonal,
            'normal_x': normal[0],
            'normal_y': normal[1],
            'normal_z': normal[2],
            'edge_count': len(edges),
        }
        raw.append((rec, edges))
        explorer.Next()

    if seen_ids != set(range(len(labels))):
        raise ValueError(f'Face IDs do not exactly match annotation indexing: {len(seen_ids)} vs {len(labels)}')

    for rec, edges in raw:
        neighbors = set().union(*(edge_to_faces[e] for e in edges)) if edges else set()
        neighbors.discard(rec['face_id'])
        rec['neighbor_count'] = len(neighbors)
        for k in COLS[3:]:
            if k != 'surface_type' and not math.isfinite(float(rec[k])):
                raise ValueError(f'Non-finite feature: {k}')
    return [rec for rec, _ in raw]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=Path('data/external/mfcad_original/dataset/step'))
    p.add_argument('--output', type=Path, default=Path('research_outputs/mfcad_face_features.csv'))
    p.add_argument('--max-parts', type=int, default=200, help='0 means all 15,488 CAD parts')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    steps = sorted(args.dataset.glob('*.step'))
    if not steps:
        raise SystemExit(f'No STEP files found at: {args.dataset.resolve()}')
    random.Random(args.seed).shuffle(steps)
    if args.max_parts:
        steps = steps[:args.max_parts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    part_count = face_count = 0
    with args.output.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        for idx, path in enumerate(steps, 1):
            try:
                records = extract_part(path)
                writer.writerows(records)
                part_count += 1
                face_count += len(records)
            except Exception as exc:
                failures.append({'part': path.name, 'error': str(exc)})
            if idx % 25 == 0 or idx == len(steps):
                print(f'{idx}/{len(steps)} parts inspected | {part_count} valid | {face_count} faces | {len(failures)} skipped', flush=True)
    report = {'requested_parts': len(steps), 'valid_parts': part_count, 'extracted_faces': face_count,
              'skipped_parts': failures, 'dataset_path': str(args.dataset.resolve()),
              'output_csv': str(args.output.resolve())}
    report_path = args.output.with_suffix('.audit.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Feature CSV: {args.output}\nAudit: {report_path}')
    if part_count < 10:
        raise SystemExit('Too few valid CAD parts; inspect audit JSON, do NOT train.')

if __name__ == '__main__':
    main()


