"""Export MFCAD face-adjacency graphs with exact STEP face-ID validation.

Needs OCP and the original research_training/extract_mfcad.py module.
Safe: read_face_labels uses restricted pickle opcodes only; never unpickles.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD
from OCP.TopoDS import TopoDS

from extract_mfcad import read_face_labels


def graph_from_step(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = read_face_labels(path.with_suffix('.face_truth'))
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() < 1:
        raise ValueError('STEP model could not be read')
    shape = reader.OneShape()
    transfer = reader.WS().TransferReader()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    seen = set()
    edges_to_faces: dict[int, set[int]] = defaultdict(set)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        entity = transfer.EntityFromShapeResult(face, 1)
        if entity is None:
            raise ValueError('Missing STEP transfer entity for face')
        name = entity.Name().ToCString().strip()
        if not name.isdecimal():
            raise ValueError(f'Non-numeric STEP face ID: {name}')
        fid = int(name)
        if fid in seen or fid >= len(labels):
            raise ValueError(f'Duplicate/out-of-range face ID: {fid}')
        seen.add(fid)
        exp_edge = TopExp_Explorer(face, TopAbs_EDGE)
        while exp_edge.More():
            edge = TopoDS.Edge(exp_edge.Current())
            # Normalize OCC orientation, so oppositely-oriented shared edges
            # map to the same topological edge identity within this shape.
            normalized = edge.Oriented(TopAbs_FORWARD)
            edges_to_faces[hash(normalized)].add(fid)
            exp_edge.Next()
        explorer.Next()
    if seen != set(range(len(labels))):
        raise ValueError('STEP face IDs do not match complete .face_truth array')
    undirected = set()
    for owners in edges_to_faces.values():
        if len(owners) > 2:
            raise ValueError(f'Non-manifold edge shared by {len(owners)} faces')
        if len(owners) == 2:
            a, b = sorted(owners)
            undirected.add((a, b))
    directed = sorted([(a,b) for a,b in undirected] + [(b,a) for a,b in undirected])
    index = np.asarray(directed, dtype=np.int32).reshape(-1,2).T
    return np.arange(len(labels), dtype=np.int32), index


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=Path('data/external/mfcad_original/dataset/step'))
    p.add_argument('--out', type=Path, default=Path('research_outputs/graphs/mfcad_adjacency'))
    p.add_argument('--max-parts', type=int, default=20, help='0 = all parts')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.max_parts < 0:
        raise SystemExit('--max-parts must be >= 0')
    paths = sorted(args.dataset.glob('*.step'))
    if not paths:
        raise SystemExit('No STEP files found')
    random.Random(args.seed).shuffle(paths)
    if args.max_parts:
        paths = paths[:args.max_parts]
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    total_nodes = total_directed = isolates = completed = 0
    for i,path in enumerate(paths,1):
        try:
            face_ids, edge_index = graph_from_step(path)
            neighbors = set(edge_index.flatten().tolist())
            isolates += len(face_ids)-len(neighbors)
            total_nodes += len(face_ids)
            total_directed += edge_index.shape[1]
            np.savez_compressed(args.out/f'{path.stem}.npz', face_ids=face_ids, edge_index=edge_index)
            completed += 1
        except Exception as exc:
            failed.append({'part':path.name,'error':str(exc)})
        if i % 25 == 0 or i == len(paths):
            print(f'{i}/{len(paths)} inspected | {completed} graphs | {total_nodes} faces | {total_directed} directed edges | {len(failed)} failed', flush=True)
    audit = {
        'requested':len(paths), 'graphs_written':completed, 'faces':total_nodes,
        'directed_edges':total_directed, 'isolated_faces':isolates,'skipped':failed,
        'note':'Exact per-face adjacency: shared OCC B-rep edges. Does not contain labels.',
    }
    report = args.out.parent/'mfcad_adjacency.audit.json'
    report.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('AUDIT:',report)
    if failed:
        raise SystemExit(f'{len(failed)} graphs failed. Inspect audit before GNN training.')

if __name__ == '__main__':
    main()
