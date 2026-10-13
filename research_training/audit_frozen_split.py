"""Audit MFCAD split manifest for data integrity, class balance and candidate CAD duplicates.

Python 3.13; deps: pandas if --features-csv provided. Never unpickle third-party data.
This is a preliminary geometry-similarity screen, not a proof of CAD-family independence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import pickletools
from collections import Counter, defaultdict
from pathlib import Path

LABELS = [
    'rectangular_through_slot', 'triangular_through_slot', 'rectangular_passage',
    'triangular_passage', '6sides_passage', 'rectangular_through_step',
    '2sides_through_step', 'slanted_through_step', 'rectangular_blind_step',
    'triangular_blind_step', 'rectangular_blind_slot', 'rectangular_pocket',
    'triangular_pocket', '6sides_pocket', 'chamfer', 'stock'
]
SAFE_OPS = {'PROTO', 'EMPTY_LIST', 'BINPUT', 'LONG_BINPUT', 'MARK', 'BININT1',
            'BININT2', 'BININT', 'APPENDS', 'APPEND', 'STOP', 'MEMOIZE', 'FRAME'}


def safe_face_labels(path: Path) -> list[int]:
    ops = list(pickletools.genops(path.read_bytes()))
    assert ops and ops[-1][0].name == 'STOP'
    if any(op.name not in SAFE_OPS for op, _, _ in ops):
        raise ValueError(f'Nonprimitive pickle in {path}')
    labels = [v for op, v, _ in ops if op.name in {'BININT1','BININT2','BININT'}]
    if not labels or any(not isinstance(v, int) or v not in range(16) for v in labels):
        raise ValueError(f'Invalid labels in {path}')
    return labels


def find_duplicate_signatures(parts_by_split: dict[str, list[str]], root: Path, do_sha: bool):
    if not do_sha:
        return [], 0
    duplicates = defaultdict(list)
    n = 0
    for split, parts in parts_by_split.items():
        for part in parts:
            path = root / (part + '.step')
            h = hashlib.sha256()
            with path.open('rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
            duplicates[h.hexdigest()].append((split, part))
            n += 1
    cross = [locs for locs in duplicates.values() if len({s for s, _ in locs}) > 1]
    return cross, n


def face_geometry_fingerprint(row: dict[str, str]) -> tuple:
    # Translation-insensitive because face centers are normalized to global bbox.
    # Rounded descriptors are ONLY a first-pass heuristic. Fingerprint deliberately
    # excludes label, face_id, filename and raw class annotations.
    cols = ['area_ratio', 'center_x', 'center_y', 'center_z',
            'bbox_dx', 'bbox_dy', 'bbox_dz',
            'normal_x', 'normal_y', 'normal_z']
    numeric = tuple(round(float(row[k]), 3) for k in cols)
    return (row['surface_type'], int(float(row['edge_count'])),
            int(float(row['neighbor_count']))) + numeric


def cross_geometry_candidates(features_csv: Path, parts_by_split: dict[str, list[str]]):
    membership = {p: s for s, ids in parts_by_split.items() for p in ids}
    fingerprints = defaultdict(list)
    rows_by_part = defaultdict(list)
    with features_csv.open('r', newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            pid = row['part_id']
            if pid in membership:
                rows_by_part[pid].append(face_geometry_fingerprint(row))
    for pid, signatures in rows_by_part.items():
        # Sorting removes ordering dependence of CAD faces.
        signature_blob = repr(sorted(signatures)).encode('utf-8')
        fingerprints[hashlib.sha256(signature_blob).hexdigest()].append((membership[pid], pid))
    cross = [ids for ids in fingerprints.values() if len({s for s, _ in ids}) > 1]
    return cross, len(rows_by_part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('research_outputs/splits/mfcad_full_provisional_group_v1.json'))
    parser.add_argument('--dataset', type=Path, default=Path('data/external/mfcad_original/dataset/step'))
    parser.add_argument('--features-csv', type=Path, default=None, help='Optional FULL extracted CSV; screen approximate geometry matches')
    parser.add_argument('--skip-hashes', action='store_true', help='Skip SHA256 scan of every STEP file')
    parser.add_argument('--out', type=Path, default=Path('research_outputs/splits/mfcad_full_provisional_group_v1.audit.json'))
    args = parser.parse_args()
    obj = json.loads(args.manifest.read_text(encoding='utf-8'))
    parts = obj['splits']
    if set(parts) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation, test')
    all_ids = [p for names in parts.values() for p in names]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Duplicated or overlapping part IDs')
    available = {p.stem for p in args.dataset.glob('*.step')}
    if available != set(all_ids):
        raise ValueError(f'Manifest mismatch: {len(available-set(all_ids))} unassigned, {len(set(all_ids)-available)} missing')
    prefix_depth = int(obj['family_prefix_depth'])
    def family(p): return '-'.join(p.split('-')[:prefix_depth])
    families = {s: {family(p) for p in ids} for s, ids in parts.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if families[a] & families[b]:
            raise ValueError(f'Filename proxy group overlap {a} vs {b}')
    listed_digest = hashlib.sha256('\n'.join(sorted(all_ids)).encode('utf-8')).hexdigest()
    if listed_digest != obj.get('source_part_list_sha256'):
        raise ValueError('Source part inventory digest mismatch')

    class_counts = {}
    total_faces = 0
    for split, ids in parts.items():
        counter = Counter()
        for part in ids:
            counter.update(safe_face_labels(args.dataset / f'{part}.face_truth'))
        class_counts[split] = {LABELS[i]: counter[i] for i in range(16)}
        total_faces += sum(counter.values())
        print(f'{split}: {len(ids)} CAD parts, {sum(counter.values())} labeled faces')
    print('\n16-class face counts (before model training):')
    print('Class                         Train      Val      Test')
    for cls in LABELS:
        print(f'{cls:28} {class_counts["train"][cls]:7d} {class_counts["validation"][cls]:8d} {class_counts["test"][cls]:8d}')
    missing = {s: [c for c in LABELS if class_counts[s][c] == 0] for s in parts}
    print('Missing classes:', missing)
    if any(missing.values()):
        print('WARNING: Some split(s) lack at least one class. Revisit protocol before training.')

    content_dups, n_hashed = find_duplicate_signatures(parts, args.dataset, not args.skip_hashes)
    print(f'\nSHA256 scanned {n_hashed} STEP files; cross-split exact-byte duplicates: {len(content_dups)}')
    if content_dups:
        print('Examples:', content_dups[:3])
    geom_cross, geom_n = [], 0
    if args.features_csv:
        geom_cross, geom_n = cross_geometry_candidates(args.features_csv, parts)
        print(f'Geometry fingerprint scanned {geom_n} unique parts; cross-split candidates: {len(geom_cross)}')
        if geom_cross:
            print('Geometry candidate examples:', geom_cross[:3])
        if geom_n != len(all_ids):
            print('WARNING: Geometry check is PARTIAL. Full CSV required to check all parts.')
    else:
        print('Geometry fingerprint NOT checked: pass --features-csv after full extraction.')
    report = dict(split_counts={k:len(v) for k,v in parts.items()}, total_labeled_faces=total_faces,
                  source_digest_verified=True, prefix_family_overlap=0, class_counts=class_counts,
                  missing_classes=missing, step_files_hashed=n_hashed,
                  exact_cross_split_duplicates=content_dups[:30], exact_cross_split_duplicate_groups=len(content_dups),
                  geometry_parts_scanned=geom_n, geometry_cross_split_candidate_groups=len(geom_cross),
                  geometry_cross_split_candidates=geom_cross[:30],
                  caution='Prefix group and numeric face-fingerprint screening cannot prove full geometric independence; manual/geometry-based family review still required.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Audit saved:', args.out)
    if content_dups or geom_cross:
        raise SystemExit('REVIEW REQUIRED: potential split leakage candidates. Do not train until checked.')
    if args.features_csv and geom_n != len(all_ids):
        raise SystemExit('Partial geometry audit only. Need complete extraction for final research validation.')

if __name__ == '__main__':
    main()
