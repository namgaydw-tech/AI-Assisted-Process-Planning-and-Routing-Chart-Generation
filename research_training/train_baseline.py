"""Part-disjoint RandomForest baseline for 16-class MFCAD face-level classification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NAME_MAP = [
    'rectangular_through_slot','triangular_through_slot','rectangular_passage',
    'triangular_passage','6sides_passage','rectangular_through_step',
    '2sides_through_step','slanted_through_step','rectangular_blind_step',
    'triangular_blind_step','rectangular_blind_slot','rectangular_pocket',
    'triangular_pocket','6sides_pocket','chamfer','stock',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, default=Path('research_outputs/mfcad_face_features.csv'))
    p.add_argument('--out', type=Path, default=Path('research_outputs/rf_baseline'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit('Empty dataset')
    df = df.dropna(axis=0)
    parts = df['part_id']
    if parts.nunique() < 15:
        raise SystemExit('Need at least 15 CAD parts for illustrative split; use many more for publication')
    if df[['part_id', 'face_id']].duplicated().any():
        raise SystemExit('Duplicated face IDs in extracted data')
    y = df['label'].astype(int)
    if not y.between(0,15).all():
        raise SystemExit('Invalid MFCAD class encoding')
    X = df.drop(columns=['part_id', 'face_id', 'label'])
    cat = ['surface_type']
    num = [c for c in X.columns if c not in cat]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_idx, rest_idx = next(outer.split(X, y, groups=parts))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    val_rel, test_rel = next(inner.split(X.iloc[rest_idx], y.iloc[rest_idx], groups=parts.iloc[rest_idx]))
    val_idx, test_idx = rest_idx[val_rel], rest_idx[test_rel]
    for a,b in [(train_idx,val_idx),(train_idx,test_idx),(val_idx,test_idx)]:
        assert set(parts.iloc[a]).isdisjoint(set(parts.iloc[b])), 'Part leakage!'
    prep = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
                              ('num', 'passthrough', num)])
    model = Pipeline([('prep', prep),
                      ('clf', RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                                     class_weight='balanced_subsample', n_jobs=-1,
                                                     random_state=args.seed))])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    args.out.mkdir(parents=True, exist_ok=True)
    val_pred = model.predict(X.iloc[val_idx])
    test_pred = model.predict(X.iloc[test_idx])
    def score(indices,pred):
        return {'accuracy': float(accuracy_score(y.iloc[indices], pred)),
                'balanced_accuracy': float(balanced_accuracy_score(y.iloc[indices], pred)),
                'macro_f1': float(f1_score(y.iloc[indices], pred, labels=list(range(16)), average='macro', zero_division=0)),
                'weighted_f1': float(f1_score(y.iloc[indices], pred, average='weighted', zero_division=0)),
                'n_faces': int(len(indices)), 'n_parts': int(parts.iloc[indices].nunique())}
    report = {'dataset': str(args.csv), 'seed': args.seed,
              'caution': 'Initial part-disjoint research baseline; geometric family leakage has not yet been ruled out.',
              'train': {'n_faces': int(len(train_idx)), 'n_parts': int(parts.iloc[train_idx].nunique())},
              'validation': score(val_idx,val_pred), 'test': score(test_idx,test_pred),
              'classes': NAME_MAP,
              'test_report': classification_report(y.iloc[test_idx], test_pred, labels=list(range(16)),
                                                   target_names=NAME_MAP, output_dict=True, zero_division=0)}
    (args.out/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.out/'split_part_ids.json').write_text(json.dumps({
        'train': sorted(parts.iloc[train_idx].unique().tolist()),
        'validation': sorted(parts.iloc[val_idx].unique().tolist()),
        'test': sorted(parts.iloc[test_idx].unique().tolist())}, indent=2), encoding='utf-8')
    cm = confusion_matrix(y.iloc[test_idx], test_pred, labels=list(range(16)))
    pd.DataFrame(cm, index=NAME_MAP, columns=NAME_MAP).to_csv(args.out/'test_confusion_matrix.csv')
    joblib.dump(model, args.out/'rf_mfcad_face_classifier.joblib')
    print(json.dumps({'validation': report['validation'], 'test': report['test']}, indent=2))
    print('Artifacts saved:', args.out.resolve())

if __name__ == '__main__':
    main()
