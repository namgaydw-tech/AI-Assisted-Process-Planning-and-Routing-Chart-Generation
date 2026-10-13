"""MFCAD RandomForest training respecting the pre-recorded full-dataset split.

Default: train-only fit and VALIDATION metrics; test set remains untouched.
Optional --final-test: one explicitly authorized final evaluation, refused if already evaluated.
Never use final-test results for model selection or threshold tuning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_CLASSES = [
    'rectangular_through_slot', 'triangular_through_slot', 'rectangular_passage',
    'triangular_passage', '6sides_passage', 'rectangular_through_step',
    '2sides_through_step', 'slanted_through_step', 'rectangular_blind_step',
    'triangular_blind_step', 'rectangular_blind_slot', 'rectangular_pocket',
    'triangular_pocket', '6sides_pocket', 'chamfer', 'stock'
]


def scores(actual, predicted, parts):
    return dict(accuracy=float(accuracy_score(actual, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(actual, predicted)),
                macro_f1=float(f1_score(actual, predicted, labels=range(16), average='macro', zero_division=0)),
                weighted_f1=float(f1_score(actual, predicted, labels=range(16), average='weighted', zero_division=0)),
                n_faces=int(len(actual)), n_parts=int(parts.nunique()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, default=Path('research_outputs/mfcad_full_face_features.csv'))
    p.add_argument('--manifest', type=Path, default=Path('research_outputs/splits/mfcad_full_provisional_group_v1.json'))
    p.add_argument('--out', type=Path, default=Path('research_outputs/fixed_benchmark_rf_v1'))
    p.add_argument('--audit-file', type=Path, default=Path('research_outputs/splits/mfcad_full_provisional_group_v1.audit.json'))
    p.add_argument('--final-test', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    splits = {k:set(v) for k, v in manifest['splits'].items()}
    assert set(splits)=={'train','validation','test'}
    assert all(not(splits[a] & splits[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
    if not args.audit_file.exists():
        raise SystemExit(f'Run and review the frozen split audit first: {args.audit_file}')
    audit = json.loads(args.audit_file.read_text(encoding='utf-8'))
    if (not audit.get('source_digest_verified') or
        audit.get('prefix_family_overlap') != 0 or
        audit.get('exact_cross_split_duplicate_groups') != 0 or
        audit.get('geometry_cross_split_candidate_groups') != 0 or
        audit.get('geometry_parts_scanned') != sum(len(v) for v in splits.values()) or
        any(audit.get('missing_classes', {}).get(k, []) for k in splits)):
        raise SystemExit('Audit incomplete or warning present; review and resolve before training')
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    df = pd.read_csv(args.csv)
    if df.empty or df[['part_id','face_id']].duplicated().any():
        raise ValueError('Empty CSV or duplicate CAD face keys')
    if not set(df.part_id).issubset(set.union(*splits.values())):
        raise ValueError('CSV contains parts outside manifest')
    if set(df.part_id) != set.union(*splits.values()):
        raise ValueError('Not all 15,488 manifest parts were successfully extracted; do not evaluate')
    labels = df['label'].astype(int)
    if not labels.between(0, 15).all():
        raise ValueError('Unsupported classes')
    if df.isna().any().any():
        raise ValueError('Missing values in extracted features')
    cols = [c for c in df.columns if c not in ('part_id', 'face_id', 'label')]
    X = df[cols]
    all_parts = df.part_id
    tr = all_parts.isin(splits['train'])
    va = all_parts.isin(splits['validation'])
    te = all_parts.isin(splits['test'])
    if not all((tr | va | te)) or int((tr & va).sum() + (tr & te).sum() + (va & te).sum()):
        raise ValueError('Split alignment failed')
    if args.final_test and not (args.out / 'validation_metrics.json').exists():
        raise SystemExit('Train and lock the development model first before evaluating test')
    if args.final_test:
        # Evaluate only the already-trained, versioned artifact. Do not refit using test.
        path = args.out / 'rf_model.joblib'
        if not path.exists():
            raise FileNotFoundError(path)
        if (args.out / 'FINAL_test_metrics.json').exists():
            raise SystemExit('Final test already evaluated in this output folder; refusing re-evaluation')
        meta = json.loads((args.out / 'validation_metrics.json').read_text(encoding='utf-8'))
        if meta['manifest_sha256'] != manifest_sha:
            raise ValueError('Frozen split manifest changed since model was trained')
        model = joblib.load(path)  # only load our own local model artifacts
        pred = model.predict(X.loc[te])
        metrics = scores(labels.loc[te], pred, all_parts.loc[te])
        metrics['model'] = 'RandomForestClassifier'
        metrics['manifest_sha256'] = manifest_sha
        (args.out / 'FINAL_test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        pd.DataFrame(confusion_matrix(labels.loc[te],pred,labels=range(16)),
                     index=FEATURE_CLASSES,columns=FEATURE_CLASSES).to_csv(args.out/'FINAL_test_confusion.csv')
        print('FINAL TEST evaluated:', json.dumps(metrics, indent=2))
        return
    if (args.out / 'FINAL_test_metrics.json').exists():
        raise SystemExit('Output folder already has final test results. Refusing to overwrite model.')
    numeric = [c for c in cols if c != 'surface_type']
    prep = ColumnTransformer([('surface', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['surface_type']),
                              ('numeric', 'passthrough', numeric)])
    model = Pipeline([('prep', prep), ('rf', RandomForestClassifier(n_estimators=150,
                      min_samples_leaf=2, class_weight='balanced_subsample',
                      random_state=args.seed, n_jobs=-1))])
    print(f'Training on {int(tr.sum())} faces, {all_parts[tr].nunique()} parts; validation on {int(va.sum())} faces',flush=True)
    model.fit(X.loc[tr], labels.loc[tr])
    pred = model.predict(X.loc[va])
    report = scores(labels.loc[va], pred, all_parts.loc[va])
    report.update(manifest_sha256=manifest_sha, training_parts=int(all_parts[tr].nunique()),
                  note='Validation only: untouched test data was not evaluated. Prefix family proxy may not imply real geometry independence.',
                  per_class=classification_report(labels.loc[va],pred,labels=list(range(16)),
                            target_names=FEATURE_CLASSES,output_dict=True,zero_division=0))
    args.out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,args.out/'rf_model.joblib')
    (args.out/'validation_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(confusion_matrix(labels.loc[va],pred,labels=range(16)),
                 index=FEATURE_CLASSES,columns=FEATURE_CLASSES).to_csv(args.out/'validation_confusion.csv')
    print('VALIDATION only:',json.dumps({k:v for k,v in report.items() if k!='per_class'},indent=2))
    print('Model saved:',args.out/'rf_model.joblib')
    print('TEST partition untouched; do not use --final-test until geometry audit and model selection are complete.')

if __name__=='__main__':
    main()
