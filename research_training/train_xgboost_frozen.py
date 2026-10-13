"""ForgePath PS04: XGBoost validation benchmark on frozen MFCAD part groups.

Uses EXACTLY the same input features and preprocessor as the Phase 3 RF script.
Reads the pre-existing dataset audit; never predicts on the held-out test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

FEATURE_CLASSES = [
    "rectangular_through_slot", "triangular_through_slot", "rectangular_passage",
    "triangular_passage", "6sides_passage", "rectangular_through_step",
    "2sides_through_step", "slanted_through_step", "rectangular_blind_step",
    "triangular_blind_step", "rectangular_blind_slot", "rectangular_pocket",
    "triangular_pocket", "6sides_pocket", "chamfer", "stock",
]

def main():
    p = argparse.ArgumentParser(description="XGBoost on the MFCAD frozen validation split")
    p.add_argument("--csv", type=Path, default=Path("research_outputs/mfcad_full_face_features.csv"))
    p.add_argument("--manifest", type=Path, default=Path("research_outputs/splits/mfcad_full_provisional_group_v1.json"))
    p.add_argument("--audit-file", type=Path, default=Path("research_outputs/splits/mfcad_full_provisional_group_v1.audit.json"))
    p.add_argument("--out", type=Path, default=Path("research_outputs/fixed_benchmark_xgb_v1"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trees", type=int, default=200, help="Fixed number of boosting rounds. Default: 200.")
    args = p.parse_args()

    # Never silently overwrite an existing versioned experiment or use final test labels.
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing experiment folder: {args.out}")
    if not (args.manifest.is_file() and args.audit_file.is_file() and args.csv.is_file()):
        raise SystemExit("Missing CSV, frozen split, or completed audit. Run Phase 3 first.")
    if not 1 <= args.trees <= 1000:
        raise SystemExit("--trees must be between 1 and 1000")

    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    splits = {k: set(v) for k, v in manifest["splits"].items()}
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Unexpected manifest partitions")
    for a, b in (("train","validation"),("train","test"),("validation","test")):
        if splits[a] & splits[b]:
            raise ValueError(f"CAD part overlap: {a}, {b}")

    audit = json.loads(args.audit_file.read_text(encoding="utf-8"))
    if (
        not audit.get("source_digest_verified")
        or audit.get("prefix_family_overlap") != 0
        or audit.get("exact_cross_split_duplicate_groups") != 0
        or audit.get("geometry_cross_split_candidate_groups") != 0
        or audit.get("geometry_parts_scanned") != sum(map(len, splits.values()))
        or any(audit.get("missing_classes", {}).get(k, []) for k in splits)
    ):
        raise SystemExit("Frozen split audit is incomplete or has warnings. Review it first.")

    df = pd.read_csv(args.csv)
    required = {"part_id", "face_id", "label", "surface_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")
    if df.empty or df[["part_id", "face_id"]].duplicated().any():
        raise ValueError("Empty feature CSV or duplicated part/face identifiers")
    all_ids = set.union(*splits.values())
    if set(df.part_id) != all_ids:
        raise ValueError("Extracted CAD parts and split manifest do not match")
    if df.isna().any().any():
        raise ValueError("Missing extracted CAD values")
    y = df["label"].astype(int)
    if not y.between(0, 15).all():
        raise ValueError("Labels must be integer IDs in 0..15")

    # Identical feature selection and one-hot preprocessing to train_frozen_split.py.
    cols = [c for c in df.columns if c not in ("part_id", "face_id", "label")]
    X = df[cols]
    numeric = [c for c in cols if c != "surface_type"]
    train_mask = df.part_id.isin(splits["train"])
    val_mask = df.part_id.isin(splits["validation"])
    test_mask = df.part_id.isin(splits["test"])
    if not bool((train_mask | val_mask | test_mask).all()):
        raise ValueError("Unassigned CAD parts")
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise ValueError("Split intersection detected")
    if set(y.loc[train_mask].unique()) != set(range(16)):
        raise ValueError("Training partition must contain all 16 classes")

    prep = ColumnTransformer([
        ("surface", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["surface_type"]),
        ("numeric", "passthrough", numeric),
    ])
    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=16,
        n_estimators=args.trees,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
    )
    pipeline = Pipeline([("prep", prep), ("xgb", clf)])
    print(
        f"TRAIN: {int(train_mask.sum()):,} faces / {df.loc[train_mask, 'part_id'].nunique():,} parts. "
        f"VALIDATION: {int(val_mask.sum()):,} faces / {df.loc[val_mask, 'part_id'].nunique():,} parts.",
        flush=True,
    )
    print("TEST: reserved, not used in fit, tuning, or evaluation.", flush=True)
    pipeline.fit(X.loc[train_mask], y.loc[train_mask])
    pred = pipeline.predict(X.loc[val_mask])
    actual = y.loc[val_mask]

    metrics = {
        "algorithm": "XGBClassifier",
        "accuracy": float(accuracy_score(actual, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, pred)),
        "macro_f1": float(f1_score(actual, pred, labels=list(range(16)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, pred, labels=list(range(16)), average="weighted", zero_division=0)),
        "n_faces": int(len(actual)),
        "n_parts": int(df.loc[val_mask, "part_id"].nunique()),
        "training_parts": int(df.loc[train_mask, "part_id"].nunique()),
        "training_faces": int(train_mask.sum()),
        "manifest_sha256": manifest_sha,
        "features": cols,
        "hyperparameters": {
            "trees": args.trees, "max_depth": 5, "learning_rate": 0.07,
            "subsample": 0.85, "colsample_bytree": 0.9, "seed": args.seed,
        },
        "note": "Validation ONLY; untouched final test never evaluated. "
                "Grouping relies on a provisional filename-family proxy.",
    }
    detail = classification_report(
        actual, pred, labels=list(range(16)), target_names=FEATURE_CLASSES,
        output_dict=True, zero_division=0
    )
    cm = pd.DataFrame(
        confusion_matrix(actual, pred, labels=list(range(16))),
        index=FEATURE_CLASSES, columns=FEATURE_CLASSES,
    )
    mistakes = []
    for i, actual_name in enumerate(FEATURE_CLASSES):
        for j, predicted_name in enumerate(FEATURE_CLASSES):
            if i != j and int(cm.iloc[i, j]):
                mistakes.append({
                    "actual": actual_name, "predicted": predicted_name,
                    "count": int(cm.iloc[i, j])
                })
    mistakes.sort(key=lambda row: row["count"], reverse=True)
    metrics["top_15_misclassifications"] = mistakes[:15]

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out / "xgb_model.joblib")
    (args.out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out / "validation_per_class.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    cm.to_csv(args.out / "validation_confusion.csv", index_label="actual_class")
    print("\n=== XGBOOST VALIDATION RESULTS ===")
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "n_faces", "n_parts"):
        print(f"{key}: {metrics[key]}")
    print(f"Saved model + reports: {args.out}")
    print("TEST PARTITION NOT EVALUATED.")

if __name__ == "__main__":
    main()
