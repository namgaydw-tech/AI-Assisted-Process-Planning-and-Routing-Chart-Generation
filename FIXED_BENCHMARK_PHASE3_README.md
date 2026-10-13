# ForgePath AI: MFCAD fixed benchmark phase 3

This is an *additional* `research_training` kit for PS04. Do not delete your existing scripts, virtual environment, CAD dataset, or pilot experiments.

**Current observation**: 15,488 STEP models; 3,860 four-component filename groups; provisional split with 10,839 train, 2,325 validation and 2,324 test models. Prefix grouping is *not verified geometric grouping*. Earlier pilot test accuracy 84.35%, macro-F1 79.26%, is not a full-dataset result.

## Install

Extract the archive so these two Python files are at `research_training/audit_frozen_split.py` and `research_training/train_frozen_split.py` under the existing project root.

Existing Python 3.13 virtual environment and dependencies are sufficient. Do not reinstall CAD data.

## Step A: Audit the provisional split BEFORE training

```powershell
cd D:\PS04_Manufacturing_AI_Starter\PS04_Manufacturing_AI
.\.venv-ml\Scripts\python.exe .\research_training\audit_frozen_split.py
```

This audits all 15,488 `STEP`/`.face_truth` files, rechecks inventory, checks 16-class presence across splits, and SHA256 scans STEP bytes for cross-split exact duplicates. It does **not** prove geometric-family independence. Inspect warnings and stop if duplicate candidates or missing classes are found. `research_outputs/splits/mfcad_full_provisional_group_v1.audit.json` is the output.

## Step B: Extract the ENTIRE dataset to a NEW output path

**Wait until Step A has passed.** The extractor is single-process, streams CSV output, and can take a while. It will not resume after interruption; choose a time you can leave VS Code running. **Keep your 1,000-part pilot files untouched.**

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\extract_mfcad.py --max-parts 0 --output .\research_outputs\mfcad_full_face_features.csv
```

Inspect the final summary: want 15,488 valid, 0 skipped. If there are skips, do not train on an incomplete file. Its audit will be at `research_outputs/mfcad_full_face_features.audit.json`.

## Step C: Screen potential geometry duplicates using extracted features

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\audit_frozen_split.py --features-csv .\research_outputs\mfcad_full_face_features.csv
```

This performs approximate, label-free, normalized per-face geometry fingerprints across parts, and checks candidate cross-split matches. The trainer requires this full geometry audit to pass. No heuristic can prove all geometric independence. Examine candidates with CAD visualization or an appropriate geometric distance and regroup/re-freeze BEFORE evaluating final test if necessary. Filename grouping at depth 4 is also a heuristic that requires validation.

## Step D: Train ONLY on frozen train IDs and report validation

After reviewing the audits and verifying split design is defensible:

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\train_frozen_split.py
```

Outputs go into `research_outputs/fixed_benchmark_rf_v1/`: `rf_model.joblib`, `validation_metrics.json`, `validation_confusion.csv`. The training script explicitly refuses an incomplete extracted CSV and will not inspect the reserved test split by default. The model does NOT generate machining routes; it only predicts feature classes for CAD faces.

**DO NOT run `--final-test` yet.** That is a one-time final evaluation option for use only after the evaluation protocol, architecture, hyperparameters and model selection have been finalized. Comparing multiple models using validation is fine; comparing them repeatedly against the held-out test is not.

## Research limitations

- Face-level labels are not route-planning ground truth.
- Sample (200/1000) and full-dataset metrics may not be directly comparable due to different splits.
- SHA256 catches byte-identical CAD files only, and numerical descriptors detect some near-duplicates only. Neither establishes that independent designs have been separated.
- Models trained on face features require the **same** extraction pipeline for future inference, including face-ID and feature-order integrity.
- Preserve the split JSON, source checksums, all audit reports, random seeds, library versions, and per-class validation reports for a conference submission.
