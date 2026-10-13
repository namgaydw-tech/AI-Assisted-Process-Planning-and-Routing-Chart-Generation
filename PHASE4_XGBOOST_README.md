# ForgePath AI — Phase 4 XGBoost Benchmark

This file trains an XGBoost classifier on the same full MFCAD **frozen training and validation split** as the Phase 3 Random Forest. It uses the same 14 input features and OneHotEncoder strategy as `train_frozen_split.py`, excluding `part_id`, `face_id` and `label`.

**The reserved test partition is not evaluated.** The script refuses to overwrite a previous XGBoost experiment and checks the Phase 3 manifest and audit before training.

## Install

From the project root in VS Code PowerShell:

```powershell
.\.venv-ml\Scripts\python.exe -m pip install xgboost
```

## Run

Place `research_training/train_xgboost_frozen.py` into your existing project and run:

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\train_xgboost_frozen.py
```

The output folder is `research_outputs/fixed_benchmark_xgb_v1/`, separate from your Random Forest model.

## Compare

```powershell
$rf = Get-Content .\research_outputs\fixed_benchmark_rf_v1\validation_metrics.json -Raw | ConvertFrom-Json
$xgb = Get-Content .\research_outputs\fixed_benchmark_xgb_v1\validation_metrics.json -Raw | ConvertFrom-Json
[pscustomobject]@{Model='Random Forest';Accuracy=$rf.accuracy;MacroF1=$rf.macro_f1;BalancedAccuracy=$rf.balanced_accuracy}
[pscustomobject]@{Model='XGBoost';Accuracy=$xgb.accuracy;MacroF1=$xgb.macro_f1;BalancedAccuracy=$xgb.balanced_accuracy}
```

## Research limitations

- Prioritize validation **macro F1**, class-specific performance and ablation results.
- These are not independent-parts-with-no-similarity guarantees: the MFCAD filename-prefix grouping is a provisional proxy.
- Tabular face descriptors exclude full neighboring-face graph relationships.
- The MFCAD dataset contains recognition labels, **not** verified manufacturing process routes.
- Do not evaluate the reserved test set until all model choices are finalized.
