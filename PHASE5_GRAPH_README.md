# ForgePath AI — Phase 5: true CAD face-adjacency GraphSAGE

This kit extends **your existing** `PS04_Manufacturing_AI` project. It does not change the original 15,488 MFCAD STEP files, the frozen part-group manifest, your Random Forest, or your XGBoost artifacts.

## What we learned before this phase

| Model | Full-data validation accuracy | Validation macro F1 |
|---|---:|---:|
| Random Forest | 89.30% | 87.02% |
| XGBoost | 88.67% | 86.46% |

Random Forest is the current aggregate leader. XGBoost does better on chamfers and rectangular through steps, but worse on triangular pockets and rectangular blind slots. This motivates investigating face adjacency, not asserting it will improve anything.

## File placement

Extract this ZIP into your existing project root. It adds:

```
research_training/
    export_mfcad_adjacency.py
    train_graphsage_frozen.py
PHASE5_GRAPH_README.md
```

It requires the earlier `research_training/extract_mfcad.py` (patched for `TopoDS.Face`, `TopoDS.Edge`, `CornerMin`, `CornerMax`) plus the fixed-split manifest and full geometry CSV already generated in Phase 3.

## Step 1: Install PyTorch in your *existing* environment

On Windows PowerShell, inside the existing project:

```powershell
.\.venv-ml\Scripts\python.exe -m pip install torch
.\.venv-ml\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

This kit does **not** require torch-geometric or DGL. Its GNN is a native PyTorch implementation.

## Step 2: Export 20 graphs as a correctness smoke test

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\export_mfcad_adjacency.py --max-parts 20
```

Read the output and inspect:

```powershell
Get-Content .\research_outputs\graphs\mfcad_adjacency.audit.json
```

Expected: 20 graphs, 0 skipped, positive adjacency edge count. If any graph fails, share the error report **before** attempting full-scale export. Do not train yet.

## Step 3: Export all 15,488 graphs

Only after Step 2 passes:

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\export_mfcad_adjacency.py --max-parts 0
```

This writes `research_outputs/graphs/mfcad_adjacency/<part_id>.npz` and an audit. The exporter checks the numbered STEP face IDs against the corresponding original `.face_truth` annotation file. It includes **only adjacency data**, not target labels, in its `.npz` output.

A valid full run should report `15488 graphs`, `350295 faces`, and zero skipped; the directed-edge count depends on topology.

## Step 4: Train GraphSAGE on frozen train/validation partitions

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\train_graphsage_frozen.py
```

The model:

- Uses train-only fitted one-hot encoding and numeric standardization.
- Aligns each graph node explicitly using STEP `face_id`, not incidental enumeration order.
- Aggregates first- and second-hop neighbor features.
- Trains on the **10,839** frozen training CAD parts only.
- Uses the **2,325** validation parts to select the best epoch by macro F1.
- Does **not** evaluate the 2,324 held-out test parts.
- Saves the model, preprocessing, per-class report, and validation confusion matrix in `research_outputs/fixed_benchmark_graphsage_v1/`.

Use `--device cpu` to force CPU or default automatic CUDA detection. Stop if memory errors occur; reduce `--batch-parts` to 8 or 16 and use a new `--out` directory.

## Step 5: Compare with your actual baseline

```powershell
$rf = Get-Content .\research_outputs\fixed_benchmark_rf_v1\validation_metrics.json -Raw | ConvertFrom-Json
$xgb = Get-Content .\research_outputs\fixed_benchmark_xgb_v1\validation_metrics.json -Raw | ConvertFrom-Json
$gnn = Get-Content .\research_outputs\fixed_benchmark_graphsage_v1\validation_metrics.json -Raw | ConvertFrom-Json
@(
    [pscustomobject]@{Model='Random Forest'; Accuracy=$rf.accuracy; MacroF1=$rf.macro_f1}
    [pscustomobject]@{Model='XGBoost';Accuracy=$xgb.accuracy;MacroF1=$xgb.macro_f1}
    [pscustomobject]@{Model='GraphSAGE';Accuracy=$gnn.validation_accuracy;MacroF1=$gnn.validation_macro_f1}
) | Format-Table -AutoSize
```

Note: This first GNN run is only a baseline. We are **not** claiming superiority, novelty, or publication acceptance. A fair final study should include repeated seeds, selected hyperparameter protocols, geometry similarity audits, and controlled ablations (same neural network with and without adjacency).

## Known limitations and responsible claims

1. The 4-component filename group is still a **provisional family proxy**; zero fingerprint collisions is not proof of complete geometry independence.
2. Matching face IDs protects correctness of correspondence with annotation ordering but the actual feature-label mapping must be tested with spot-visualized geometry to be medically/manufacturing meaningful.
3. This dataset evaluates **CAD machining-feature recognition**, not full routing-chart generation. The PS 04 routing module needs a separate validated benchmark.
4. Pure PyTorch GraphSAGE here uses a deliberately simple two-layer mean aggregation; it is **not a reproduction of AAGNet**.
5. Do not run any `--final-test` command until architectures, grouping, and ablations are frozen.

## Validation performed for this kit

- Python syntax checks passed for both files.
- Two unit tests passed, checking GraphSAGE output and normalized OpenCascade edge identity on a test cube.
- An **end-to-end training smoke test passed on synthetic graph data only**. It does not establish real MFCAD accuracy or successful export of real MFCAD STEP files.
