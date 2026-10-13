# ForgePath AI · Phase 6: GraphSAGE adjacency ablation

This kit adds two files to your **existing** project without replacing or modifying the trained GraphSAGE or any earlier Random Forest / XGBoost results:

```text
research_training/
  train_graphsage_no_edges.py
  compare_graphsage_ablation.py
```

The original GraphSAGE validation model achieved **99.6963% accuracy** and **99.6937% macro F1** on **53,349 validation faces**. These numbers passed an artifact/manifest integrity check, but this does not prove independent real-world generalization.

## Purpose

A controlled training run compares the original `train_graphsage_frozen.py` (using true CAD face adjacency) to an **identical GraphSAGE architecture with every graph edge removed**. The models see the same tabular node descriptors, labels, train/validation CAD parts, seed, epoch limit, patience, hidden width, loss and optimizer. No train/validation graph contains neighbor information in the ablation.

The no-edge script does not touch the final **TEST** partition and refuses to overwrite a nonempty output folder. It writes to `research_outputs/fixed_benchmark_graphsage_no_edges_v1/` instead of overwriting `fixed_benchmark_graphsage_v1/`.

## Run from VS Code PowerShell (your existing project root)

```powershell
cd D:\PS04_Manufacturing_AI_Starter\PS04_Manufacturing_AI

Test-Path .\research_training\train_graphsage_no_edges.py
Test-Path .\research_training\compare_graphsage_ablation.py

.\.venv-ml\Scripts\python.exe .\research_training\train_graphsage_no_edges.py --device cpu --epochs 30 --patience 6 --batch-parts 16 --seed 42

.\.venv-ml\Scripts\python.exe .\research_training\compare_graphsage_ablation.py
```

Run after preserving the original checkpoint. **Do not run** a test-set evaluation now.

## Research interpretation

- If real-edge GraphSAGE scores higher than no-edge GraphSAGE on validation, this is evidence that adjacency contributes predictive information on the current data split.
- If results are similar, face descriptors or the procedurally generated dataset may explain much of the near-perfect performance.
- This single ablation does not establish causal generalization to real factory CAD, manufacturing route feasibility, or publication novelty.
- Repeat with multiple seeds and evaluate graph-family grouping and unseen CAD sources before a final frozen test.
- The graph export contains edge connectivity, while the separate training CSV provides labeled node features. The model does not pass target labels as inputs in the provided training source.

## Testing performed on this kit

- Code source was compared against the original Phase 5 trainer.
- Ran `--help` for the ablation script.
- Completed a 2-epoch real execution on small synthetic CAD-graph fixtures with zero adjacency edges and verified independent output artifacts.
- **Full-scale ablation on your actual MFCAD dataset has not been run here.**
