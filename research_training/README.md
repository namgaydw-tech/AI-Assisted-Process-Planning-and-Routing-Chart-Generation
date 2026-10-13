# ForgePath PS04: MFCAD Phase 2 Starter

A **real** geometric feature extraction + grouped face classification baseline for the original MFCAD, using verified STEP face-name IDs and no unsafe pickle execution.

## Set up on Windows (Python 3.13)

Copy this `research_training` folder to your project root so you have `PS04_Manufacturing_AI/research_training/`.

Run from **your project root**:

```powershell
py -3.13 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install -r .\research_training\requirements.txt
.\.venv-ml\Scripts\python.exe -c "from OCP.STEPControl import STEPControl_Reader; import sklearn; print('OCP + scikit-learn OK')"
```

## Extract an initial geometric subset

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\extract_mfcad.py --max-parts 200
```

This randomly selects (seed=42) 200 STEP models and stops any sample where numeric STEP face IDs and `.face_truth` labels disagree. Output is `research_outputs/mfcad_face_features.csv` plus an audit JSON.

## Train an honest Random Forest baseline

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\train_baseline.py
```

Saves held-out evaluation, a 16-class confusion matrix, train/validation/test part lists, and a trained model in `research_outputs/rf_baseline/`.

**This is not a routing-chart predictor:** it only classifies the machining-feature label of each CAD face. The existing rule/constraint-based routing system is a separate component.

## Full experiment

Only after 200-part smoke test succeeds, extract the entire dataset:

```powershell
.\.venv-ml\Scripts\python.exe .\research_training\extract_mfcad.py --max-parts 0
.\.venv-ml\Scripts\python.exe .\research_training\train_baseline.py
```

Extraction may take substantial time. Do not run full extraction until the sample output has been checked.

## Scientific safeguards

- 16 original MFCAD classes; MFCAD face_truth files are decoded with a strict primitive pickle opcode allowlist, *never* unpickled via `pickle.loads`.
- STEP face IDs are read from the same named entities as the authors' visualizer. A malformed/mismatched part is skipped and reported, not silently relabeled.
- Uses a random group-disjoint **CAD-part** 70/15/15 split, not face-level random leakage.
- This does **not** guarantee design-family-independent generalization. We need an additional near-duplicate/family grouping audit before making publication-level results claims.
- Exact geometry parser compatibility and face ID matching must be verified on **your real 15,488 downloaded parts**. Local tests in the authoring environment used a small synthetic STEP sample, not your complete MFCAD dataset.
- Geometry features contain area, relative position, dimensions, surface type and adjacency count. The method is a baseline, not a state-of-the-art GNN.
- Dataset contains **machining feature face labels**, not machining operation sequences or verified machine routes.
- Test results must be reported with actual class support counts. Rare classes may be absent from small test subsets; start with smoke tests and use the full dataset for research conclusions.
