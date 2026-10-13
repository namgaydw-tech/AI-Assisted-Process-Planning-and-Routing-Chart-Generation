# Research plan: PS 04

## Initial research question
Can a constraint-verified CAD-feature-to-route pipeline produce more feasible and lower-estimated-changeover process routes than a feature-order baseline when resources change?

## Current baseline (this repository)
- Features entered manually; basic STEP shape *metadata only* if CadQuery installed.
- Explainable manufacturing rules for holes, pockets, slots and planar faces.
- Deterministic simple time heuristics and tool/machine changeover penalties, which are **illustrative and uncalibrated**.
- Exact dynamic programming sequence optimization for <=14 operations; greedy heuristic for larger instances.
- Per-feature precedence constraints; all operations sequential. Not a multi-machine parallel shop scheduler.

## Dataset and training roadmap
1. Review feature-based CAD datasets (MFCAD, MFCAD++) and independent routing/operation sequence data; confirm licenses.
2. Curate machining-feature recognition ground truth, machining restrictions and expert-reviewed operation plans.
3. Establish baseline with known CAD solid geometry and verified feature labels.
4. Compare geometric heuristics to an appropriately selected *supervised ML classifier* if data quality and scale justify one. Consider a geometry-aware neural model only if geometry labels and compute allow.
5. Add constraint checking for tool dimensions, workholding, accessibility, machine envelope, and toolpath feasibility in separate validated stages.
6. Compare rule-based, ML-assisted and hybrid-planner outputs under the same parts, equipment and evaluation rules.
7. Report feature-level macro F1, manufacturing plan feasibility (expert reviewed), precedence violations, processing time, and optimization cost, with seeds, confidence intervals and failure analysis.
8. Optional later: small instruction LLM to explain verified plan decisions. Treat generated text as non-authoritative.

## Scope & publication honesty
Current scores measure an objective *we invented* (heuristic estimated time); they do not establish physical machining improvements. Train/test scores must come from actual datasets. Research novelty must be established with literature review and stronger baselines. Expert consultation strongly recommended.
