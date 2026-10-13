# Dataset acquisition: PHASE 2, NOT YET DOWNLOADED

The initial release deliberately uses user-entered synthetic/verified feature specifications and deterministic planning rules. **No dataset, trained machine-learning model, benchmark score or industrial time estimate is claimed.**

Before training, identify licensing and split strategies for manufacturing-feature CAD benchmarks (e.g. MFCAD/MFCAD++), and separately locate or commission **expert-reviewed process-route ground truth**. Feature labels do not imply route labels.

Separate benchmark splits by originating CAD part/part family to prevent duplicates and derived geometries from leaking across training and test. Save sources, licenses, manifest hashes and evaluation protocol in `docs/DATASET_CARD.md`.

Do not upload proprietary STEP files or real factory production data to public GitHub.
