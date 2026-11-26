# M4 TODO: board assembly + evaluation

Done in the scaffold: `notebooks/board_assembly.ipynb: assemble` (argmax, FEN, confidence, low-confidence
flags), basic `check_legality`, metric functions, confusion-matrix plot,
`notebooks/chesscog_compare.ipynb: compare`. Checks: the `## Self-check` sections of `notebooks/board_assembly.ipynb`, `evaluate.ipynb`, `chesscog_compare.ipynb`.
Run everything from `notebooks/run_evaluate.ipynb` and `notebooks/run_chesscog.ipynb`.

## Remaining, in order

1. [ ] **Legality checks**: `notebooks/board_assembly.ipynb: check_legality` — add
   `board.status()`-based checks (kings adjacent, side not to move in check); handle the
   unknown turn (try both). Decide on promotion-aware piece caps (`MAX_PIECES`).
2. [ ] **Repair**: `notebooks/board_assembly.ipynb: repair` — swap lowest-confidence squares to their
   second-best class until legal; call it from `assemble` when `legal` is False.
   Turn the repair TODO note in the self-check into an assert.
3. [ ] **Evaluation loop**: `notebooks/evaluate.ipynb: evaluate_model` — load the 64 crops
   per frame in square order (needs `dataset.load_manifest` + a crop loader from M3),
   fill the per-square rows, add per-class / occupancy-only accuracy and repair rate.
   Also write a per-board CSV (`session_id, frame_id, fen_true, fen_ours`) for step 5.
4. [ ] **Run both models** on `data.test_sessions` (set them in `configs/train.yaml`):
   baseline (occupancy/color only, so report occupancy & color accuracy) and ResNet-18.
   Use a different `eval.run_name` per model.
5. [ ] **chesscog**: `notebooks/chesscog_compare.ipynb: run_chesscog` — install in a
   separate venv, check it works on our `frames/*.png` (warped) or needs raw frames,
   write `fen_chesscog` CSV, then `compare()` -> `results/<run>/chesscog_vs_ours.csv`.
6. [ ] **Threshold**: sweep `assembly.low_confidence_threshold`; report how many errors are
   caught by the flag vs. how many correct squares are flagged.
7. [ ] Fill the TODO cells in `notebooks/run_evaluate.ipynb` / `run_chesscog.ipynb` and turn the
   remaining self-check TODO notes into asserts.

## Results the final report needs

| method | per-square acc | occupancy acc | full-board acc | legal rate | notes |
|---|---|---|---|---|---|
| Baseline (color/edge features) | n/a (coarse targets) | | n/a | n/a | occupancy + color only |
| ResNet-18 (ours) | | | | | |
| ResNet-18 + repair | | | | | |
| chesscog (pretrained, no fine-tuning) | | | | | on my board |

Plus: 13x13 confusion matrix for the CNN (`results/<run>/confusion_matrix.png`), test
sessions used and number of boards/positions, and a few failure examples
(low-confidence squares, common confusions such as B/P or Q/K).
