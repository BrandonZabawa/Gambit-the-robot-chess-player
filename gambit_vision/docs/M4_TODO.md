# M4 TODO: board assembly + evaluation

Done in the scaffold: `board_assembly.assemble` (argmax, FEN, confidence, low-confidence
flags), basic `check_legality`, metric functions, confusion-matrix plot,
`chesscog_compare.compare`. Tests: `tests/test_board_assembly.py`, `tests/test_evaluate.py`.

## Remaining, in order

1. [ ] **Legality checks**: `src/gambit_vision/board_assembly.py:check_legality` — add
   `board.status()`-based checks (kings adjacent, side not to move in check); handle the
   unknown turn (try both). Decide on promotion-aware piece caps (`MAX_PIECES`).
2. [ ] **Repair**: `board_assembly.py:repair` — swap lowest-confidence squares to their
   second-best class until legal; call it from `assemble` when `legal` is False.
   Un-skip `test_repair_fixes_second_king`.
3. [ ] **Evaluation loop**: `src/gambit_vision/evaluate.py:evaluate_model` — load the 64 crops
   per frame in square order (needs `dataset.load_manifest` + a crop loader from M3),
   fill the per-square rows, add per-class / occupancy-only accuracy and repair rate.
   Also write a per-board CSV (`session_id, frame_id, fen_true, fen_ours`) for step 5.
4. [ ] **Run both models** on `data.test_sessions` (set them in `configs/train.yaml`):
   baseline (occupancy/color only, so report occupancy & color accuracy) and ResNet-18.
   Use a different `eval.run_name` per model.
5. [ ] **chesscog**: `src/gambit_vision/chesscog_compare.py:run_chesscog` — install in a
   separate venv, check it works on our `frames/*.png` (warped) or needs raw frames,
   write `fen_chesscog` CSV, then `compare()` -> `results/<run>/chesscog_vs_ours.csv`.
6. [ ] **Threshold**: sweep `assembly.low_confidence_threshold`; report how many errors are
   caught by the flag vs. how many correct squares are flagged.
7. [ ] Register CLI commands in `cli.py` (`evaluate`, `chesscog-run`, `chesscog-compare`)
   and un-skip remaining tests.

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
