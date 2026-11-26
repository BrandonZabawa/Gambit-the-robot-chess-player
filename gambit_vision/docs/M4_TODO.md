# M4 TODO: board assembly + evaluation

Design skeleton: `notebooks/board_assembly.ipynb`, `evaluate.ipynb`, `chesscog_compare.ipynb`
hold the dataclasses, constants, signatures and design notes; every function body is a TODO
and each `## Self-check` is a test plan to turn into asserts. Depends on M3 (`labels`,
`dataset`, a classifier with `predict_proba(crops) -> (N, 13)`).
Run everything from `notebooks/run_evaluate.ipynb` and `notebooks/run_chesscog.ipynb`.

## To do, in order

1. [ ] **Assembly**: `notebooks/board_assembly.ipynb: assemble` — argmax per square, FEN,
   confidence, low-confidence flags (`assembly.low_confidence_threshold`), expected-board
   diff in `issues`. Plus the `one_hot` test helper.
2. [ ] **Legality checks**: `notebooks/board_assembly.ipynb: check_legality` — king counts,
   piece caps (`MAX_PIECES`, `MAX_PER_COLOR`), pawns on back ranks; then `board.status()`-based
   checks (kings adjacent, side not to move in check); handle the unknown turn (try both).
   Decide on promotion-aware piece caps.
3. [ ] **Repair**: `notebooks/board_assembly.ipynb: repair` — swap lowest-confidence squares to their
   second-best class until legal; call it from `assemble` when `legal` is False.
   Write the board_assembly self-check from its plan.
4. [ ] **Metrics**: `notebooks/evaluate.ipynb: per_square_accuracy`, `full_board_accuracy`,
   `confusion_matrix_plot` + their self-checks.
5. [ ] **Evaluation loop**: `notebooks/evaluate.ipynb: evaluate_model` — load the 64 crops
   per frame in square order (`dataset.load_manifest` + `read_crop`), fill the per-square
   rows, add per-class / occupancy-only accuracy and repair rate. Also write a per-board
   CSV (`session_id, frame_id, fen_true, fen_ours`) for step 7.
6. [ ] **Run both models** on `data.test_sessions` (set them in `configs/train.yaml`):
   baseline (occupancy/color only, so report occupancy & color accuracy) and ResNet-18.
   Use a different `eval.run_name` per model.
7. [ ] **chesscog**: `notebooks/chesscog_compare.ipynb: compare`, `squares_wrong`, `_placement`,
   then `run_chesscog` — install in a separate venv, check it works on our `frames/*.png`
   (warped) or needs raw frames, write `fen_chesscog` CSV, then `compare()` ->
   `results/<run>/chesscog_vs_ours.csv`.
8. [ ] **Threshold**: sweep `assembly.low_confidence_threshold`; report how many errors are
   caught by the flag vs. how many correct squares are flagged.
9. [ ] Write the numbered steps in `notebooks/run_evaluate.ipynb` / `run_chesscog.ipynb` as code.

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
