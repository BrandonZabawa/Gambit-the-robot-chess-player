# M4 TODO: board assembly + evaluation

## Done (code + tests)

- [x] **Legality checks** (`board_assembly.check_legality(board, turn=None)`): one king per
  color, <= 16 pieces, <= 8 pawns, no pawns on rank 1/8, **promotion-aware caps** (pieces
  beyond the starting set <= missing pawns), **kings not adjacent**, and the check rule:
  with unknown turn only "both kings in check" is flagged; with a known `turn` the side not
  to move must not be in check.
- [x] **Repair** (`board_assembly.repair`, `assemble(..., repair=True)`): force exactly one
  king per color (highest king probability), forbid pawns on back ranks, then greedily
  switch the cheapest square (smallest log-prob drop) to its next-best class while that
  reduces the violation score; gives up (None) after `max_steps=8`. `assemble` reports
  `repaired`, `changed_squares` and the changes in `issues`.
- [x] **Evaluation loop** (`evaluate.evaluate_model`): loads the 64 crops per frame in
  square order, assembles with and without repair, writes `metrics.json`,
  `per_square.csv`, `boards.csv` (session_id, frame_id, fen_true, fen_pred,
  fen_pred_repaired, n_wrong, ...), `confusion_matrix.png`, `failures.png`
  (most confident mistakes, `evaluate.failure_gallery`). Metrics: per-square, per-class,
  occupancy, color3, full-board, legal rate, accuracy on low/high-confidence squares,
  fraction of errors caught by the flag, repair rate and post-repair accuracies.
- [x] **Baseline evaluation** (`evaluate.evaluate_baseline(cfg, clf)`): occupancy + color3
  accuracy (per square, per class, full board) and confusion matrices.
- [x] **chesscog wrapper** (`chesscog_compare.run_chesscog`, `run_chesscog_to_csv`): lazy
  import with install instructions; per-image failures become "". `compare()` reads
  `boards.csv` directly (`ours_column="fen_pred"` by default, or `"fen_pred_repaired"`).

## Remaining (needs the student)

1. [ ] **Run both models** on `data.test_sessions` (set them in `configs/train.yaml`) with a
   different `eval.run_name` each: `evaluate_baseline` for the baseline, `evaluate_model`
   (CLI: `gambit-vision evaluate <checkpoint>`) for ResNet-18.
2. [ ] **chesscog**: install in a separate venv (see `CHESSCOG_INSTALL_HELP`). chesscog
   needs **raw, unwarped camera frames** (it detects the board itself), so make sure M2
   capture saves them, named `<frame_id>.png` so they join with `boards.csv`. Run
   `run_chesscog_to_csv("data/crops/<session>/raw/*.png", "results/chesscog.csv", "white")`,
   check `turn` matches the camera orientation, then
   `compare("results/<run>/boards.csv", "results/chesscog.csv", "results/<run>/chesscog_vs_ours.csv")`.
   The `corner_hint` path uses chesscog private internals and is unverified; only try it
   if chesscog's own corner detection fails on our frames.
3. [ ] **Threshold**: sweep `assembly.low_confidence_threshold` using the `confidence` and
   `correct` columns of `per_square.csv`; report errors caught by the flag vs. correct
   squares flagged.
4. [ ] Optional: register `chesscog-run` / `evaluate-baseline` CLI commands in `cli.py`
   (owned elsewhere; the functions are CLI-friendly).

## Results the final report needs

| method | per-square acc | occupancy acc | full-board acc | legal rate | notes |
|---|---|---|---|---|---|
| Baseline (color/edge features) | n/a (coarse targets) | | n/a | n/a | occupancy + color only |
| ResNet-18 (ours) | | | | | |
| ResNet-18 + repair | | | | | |
| chesscog (pretrained, no fine-tuning) | | | | | on my board |

Plus: 13x13 confusion matrix for the CNN (`results/<run>/confusion_matrix.png`), test
sessions used and number of boards/positions, and a few failure examples
(`results/<run>/failures.png`; common confusions such as B/P or Q/K).
