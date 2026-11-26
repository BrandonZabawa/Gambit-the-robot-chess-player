# M3 TODO: square classifiers

The code is done and tested on fake data. What remains needs real capture sessions, a GPU or
your judgment. Implemented:

- [x] Manifest loading, session splits, label columns (`y`, `y_occ`, `y_color3`), `SquareCropDataset`.
- [x] `dataset.py:build_transforms`: ColorJitter (+ small hue), RandomAffine, horizontal flip,
      occasional Gaussian blur. No vertical flip: the crop is not symmetric up/down.
- [x] Shared board geometry in `dataset.py` (M2 `capture.py` should reuse it):
      `CropGeometry` / `load_crop_geometry` (reads `configs/camera.yaml`), `warp_board`
      (4 corners a1, h1, h8, a8 -> top-down image), `crop_squares` (64 crops in a1..h8 order),
      `crops_from_image`, `sort_corner_points`, `corner_orientations`.
- [x] Baseline features: `hsv_histogram`, `edge_density` (Canny, global + 2x2 grid),
      `lbp_histogram` (uniform LBP in plain numpy, no scikit-image).
- [x] `baseline.py:train_baseline`: train + val accuracy and confusion matrix per target, written
      to `models/baseline/baseline.json` next to the pickle. `predict_proba` keeps a fixed column
      count even if a class was missing from training.
- [x] `train.py:train_cnn`: class-weighted CrossEntropy, AdamW with backbone lr =
      `lr * backbone_lr_mult` (default 0.1), head-only warm-up for `freeze_backbone_epochs`,
      cosine schedule, val accuracy + 13x13 confusion matrix, `best.pt` by val accuracy,
      `history.json`, TensorBoard scalars (`train/loss`, `train/acc`, `val/acc`, `val/loss`, `lr/*`).
- [x] Optional pretraining via `cfg.cnn.pretrain_on` (`chessred` | `synthetic`):
      `load_chessred`, `load_chesscog_synthetic` warp + crop the external images once and cache
      them as normal sessions under `<root>/gambit_crops/<split>/`.
- [x] Tests: features, transforms, warp/crop round trip, both external loaders on tiny fixtures,
      `train_baseline` end to end, 1-2 epoch CPU training smoke test, pretraining hook.

Optional config keys (read with `config.cfg_get`, defaults in brackets; add them to
`configs/train.yaml` under `cnn:` if you want to change them): `backbone_lr_mult` [0.1],
`weight_decay` [1e-4], `num_workers` [2], `class_weighted_loss` [true], `pretrain_epochs` [5],
`pretrain_max_images` [all].

## 1. Data (needs M2 sessions)
- [ ] Fill `data.train_sessions / val_sessions / test_sessions` in `configs/train.yaml` once
      M2 capture sessions exist. Keep each session in one split only (`split_by_session` checks).
- [ ] Choose a session naming rule that separates sim from real (e.g. `sim_*` vs `s01_*`) so the
      three experiments are just different session lists.
- [ ] Look at a few augmented crops (e.g. in the notebook) and adjust `cnn.augment` if they look
      unrealistic for your camera.

## 2. External datasets (only if you use pretraining)
- [ ] ChessReD: download, then check `annotations.json` against the field names in the
      `load_chessred` docstring (written from memory: `images[].path`, `annotations.pieces[]`
      with `category_id` / `chessboard_position` / `bbox`, `annotations.corners[]` with
      `corners.top_left` ..., `splits.<split>.image_ids`). Fix the names if they differ.
      The board rotation per photo is inferred from the piece boxes, so corner naming does not
      matter. Only images with corner annotations (ChessReD2K) are used.
- [ ] chesscog synthetic: download the render dataset; check that `<split>/NNNN.json` has `fen`,
      `corners`, `white_turn`. Open a few cached crops in `<root>/gambit_crops/train/` to confirm
      the orientation (a1 crop shows the a1 square).

## 3. Baseline
- [ ] Tune `make_estimator` hyperparameters on the val sessions (`make_estimator("svm", C=...,
      gamma=...)`, `n_estimators` for the forest) and confirm the feature set in `train.yaml`.

## 4. CNN runs (Colab)
- [ ] `pip install -e gambit_vision[train]`, then `gambit-vision train-cnn --run-name <name>`.
- [ ] Decide whether pretraining helps (one run with `pretrain_on: synthetic` or `chessred`).

## 5. Experiments (from the README plan)
Train ResNet-18 three ways, same val/test sessions for all, one `run_name` each:

| run | train sessions |
|---|---|
| `sim_only` | Gazebo sessions only |
| `real_only` | real capture sessions only |
| `mixed` | sim + real |

The test sessions are real and held out from all three runs.

## 6. What to report
- Per-square accuracy on the test sessions, baseline vs CNN:
  - occupancy accuracy (baseline vs CNN collapsed with `label_to_occupancy`)
  - color3 accuracy (baseline vs CNN collapsed with `label_to_color3`)
  - 13-class accuracy (CNN only) + confusion matrix
- The three CNN runs side by side (sim only / real only / mixed): does sim data help real?
- Per-session breakdown (lighting), and which classes get confused (likely B/P, Q/K).
- Hand the CNN checkpoint to M4 via `CNNClassifier.from_checkpoint(...).predict_proba`.
