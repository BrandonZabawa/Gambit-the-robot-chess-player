# M3 TODO: square classifiers

Scaffold is in place: `dataset.py`, `baseline.py`, `train.py`, `config.py`. Done already:
manifest loading, session splits, label -> index columns (`y`, `y_occ`, `y_color3`),
`SquareCropDataset`, `build_model`, `CNNClassifier.predict_proba` (N, 13), baseline
sklearn wrapper (fit / predict / save / load). Work through the list below in order.

## 1. Data
- [ ] Fill `data.train_sessions / val_sessions / test_sessions` in `configs/train.yaml` once
      M2 capture sessions exist. Keep each session in one split only (`split_by_session` checks).
- [ ] Choose a session naming rule that separates sim from real (e.g. `sim_*` vs `s01_*`) so the
      three experiments are just different session lists.
- [ ] `dataset.py:build_transforms`: settle the augmentation set (flips, perspective, blur).

## 2. Baseline (color + edge features)
- [ ] `baseline.py:hsv_histogram`
- [ ] `baseline.py:edge_density` (Canny)
- [ ] `baseline.py:lbp_histogram` (skimage or numpy; add the dependency if skimage)
- [ ] Un-skip the feature tests in `tests/test_baseline.py`.
- [ ] `baseline.py:train_baseline`: val evaluation per target, write `results/<run>/baseline.json`.
- [ ] `baseline.py:make_estimator`: tune hyperparameters on val sessions.

## 3. CNN (ResNet-18, 13 classes)
- [ ] `train.py:train_cnn`: training step (forward, loss, backward, step, TensorBoard loss).
- [ ] `train.py:train_cnn`: validation step (accuracy + confusion matrix).
- [ ] Decide on class-weighted loss (`dataset.class_weights`) and a lower backbone lr.
- [ ] Optional pretraining: `dataset.py:load_chesscog_synthetic`, `dataset.py:load_chessred`,
      wired in through `cfg.cnn.pretrain_on`.
- [ ] Un-skip `test_build_transforms_output_shape` and loader tests in `tests/test_dataset.py`.
- [ ] Training runs on Colab: `pip install -e gambit_vision[train]`, then the CLI command.

## 4. Experiments (from the README plan)
Train ResNet-18 three ways, same val/test sessions for all, one `run_name` each:

| run | train sessions |
|---|---|
| `sim_only` | Gazebo sessions only |
| `real_only` | real capture sessions only |
| `mixed` | sim + real |

The test sessions are real and held out from all three runs.

## 5. What to report
- Per-square accuracy on the test sessions, baseline vs CNN:
  - occupancy accuracy (baseline vs CNN collapsed with `label_to_occupancy`)
  - color3 accuracy (baseline vs CNN collapsed with `label_to_color3`)
  - 13-class accuracy (CNN only) + confusion matrix
- The three CNN runs side by side (sim only / real only / mixed): does sim data help real?
- Per-session breakdown (lighting), and which classes get confused (likely B/P, Q/K).
- Hand the CNN checkpoint to M4 via `CNNClassifier.from_checkpoint(...).predict_proba`.
