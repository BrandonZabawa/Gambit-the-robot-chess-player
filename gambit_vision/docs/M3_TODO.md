# M3 TODO: square classifiers

Design skeleton (all in `notebooks/`): `labels.ipynb`, `dataset.ipynb`, `baseline.ipynb`,
`train.ipynb` hold the constants, signatures and design notes; every function body is a TODO
and every `## Self-check` is a test plan to turn into asserts. Only `config.ipynb` (YAML
loading) is implemented. Work through the list below in order.

## 1. Data
- [ ] `notebooks/labels.ipynb`: `label_to_occupancy`, `label_to_color3`, `board_to_labels`,
      `labels_to_board` + self-check (everything else builds on these).
- [ ] `notebooks/dataset.ipynb`: `list_sessions`, `load_manifest` (adds `abs_path`, `y`, `y_occ`,
      `y_color3`), `split_by_session`, `read_crop`, `SquareCropDataset`, `class_weights`.
- [ ] Fill `data.train_sessions / val_sessions / test_sessions` in `configs/train.yaml` once
      M2 capture sessions exist. Keep each session in one split only (`split_by_session` checks).
- [ ] Choose a session naming rule that separates sim from real (e.g. `sim_*` vs `s01_*`) so the
      three experiments are just different session lists.
- [ ] `notebooks/dataset.ipynb: build_transforms`: settle the augmentation set (flips, perspective, blur).

## 2. Baseline (color + edge features)
- [ ] `notebooks/baseline.ipynb: hsv_histogram`
- [ ] `notebooks/baseline.ipynb: edge_density` (Canny)
- [ ] `notebooks/baseline.ipynb: lbp_histogram` (skimage or numpy; add the dependency if skimage)
- [ ] `notebooks/baseline.ipynb: extract_features`, `make_estimator`, `BaselineClassifier`
      (fit / predict_proba / predict / save / load).
- [ ] Turn the test plan in the `## Self-check` of `notebooks/baseline.ipynb` into asserts.
- [ ] `notebooks/baseline.ipynb: train_baseline`: fit on train sessions, val evaluation per target, write `results/<run>/baseline.json`.
- [ ] `notebooks/baseline.ipynb: make_estimator`: tune hyperparameters on val sessions.

## 3. CNN (ResNet-18, 13 classes)
- [ ] `notebooks/train.ipynb: build_model`, `set_backbone_trainable`.
- [ ] `notebooks/train.ipynb: train_cnn`: loaders, optimizer / scheduler, checkpoints (best.pt, last.pt, config.yaml).
- [ ] `notebooks/train.ipynb: train_cnn`: training step (forward, loss, backward, step, TensorBoard loss).
- [ ] `notebooks/train.ipynb: train_cnn`: validation step (accuracy + confusion matrix).
- [ ] Decide on class-weighted loss (`dataset.class_weights`) and a lower backbone lr.
- [ ] Optional pretraining: `notebooks/dataset.ipynb: load_chesscog_synthetic`, `notebooks/dataset.ipynb: load_chessred`,
      wired in through `cfg.cnn.pretrain_on`.
- [ ] Turn the test plan in the `## Self-check` of `notebooks/dataset.ipynb` into asserts.
- [ ] `notebooks/train.ipynb: CNNClassifier` (`__init__`, `from_checkpoint`, `predict_proba`, `predict`) + self-check.
- [ ] Training runs on Colab: open `notebooks/run_train_cnn.ipynb` (clones the repo, installs torch).

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
- Hand the CNN checkpoint to M4 via `notebooks/train.ipynb: CNNClassifier.from_checkpoint(...).predict_proba`.
