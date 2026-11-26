# Gambit Vision data contracts

The stages talk to each other only through the formats below. If you change one, update
this file and every reader in the same commit. Paths are relative to `gambit_vision/`.

## Labels and squares
Defined in `src/gambit_vision/labels.py`. 13 classes in the fixed order
`empty, P, N, B, R, Q, K, p, n, b, r, q, k`. Squares use python-chess indexing (a1 = 0 ... h8 = 63).

## Square crops (written by M2 capture.py, read by M3 dataset.py)

```
data/crops/<session_id>/
    frames/<frame_id>.png            full warped top-down board image (optional, for M4/M5)
    squares/<frame_id>_<square>.png  one crop per square, e.g. 000123_e4.png
    manifest.csv
```

`session_id`: one capture session (one lighting setup / camera placement), e.g. `s01_daylight`.
Splits are always by `session_id`, never by frame.

`manifest.csv`, one row per square crop:

| column | type | meaning |
|---|---|---|
| session_id | str | capture session |
| game_id | str | which recorded game was replayed |
| move_idx | int | ply index in that game (0 = start position) |
| frame_id | str | unique within the session |
| square | str | `a1` ... `h8` |
| label | str | one of `labels.CLASSES` |
| path | str | crop path relative to the session directory |
| fen | str | full board FEN of this frame (same for all 64 rows of a frame) |

Crop geometry: a square plus a margin above it (pieces are tall and lean in perspective).
Default: crop is the square widened by 50% on each side and extended 100% upward in the
warped image, resized to 100 x 200 (w x h). M2 owns these numbers; record them in
`configs/camera.yaml` under `crop:`.

## Model output (M3 -> M4)
Any square classifier exposes

```python
predict_proba(crops: np.ndarray | list[np.ndarray]) -> np.ndarray  # shape (N, 13), rows sum to 1
```

For a whole board, N = 64 in square-index order, so M4 receives a (64, 13) array.

## Board prediction (M4 -> M5, and to the robot)
`board_assembly.assemble(probs) -> BoardPrediction` with at least:
`fen` (piece placement), `labels` (64 str), `confidence` (64 float), `low_confidence`
(list of square names), `legal` (bool), `issues` (list of str).

## Results
Evaluation writes to `results/<run_name>/` (git-ignored): `metrics.json`,
`confusion_matrix.png`, `per_square.csv`, and for the chesscog comparison `chesscog_vs_ours.csv`.
