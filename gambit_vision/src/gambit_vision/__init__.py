"""Gambit Vision: overhead camera image -> legal board state (FEN).

Pipeline (one module per stage):
    calibration.py     M1  camera intrinsics (ChArUco)
    homography.py      M1  ArUco corners -> top-down warp + 8x8 grid
    capture.py         M2  capture + automatic labeling of square crops
    dataset.py         M3  loading crops, session-based splits
    baseline.py        M3  color/edge feature classifier (occupancy, piece color)
    train.py           M3  ResNet-18 transfer learning, 13 classes
    board_assembly.py  M4  64 predictions -> FEN, legality check, low-confidence flags
    evaluate.py        M4  metrics, confusion matrix, chesscog comparison
    move_detection.py  M5  stable-frame detection + move inference (optional)

Shared definitions every stage must use live in labels.py.
See docs/contracts.md for the on-disk data formats.
"""

__version__ = "0.1.0"
