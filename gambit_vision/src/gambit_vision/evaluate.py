"""M4: evaluation metrics and the results/<run_name>/ report.

Outputs (docs/contracts.md): metrics.json, per_square.csv, confusion_matrix.png.
Heavy imports (sklearn, matplotlib, pandas, dataset) are done inside functions so this
module imports without them (and without torch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from gambit_vision.labels import CLASSES


def per_square_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Fraction of square crops whose predicted label equals the true label."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def full_board_accuracy(true_fens: Sequence[str], pred_fens: Sequence[str]) -> float:
    """Fraction of boards where the predicted placement matches exactly (all 64 squares).

    Compares only the piece-placement field, so full FENs and board_fens both work.
    """
    if len(true_fens) != len(pred_fens):
        raise ValueError("true_fens and pred_fens must have the same length")
    if not true_fens:
        return float("nan")
    hits = sum(t.split()[0] == p.split()[0] for t, p in zip(true_fens, pred_fens))
    return hits / len(true_fens)


def confusion_matrix_plot(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    out_path: str | Path,
    normalize: bool = True,
    title: str = "Square classifier confusion matrix",
) -> np.ndarray:
    """Save a 13x13 confusion matrix (rows = true, cols = predicted) and return it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES),
                          normalize="true" if normalize else None)
    cm = np.nan_to_num(cm)  # classes absent from y_true give NaN rows
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=list(CLASSES)).plot(
        ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def evaluate_model(cfg: dict, classifier) -> dict:
    """Run `classifier` on the held-out test sessions and write results/<run_name>/.

    cfg: parsed configs/train.yaml (uses data.crops_root, data.test_sessions,
        assembly.low_confidence_threshold, eval.run_name, eval.results_dir).
    classifier: anything with predict_proba(crops) -> (N, 13) (baseline or CNN).
    Returns the metrics dict that is also written to metrics.json.
    """
    import pandas as pd

    from gambit_vision.board_assembly import assemble
    from gambit_vision.dataset import load_manifest

    out_dir = Path(cfg["eval"]["results_dir"]) / cfg["eval"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = cfg.get("assembly", {}).get("low_confidence_threshold", 0.6)

    df = load_manifest(cfg["data"]["crops_root"], cfg["data"]["test_sessions"])

    rows, true_fens, pred_fens, n_legal = [], [], [], 0
    for (session, frame_id), frame in df.groupby(["session_id", "frame_id"]):
        # TODO M4: sort the 64 rows into square-index order (a1..h8) and load the crop
        # images (e.g. a dataset.py helper; path is relative to the session directory).
        crops = None  # TODO M4: list of 64 images in square order
        probs = classifier.predict_proba(crops)  # (64, 13)
        pred = assemble(probs, threshold=threshold)
        n_legal += pred.legal
        true_fens.append(frame["fen"].iloc[0])
        pred_fens.append(pred.fen)
        # TODO M4: one row per square with session_id, frame_id, square, label (true),
        # pred, confidence, correct, low_confidence.
        rows.extend([])

    per_sq = pd.DataFrame(rows)
    per_sq.to_csv(out_dir / "per_square.csv", index=False)

    metrics = {
        "run_name": cfg["eval"]["run_name"],
        "test_sessions": list(cfg["data"]["test_sessions"]),
        "n_boards": len(true_fens),
        "per_square_accuracy": per_square_accuracy(per_sq.get("label", []), per_sq.get("pred", [])),
        "full_board_accuracy": full_board_accuracy(true_fens, pred_fens),
        "legal_rate": n_legal / len(true_fens) if true_fens else float("nan"),
        # TODO M4: per-class accuracy, accuracy on low- vs high-confidence squares,
        # occupancy-only accuracy (to compare with the baseline), repair success rate.
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if len(per_sq):
        confusion_matrix_plot(per_sq["label"], per_sq["pred"], out_dir / "confusion_matrix.png")
    return metrics
