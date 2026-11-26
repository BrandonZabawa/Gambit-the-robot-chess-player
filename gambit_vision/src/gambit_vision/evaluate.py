"""M4: evaluation metrics and the results/<run_name>/ report.

Outputs (docs/contracts.md) in results/<run_name>/:
    metrics.json          headline numbers (see evaluate_model)
    per_square.csv        one row per test crop (true/pred label, confidence, ...)
    boards.csv            one row per test frame (fen_true, fen_pred, fen_pred_repaired, ...)
                          -> input for chesscog_compare.compare()
    confusion_matrix.png  13x13, row-normalized
    failures.png          most confident wrong crops (failure_gallery)
Heavy imports (sklearn, matplotlib, pandas, dataset) are done inside functions so this
module imports without them (and without torch).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from gambit_vision.labels import CLASSES, COLOR3, label_to_color3, label_to_occupancy


# --------------------------------------------------------------------------- metrics
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


def per_class_accuracy(y_true: Sequence[str], y_pred: Sequence[str],
                       classes: Sequence[str] = CLASSES) -> dict[str, float | None]:
    """Recall per true class: of the crops that really are class c, the fraction predicted c.

    Classes absent from y_true map to None (JSON null) rather than NaN.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out: dict[str, float | None] = {}
    for c in classes:
        mask = y_true == c
        out[c] = float((y_pred[mask] == c).mean()) if mask.any() else None
    return out


def occupancy_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Accuracy of empty-vs-occupied derived from 13-class labels (comparable with the baseline)."""
    return per_square_accuracy([label_to_occupancy(t) for t in y_true],
                               [label_to_occupancy(p) for p in y_pred])


def color3_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Accuracy of empty/white/black derived from 13-class labels (comparable with the baseline)."""
    return per_square_accuracy([label_to_color3(t) for t in y_true],
                               [label_to_color3(p) for p in y_pred])


def confusion_matrix_plot(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    out_path: str | Path,
    normalize: bool = True,
    title: str = "Square classifier confusion matrix",
    class_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Save a confusion matrix (rows = true, cols = predicted) and return it.

    class_names defaults to the 13 labels.CLASSES; pass e.g. labels.COLOR3 for the baseline.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    names = list(class_names) if class_names is not None else list(CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=names, normalize="true" if normalize else None)
    cm = np.nan_to_num(cm)  # classes absent from y_true give NaN rows
    size = max(4, 0.6 * len(names) + 1)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, cmap="Blues", values_format=".2f" if normalize else "d", colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def failure_gallery(per_square, out_path: str | Path, n: int = 24, cols: int = 8,
                    pred_col: str = "pred") -> int:
    """Save a grid of the most confidently wrong crops (good slides for "failure cases").

    per_square: DataFrame with columns path, square, label, `pred_col`, confidence
        (as written by evaluate_model). Returns the number of crops drawn (0 = no file).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gambit_vision.dataset import read_crop

    wrong = per_square[per_square["label"] != per_square[pred_col]]
    wrong = wrong.sort_values("confidence", ascending=False).head(n)
    if len(wrong) == 0:
        return 0
    cols = min(cols, len(wrong))
    rows = math.ceil(len(wrong) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
        ax.imshow(read_crop(r["path"]))
        ax.set_title(f"{r['square']} {r['label']}->{r[pred_col]}\np={r['confidence']:.2f}", fontsize=8)
    fig.suptitle("Most confident mistakes (true->pred)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return len(wrong)


# --------------------------------------------------------------------------- data access
def _prepare(cfg):
    """cfg -> (cfg dict, output dir, test manifest DataFrame)."""
    from gambit_vision.config import to_dict
    from gambit_vision.dataset import load_manifest

    if not isinstance(cfg, dict):  # accept config.load_config()'s namespace too
        cfg = to_dict(cfg)
    sessions = list(cfg["data"].get("test_sessions") or [])
    if not sessions:
        raise ValueError("cfg data.test_sessions is empty; set the held-out sessions in configs/train.yaml")
    out_dir = Path(cfg["eval"]["results_dir"]) / cfg["eval"]["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return cfg, out_dir, load_manifest(cfg["data"]["crops_root"], sessions)


def iter_frames(df) -> Iterator[tuple[str, str, object, list[np.ndarray]]]:
    """Yield (session_id, frame_id, frame_rows, crops) per frame.

    frame_rows: the 64 manifest rows sorted into square-index order (a1, b1, ..., h8);
    crops: the matching RGB images via dataset.read_crop. Raises if a frame is incomplete.
    """
    import chess

    from gambit_vision.dataset import read_crop

    for (session, frame_id), frame in df.groupby(["session_id", "frame_id"], sort=True):
        frame = frame.assign(sq_idx=frame["square"].map(chess.parse_square)).sort_values("sq_idx")
        if list(frame["sq_idx"]) != list(range(64)):
            raise ValueError(f"{session}/{frame_id}: expected 64 distinct squares, got {len(frame)} rows")
        yield session, frame_id, frame, [read_crop(p) for p in frame["abs_path"]]


def _nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def _write_metrics(metrics: dict, out_dir: Path) -> None:
    clean = {k: _nan_to_none(v) for k, v in metrics.items()}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(clean, f, indent=2)


# --------------------------------------------------------------------------- 13-class model
def evaluate_model(cfg, classifier) -> dict:
    """Run `classifier` on the held-out test sessions and write results/<run_name>/.

    cfg: parsed configs/train.yaml (uses data.crops_root, data.test_sessions,
        assembly.low_confidence_threshold, eval.run_name, eval.results_dir).
    classifier: anything with predict_proba(crops) -> (N, 13) (the CNN, or a fake in tests).
    Every board is assembled twice: plain argmax, and with repair=True.
    Returns the metrics dict that is also written to metrics.json.
    """
    import pandas as pd

    from gambit_vision.board_assembly import assemble
    from gambit_vision.chesscog_compare import squares_wrong

    cfg, out_dir, df = _prepare(cfg)
    threshold = cfg.get("assembly", {}).get("low_confidence_threshold", 0.6)

    rows, boards = [], []
    for session, frame_id, frame, crops in iter_frames(df):
        probs = np.asarray(classifier.predict_proba(crops))  # (64, 13)
        pred = assemble(probs, threshold=threshold)
        pred_r = assemble(probs, threshold=threshold, repair=True)
        fen_true = frame["fen"].iloc[0].split()[0]
        low = set(pred.low_confidence)
        for i, r in enumerate(frame.itertuples(index=False)):
            rows.append({
                "session_id": session, "frame_id": frame_id, "square": r.square,
                "label": r.label, "pred": pred.labels[i], "confidence": pred.confidence[i],
                "correct": r.label == pred.labels[i], "low_confidence": r.square in low,
                "pred_repaired": pred_r.labels[i], "correct_repaired": r.label == pred_r.labels[i],
                "path": r.abs_path,
            })
        boards.append({
            "session_id": session, "frame_id": frame_id, "fen_true": fen_true,
            "fen_pred": pred.fen, "fen_pred_repaired": pred_r.fen,
            "n_wrong": squares_wrong(fen_true, pred.fen),
            "n_wrong_repaired": squares_wrong(fen_true, pred_r.fen),
            "legal": pred.legal, "legal_repaired": pred_r.legal, "repaired": pred_r.repaired,
            "issues": "; ".join(pred_r.issues),
        })

    per_sq = pd.DataFrame(rows)
    per_sq.to_csv(out_dir / "per_square.csv", index=False)
    bd = pd.DataFrame(boards)
    bd.to_csv(out_dir / "boards.csv", index=False)

    y, p = per_sq["label"], per_sq["pred"]
    flagged = per_sq["low_confidence"]
    illegal = ~bd["legal"]
    n_wrong_sq = int((~per_sq["correct"]).sum())
    metrics = {
        "run_name": cfg["eval"]["run_name"],
        "test_sessions": list(cfg["data"]["test_sessions"]),
        "n_boards": len(bd),
        "n_squares": len(per_sq),
        "per_square_accuracy": per_square_accuracy(y, p),
        "occupancy_accuracy": occupancy_accuracy(y, p),
        "color3_accuracy": color3_accuracy(y, p),
        "per_class_accuracy": per_class_accuracy(y, p),
        "full_board_accuracy": full_board_accuracy(bd["fen_true"].tolist(), bd["fen_pred"].tolist()),
        "mean_squares_wrong": float(bd["n_wrong"].mean()),
        "legal_rate": float(bd["legal"].mean()),
        # low-confidence flag (threshold sweep: recompute from per_square.csv `confidence`)
        "low_confidence_threshold": threshold,
        "accuracy_high_confidence": per_square_accuracy(y[~flagged], p[~flagged]),
        "accuracy_low_confidence": per_square_accuracy(y[flagged], p[flagged]),
        "flagged_fraction": float(flagged.mean()),
        "errors_caught_by_flag": float(flagged[~per_sq["correct"]].mean()) if n_wrong_sq else float("nan"),
        # repair
        "n_illegal_boards": int(illegal.sum()),
        "repair_rate": float(bd.loc[illegal, "repaired"].mean()) if illegal.any() else float("nan"),
        "repair_made_board_correct": (float((bd.loc[illegal, "n_wrong_repaired"] == 0).mean())
                                      if illegal.any() else float("nan")),
        "per_square_accuracy_repaired": per_square_accuracy(y, per_sq["pred_repaired"]),
        "full_board_accuracy_repaired": full_board_accuracy(bd["fen_true"].tolist(),
                                                            bd["fen_pred_repaired"].tolist()),
        "legal_rate_repaired": float(bd["legal_repaired"].mean()),
    }
    _write_metrics(metrics, out_dir)
    confusion_matrix_plot(y, p, out_dir / "confusion_matrix.png")
    failure_gallery(per_sq, out_dir / "failures.png")
    return metrics


# --------------------------------------------------------------------------- baseline
def evaluate_baseline(cfg, baseline_classifier) -> dict:
    """Evaluate the M3 baseline (occupancy + color3 only) on the test sessions.

    baseline_classifier: a baseline.BaselineClassifier (anything with
        predict(crops, target=...) -> int labels and a `targets` tuple).
    Writes per_square.csv, metrics.json and confusion_matrix_<target>.png to
    results/<run_name>/ (use a different run_name than the CNN run).
    Reports per-square accuracy per target and the fraction of boards where all 64
    squares are right at that level ("full-board" occupancy / color accuracy).
    """
    import pandas as pd

    from gambit_vision.labels import OCCUPANCY

    names = {"occupancy": OCCUPANCY, "color3": COLOR3}
    truth_fn = {"occupancy": label_to_occupancy, "color3": label_to_color3}
    targets = [t for t in getattr(baseline_classifier, "targets", ("occupancy", "color3")) if t in names]

    cfg, out_dir, df = _prepare(cfg)
    rows, board_ok = [], {t: [] for t in targets}
    for session, frame_id, frame, crops in iter_frames(df):
        preds = {t: np.asarray(baseline_classifier.predict(crops, target=t)).astype(int) for t in targets}
        row_base = [{"session_id": session, "frame_id": frame_id, "square": r.square, "label": r.label}
                    for r in frame.itertuples(index=False)]
        for t in targets:
            true = np.array([truth_fn[t](lab) for lab in frame["label"]])
            board_ok[t].append(bool((true == preds[t]).all()))
            for i, rb in enumerate(row_base):
                rb[f"{t}_true"] = names[t][true[i]]
                rb[f"{t}_pred"] = names[t][preds[t][i]]
        rows.extend(row_base)

    per_sq = pd.DataFrame(rows)
    per_sq.to_csv(out_dir / "per_square.csv", index=False)
    metrics = {
        "run_name": cfg["eval"]["run_name"],
        "model": "baseline",
        "test_sessions": list(cfg["data"]["test_sessions"]),
        "n_boards": len(next(iter(board_ok.values()), [])),
        "n_squares": len(per_sq),
    }
    for t in targets:
        metrics[f"{t}_accuracy"] = per_square_accuracy(per_sq[f"{t}_true"], per_sq[f"{t}_pred"])
        metrics[f"{t}_full_board_accuracy"] = float(np.mean(board_ok[t])) if board_ok[t] else float("nan")
        metrics[f"{t}_per_class_accuracy"] = per_class_accuracy(
            per_sq[f"{t}_true"], per_sq[f"{t}_pred"], classes=names[t])
        confusion_matrix_plot(per_sq[f"{t}_true"], per_sq[f"{t}_pred"],
                              out_dir / f"confusion_matrix_{t}.png",
                              title=f"Baseline {t}", class_names=names[t])
    _write_metrics(metrics, out_dir)
    return metrics
