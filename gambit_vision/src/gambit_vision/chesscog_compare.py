"""M4: compare against the pretrained chesscog system on my own board.

chesscog (github.com/georg-wolflein/chesscog, Wolflein & Arandjelovic 2021) is a
comparison baseline only; nothing in the Gambit pipeline depends on it.

Workflow:
    1. run_chesscog() on data/crops/<session>/frames/*.png of the test sessions
       -> save a CSV with columns frame_id, session_id, fen_chesscog
    2. our evaluate run writes per-board predictions (frame_id, session_id, fen_true, fen_ours)
    3. compare() joins both into results/<run_name>/chesscog_vs_ours.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chess
import numpy as np


def run_chesscog(
    image_paths: Sequence[str | Path],
    corner_hint: np.ndarray | None = None,
    turn: chess.Color = chess.WHITE,
) -> list[str]:
    """TODO M4: run chesscog on full-board images and return one board_fen per image.

    Install (separate venv recommended; chesscog pins its own torch/torchvision versions):
        pip install chesscog
        python -m chesscog.occupancy_classifier.download_model
        python -m chesscog.piece_classifier.download_model

    Invocation (verify against the repo, chesscog/recognition/recognition.py):
        from chesscog.recognition.recognition import ChessRecognizer
        recognizer = ChessRecognizer()            # loads models from models://
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)   # expects RGB
        board, corners = recognizer.predict(img, turn)            # chess.Board, (4, 2)
        fen = board.board_fen()
    CLI equivalent: `python -m chesscog.recognition.recognition <image> --white`.

    Open questions (TODO):
    - chesscog finds the board corners itself from a perspective photo. Our frames/ images
      are already warped top-down; check whether its corner detection works on them, or
      feed it raw camera frames instead (then save raw frames in M2 capture).
    - corner_hint: chesscog has no public corner-input API as far as I know; if detection
      fails, consider bypassing its corner step (its internals warp with the found
      corners) using our ArUco corners. Unverified.
    - `turn` sets the board orientation chesscog assumes (which side is at the bottom of
      the image); make sure it matches our camera placement.
    - Record failures (exception / no board found) as an empty string, not a crash.
    """
    raise NotImplementedError("TODO M4: run_chesscog()")


def compare(ours_csv: str | Path, chesscog_csv: str | Path, out_csv: str | Path):
    """Join our per-board predictions with chesscog's and write chesscog_vs_ours.csv.

    ours_csv columns: session_id, frame_id, fen_true, fen_ours
    chesscog_csv columns: session_id, frame_id, fen_chesscog
    Output adds: correct_ours, correct_chesscog, squares_wrong_ours, squares_wrong_chesscog.
    Returns the merged DataFrame.
    """
    import pandas as pd

    ours = pd.read_csv(ours_csv, dtype=str, keep_default_na=False)
    theirs = pd.read_csv(chesscog_csv, dtype=str, keep_default_na=False)
    df = ours.merge(theirs, on=["session_id", "frame_id"], how="left").fillna("")

    for who in ("ours", "chesscog"):
        df[f"correct_{who}"] = [
            _placement(t) == _placement(p) for t, p in zip(df["fen_true"], df[f"fen_{who}"])]
        df[f"squares_wrong_{who}"] = [
            squares_wrong(t, p) for t, p in zip(df["fen_true"], df[f"fen_{who}"])]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def squares_wrong(true_fen: str, pred_fen: str) -> int:
    """Number of the 64 squares that differ; 64 if the prediction is missing or invalid."""
    try:
        t = chess.Board(None)
        t.set_board_fen(_placement(true_fen))
        p = chess.Board(None)
        p.set_board_fen(_placement(pred_fen))
    except ValueError:
        return 64
    return sum(t.piece_at(sq) != p.piece_at(sq) for sq in chess.SQUARES)


def _placement(fen: str) -> str:
    return fen.split()[0] if fen.strip() else ""
