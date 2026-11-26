"""M4: compare against the pretrained chesscog system on my own board.

chesscog (github.com/georg-wolflein/chesscog, Wolflein & Arandjelovic 2021) is a
comparison baseline only; nothing in the Gambit pipeline depends on it.

Workflow:
    1. run_chesscog_to_csv() on the RAW (unwarped) camera frames of the test sessions
       -> CSV with columns session_id, frame_id, fen_chesscog
    2. evaluate.evaluate_model writes results/<run_name>/boards.csv
       (session_id, frame_id, fen_true, fen_pred, fen_pred_repaired, ...)
    3. compare() joins both into results/<run_name>/chesscog_vs_ours.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chess
import numpy as np


CHESSCOG_INSTALL_HELP = """chesscog is not installed (it is only needed for the comparison baseline).
Install it, ideally in a separate virtualenv because it pins its own torch/torchvision:
    pip install chesscog
    python -m chesscog.occupancy_classifier.download_model
    python -m chesscog.piece_classifier.download_model
Then re-run. See https://github.com/georg-wolflein/chesscog for details."""


def _load_recognizer():
    """Import chesscog lazily and build its ChessRecognizer (loads models from models://)."""
    try:
        from chesscog.recognition.recognition import ChessRecognizer
    except ImportError as e:
        raise ImportError(CHESSCOG_INSTALL_HELP) from e
    return ChessRecognizer()


def _predict_with_corners(recognizer, img: np.ndarray, corners: np.ndarray, turn: chess.Color) -> chess.Board:
    """Run chesscog with OUR board corners instead of its own corner detector.

    chesscog has no public API for this. This mirrors ChessRecognizer.predict() but skips
    find_corners(); it relies on private internals (`_corner_detection_cfg`,
    `_classify_occupancy`, `_classify_pieces`, `_squares`) as of chesscog 1.x, so verify
    against chesscog/recognition/recognition.py if it breaks.
    corners: (4, 2) pixel coordinates in the RAW image, ordered top-left, top-right,
    bottom-right, bottom-left (chesscog's sort_corner_points order).
    """
    from chesscog.corner_detection import resize_image

    img, scale = resize_image(recognizer._corner_detection_cfg, img)
    corners = np.asarray(corners, dtype=float) * scale
    occupancy = recognizer._classify_occupancy(img, turn, corners)
    pieces = recognizer._classify_pieces(img, turn, corners, occupancy)
    board = chess.Board(None)
    for square, piece in zip(recognizer._squares, pieces):
        if piece:
            board.set_piece_at(square, piece)
    return board


def run_chesscog(
    image_paths: Sequence[str | Path],
    corner_hint: np.ndarray | None = None,
    turn: chess.Color = chess.WHITE,
    recognizer=None,
) -> list[str]:
    """Run the pretrained chesscog system on full-board images; one board_fen per image.

    IMPORTANT: chesscog finds the board itself in a perspective photo (its corner detector
    looks for the grid lines of the whole board plus surroundings). It needs RAW, unwarped
    camera frames, not our warped top-down frames/*.png -> save raw frames in M2 capture.

    Invocation (chesscog/recognition/recognition.py):
        recognizer = ChessRecognizer()                  # loads models from models://
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)   # expects RGB
        board, corners = recognizer.predict(img, turn)  # chess.Board, (4, 2)
    CLI equivalent: `python -m chesscog.recognition.recognition <image> --white`.

    corner_hint: optional (4, 2) board corners in raw-image pixels (e.g. from our ArUco
        calibration; camera is fixed so one set works for all frames). Bypasses chesscog's
        corner detection via its private internals (see _predict_with_corners). Unverified.
    turn: chesscog uses it for board orientation: WHITE means white's side is at the
        bottom of the image. Make sure it matches our camera placement.
    recognizer: inject an already-built recognizer (tests, or to reuse loaded models).
    Failures (no board found, any exception) give "" for that image instead of a crash.
    Raises ImportError with install instructions if chesscog is missing.
    """
    import cv2

    if recognizer is None:
        recognizer = _load_recognizer()
    fens: list[str] = []
    for path in image_paths:
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if corner_hint is None:
                board, _corners = recognizer.predict(img, turn)
            else:
                board = _predict_with_corners(recognizer, img, corner_hint, turn)
            fens.append(board.board_fen())
        except Exception as e:  # noqa: BLE001 - chesscog raises its own exception types
            print(f"[chesscog] failed on {path}: {type(e).__name__}: {e}")
            fens.append("")
    return fens


def run_chesscog_to_csv(
    frames_glob: str,
    out_csv: str | Path,
    turn: chess.Color | str = chess.WHITE,
    corner_hint: np.ndarray | None = None,
):
    """Run chesscog on every image matching frames_glob and write the CSV compare() reads.

    Expects paths like data/crops/<session_id>/<any dir>/<frame_id>.png (e.g. a raw/ or
    frames/ folder): session_id = grandparent directory name, frame_id = file stem.
    turn may be chess.WHITE/BLACK or "white"/"black". Output columns: session_id,
    frame_id, fen_chesscog. Returns the DataFrame.
    """
    import glob

    import pandas as pd

    if isinstance(turn, str):
        if turn.lower() not in ("white", "black"):
            raise ValueError(f"turn must be 'white' or 'black', got {turn!r}")
        turn = turn.lower() == "white"
    paths = sorted(Path(p) for p in glob.glob(str(frames_glob)))
    if not paths:
        raise FileNotFoundError(f"no images match {frames_glob}")
    fens = run_chesscog(paths, corner_hint=corner_hint, turn=turn)
    df = pd.DataFrame({
        "session_id": [p.parent.parent.name for p in paths],
        "frame_id": [p.stem for p in paths],
        "fen_chesscog": fens,
    })
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def compare(ours_csv: str | Path, chesscog_csv: str | Path, out_csv: str | Path,
            ours_column: str | None = None):
    """Join our per-board predictions with chesscog's and write chesscog_vs_ours.csv.

    ours_csv columns: session_id, frame_id, fen_true and our prediction column. That is
    `ours_column` if given (e.g. "fen_pred_repaired"), else fen_ours, else fen_pred (so
    evaluate's boards.csv works directly). It is renamed to fen_ours in the output.
    chesscog_csv columns: session_id, frame_id, fen_chesscog
    Output adds: correct_ours, correct_chesscog, squares_wrong_ours, squares_wrong_chesscog.
    Returns the merged DataFrame.
    """
    import pandas as pd

    ours = pd.read_csv(ours_csv, dtype=str, keep_default_na=False)
    if ours_column is None:
        ours_column = "fen_ours" if "fen_ours" in ours.columns else "fen_pred"
    if ours_column not in ours.columns:
        raise ValueError(f"{ours_csv} has no column {ours_column!r}")
    ours = ours.rename(columns={ours_column: "fen_ours"})
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
