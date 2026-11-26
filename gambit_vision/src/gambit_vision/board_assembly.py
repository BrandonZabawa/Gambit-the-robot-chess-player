"""M4: combine 64 square predictions into a board state.

Input: a (64, 13) probability array from any square classifier (rows in python-chess
square order a1..h8, columns in labels.CLASSES order). See docs/contracts.md.
Output: a BoardPrediction (FEN, per-square labels/confidence, low-confidence squares,
legality verdict).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from gambit_vision.labels import CLASSES, NUM_CLASSES, SQUARE_NAMES, labels_to_board

# Max count per piece letter in a reachable position (8 pawns; promotions ignored here).
# TODO M4: promotions make e.g. 3 queens legal; decide whether to allow
# "pawns + extra pieces <= 8 + original pieces" instead of fixed caps.
MAX_PIECES = {"K": 1, "Q": 9, "R": 10, "B": 10, "N": 10, "P": 8}
MAX_PER_COLOR = 16


@dataclass
class BoardPrediction:
    fen: str                       # piece placement only (board_fen), e.g. "rnbqkbnr/pppppppp/8/..."
    labels: list[str]              # 64 labels, square-index order
    confidence: list[float]        # 64 max-probabilities
    low_confidence: list[str] = field(default_factory=list)  # square names below threshold
    legal: bool = True
    issues: list[str] = field(default_factory=list)          # human-readable problems


def check_legality(board: chess.Board) -> list[str]:
    """Basic plausibility checks on piece placement. Empty list = no problems found."""
    issues: list[str] = []
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        n_kings = len(board.pieces(chess.KING, color))
        if n_kings != 1:
            issues.append(f"{name} has {n_kings} kings (expected 1)")

        total = sum(len(board.pieces(pt, color)) for pt in chess.PIECE_TYPES)
        if total > MAX_PER_COLOR:
            issues.append(f"{name} has {total} pieces (max {MAX_PER_COLOR})")

        for pt in chess.PIECE_TYPES:
            symbol = chess.piece_symbol(pt).upper()
            n = len(board.pieces(pt, color))
            if symbol != "K" and n > MAX_PIECES[symbol]:
                issues.append(f"{name} has {n} {symbol} (max {MAX_PIECES[symbol]})")

    back_ranks = chess.BB_RANK_1 | chess.BB_RANK_8
    for sq in chess.SquareSet(board.pawns & back_ranks):
        issues.append(f"pawn on back rank at {chess.square_name(sq)}")

    # TODO M4: further checks, e.g. board.status() (side not to move in check,
    # too many checkers, opposing kings adjacent). Note board_fen has no turn info, so
    # try both turns or take the turn from the game record.
    return issues


def assemble(
    probs: np.ndarray,
    threshold: float = 0.6,
    expected_board: chess.Board | None = None,
) -> BoardPrediction:
    """Turn (64, 13) class probabilities into a BoardPrediction.

    threshold: squares whose top probability is below this are flagged as low-confidence
        (config: assembly.low_confidence_threshold).
    expected_board: optional prior (e.g. the previous position from M5). Currently only
        used to report disagreements in `issues`.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (64, NUM_CLASSES):
        raise ValueError(f"expected probs of shape (64, {NUM_CLASSES}), got {probs.shape}")

    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    labels = [CLASSES[i] for i in idx]
    board = labels_to_board(labels)

    low = [SQUARE_NAMES[sq] for sq in range(64) if conf[sq] < threshold]
    issues = check_legality(board)
    legal = not issues  # expected_board mismatch below is informative, not illegal

    if expected_board is not None:
        # TODO M4/M5: use the prior properly (e.g. only allow positions one legal move away).
        diff = [SQUARE_NAMES[sq] for sq in range(64)
                if board.piece_at(sq) != expected_board.piece_at(sq)]
        if diff:
            issues.append(f"differs from expected board at {', '.join(diff)}")

    # TODO M4: if not legal, try repair(probs, issues) and re-check.
    return BoardPrediction(
        fen=board.board_fen(),
        labels=labels,
        confidence=conf.tolist(),
        low_confidence=low,
        legal=legal,
        issues=issues,
    )


def repair(probs: np.ndarray, issues: list[str]) -> np.ndarray | None:
    """TODO M4: constrained fixing of an illegal board.

    Idea: among low-confidence squares, swap the lowest-confidence one to its
    second-best class (or search the k lowest squares x top-2 classes), re-run
    check_legality, and keep the most probable legal assignment (sum of log-probs).
    Return the modified (64, 13) probs (or a label list), or None if no legal fix found.
    Report how often repair helps in the evaluation.
    """
    raise NotImplementedError("TODO M4: repair()")
