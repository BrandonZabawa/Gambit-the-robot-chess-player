"""Shared label and square conventions. Every module imports from here.

Squares use python-chess indexing: a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
Classes use FEN piece letters (uppercase white, lowercase black) plus "empty".
"""

from __future__ import annotations

import chess

# Order is fixed: model output index i means CLASSES[i]. Do not reorder after training.
CLASSES: tuple[str, ...] = (
    "empty",
    "P", "N", "B", "R", "Q", "K",   # white
    "p", "n", "b", "r", "q", "k",   # black
)
NUM_CLASSES = len(CLASSES)  # 13
CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
EMPTY = 0

# Baseline (M3) predicts coarser targets derived from the 13 classes.
OCCUPANCY = ("empty", "occupied")
COLOR3 = ("empty", "white", "black")

SQUARE_NAMES: tuple[str, ...] = tuple(chess.SQUARE_NAMES)  # index -> "a1".."h8"


def label_to_occupancy(label: str) -> int:
    return 0 if label == "empty" else 1


def label_to_color3(label: str) -> int:
    if label == "empty":
        return 0
    return 1 if label.isupper() else 2


def board_to_labels(board: chess.Board) -> list[str]:
    """64 labels in square-index order for a python-chess board."""
    out = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        out.append("empty" if piece is None else piece.symbol())
    return out


def labels_to_board(labels: list[str]) -> chess.Board:
    """Inverse of board_to_labels. Piece placement only; turn/castling/ep left default."""
    if len(labels) != 64:
        raise ValueError(f"expected 64 labels, got {len(labels)}")
    board = chess.Board(None)
    for sq, lab in enumerate(labels):
        if lab != "empty":
            board.set_piece_at(sq, chess.Piece.from_symbol(lab))
    return board
