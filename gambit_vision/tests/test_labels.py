import chess

from gambit_vision.labels import (
    CLASSES, NUM_CLASSES, board_to_labels, label_to_color3, labels_to_board,
)


def test_thirteen_classes():
    assert NUM_CLASSES == 13 and CLASSES[0] == "empty"


def test_board_roundtrip():
    board = chess.Board()
    labels = board_to_labels(board)
    assert labels[chess.E1] == "K" and labels[chess.E8] == "k" and labels[chess.E4] == "empty"
    assert labels_to_board(labels).board_fen() == board.board_fen()


def test_color3():
    assert [label_to_color3(x) for x in ("empty", "Q", "q")] == [0, 1, 2]
