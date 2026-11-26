import chess
import numpy as np
import pytest

from gambit_vision.board_assembly import BoardPrediction, assemble, check_legality
from gambit_vision.labels import CLASS_TO_INDEX, NUM_CLASSES, board_to_labels


def one_hot(board: chess.Board) -> np.ndarray:
    probs = np.zeros((64, NUM_CLASSES))
    for sq, lab in enumerate(board_to_labels(board)):
        probs[sq, CLASS_TO_INDEX[lab]] = 1.0
    return probs


def test_start_position_roundtrip():
    pred = assemble(one_hot(chess.Board()))
    assert isinstance(pred, BoardPrediction)
    assert pred.fen == chess.STARTING_BOARD_FEN
    assert pred.legal and pred.issues == [] and pred.low_confidence == []
    assert len(pred.labels) == 64 and pred.labels[chess.E1] == "K"
    assert pred.confidence == [1.0] * 64


def test_low_confidence_flagged():
    probs = one_hot(chess.Board())
    probs[chess.E4] = 0.0
    probs[chess.E4, CLASS_TO_INDEX["empty"]] = 0.5
    probs[chess.E4, CLASS_TO_INDEX["P"]] = 0.5 - 1e-6
    probs[chess.E4, CLASS_TO_INDEX["p"]] = 1e-6
    pred = assemble(probs, threshold=0.6)
    assert pred.low_confidence == ["e4"]
    assert pred.labels[chess.E4] == "empty"


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        assemble(np.zeros((63, NUM_CLASSES)))


def test_two_white_kings_illegal():
    board = chess.Board()
    board.set_piece_at(chess.D1, chess.Piece.from_symbol("K"))  # queen -> second king
    pred = assemble(one_hot(board))
    assert not pred.legal
    assert any("white has 2 kings" in s for s in pred.issues)


def test_missing_king_and_back_rank_pawn():
    board = chess.Board()
    board.remove_piece_at(chess.E8)
    board.set_piece_at(chess.A8, chess.Piece.from_symbol("P"))
    issues = check_legality(board)
    assert any("black has 0 kings" in s for s in issues)
    assert any("a8" in s for s in issues)


def test_too_many_pawns():
    board = chess.Board()
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("P"))
    assert any("9 P" in s for s in check_legality(board))


def test_expected_board_mismatch_reported_but_still_legal():
    after = chess.Board()
    after.push_san("e4")
    pred = assemble(one_hot(after), expected_board=chess.Board())
    assert pred.legal
    assert any("e2" in s and "e4" in s for s in pred.issues)


@pytest.mark.skip(reason="TODO M4: repair() not implemented")
def test_repair_fixes_second_king():
    pass


@pytest.mark.skip(reason="TODO M4: board.status()-based checks (kings adjacent, check)")
def test_kings_adjacent_illegal():
    pass
