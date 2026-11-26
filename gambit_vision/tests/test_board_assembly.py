import chess
import numpy as np
import pytest

from gambit_vision.board_assembly import BoardPrediction, assemble, check_legality, repair
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


def test_repair_fixes_second_king():
    # d1 looks like a king (0.6) but its second-best class is the true queen (0.4).
    probs = one_hot(chess.Board())
    probs[chess.D1] = 0.0
    probs[chess.D1, CLASS_TO_INDEX["K"]] = 0.6
    probs[chess.D1, CLASS_TO_INDEX["Q"]] = 0.4
    assert not assemble(probs).legal

    fixed = repair(probs, ["white has 2 kings (expected 1)"])
    assert fixed is not None and fixed.shape == (64, NUM_CLASSES)
    assert fixed[chess.D1].argmax() == CLASS_TO_INDEX["Q"]
    assert fixed[chess.E1].argmax() == CLASS_TO_INDEX["K"]  # e1 has p(K)=1 > 0.6, keeps the king

    pred = assemble(probs, repair=True)
    assert pred.legal and pred.repaired
    assert pred.fen == chess.STARTING_BOARD_FEN
    assert pred.changed_squares == ["d1"]
    assert any("fixed: white has 2 kings" in s for s in pred.issues)
    assert any(s.startswith("repaired d1: K -> Q") for s in pred.issues)
    assert pred.confidence[chess.D1] == pytest.approx(0.4)
    assert "d1" in pred.low_confidence


def test_repair_missing_king_and_back_rank_pawn():
    probs = one_hot(chess.Board())
    # black king predicted as queen (k is the runner-up), h8 rook predicted as pawn
    probs[chess.E8] = 0.0
    probs[chess.E8, CLASS_TO_INDEX["q"]] = 0.7
    probs[chess.E8, CLASS_TO_INDEX["k"]] = 0.3
    probs[chess.H8] = 0.0
    probs[chess.H8, CLASS_TO_INDEX["p"]] = 0.8
    probs[chess.H8, CLASS_TO_INDEX["r"]] = 0.2
    pred = assemble(probs, repair=True)
    assert pred.repaired and pred.fen == chess.STARTING_BOARD_FEN
    assert sorted(pred.changed_squares) == ["e8", "h8"]


def test_repair_greedy_fixes_extra_pawn():
    # A ninth white pawn on an empty square: greedy step switches it to its runner-up "empty".
    board = chess.Board()
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("P"))
    probs = one_hot(board)
    probs[chess.E4] = 0.0
    probs[chess.E4, CLASS_TO_INDEX["P"]] = 0.55
    probs[chess.E4, CLASS_TO_INDEX["empty"]] = 0.45
    pred = assemble(probs, repair=True)
    assert pred.repaired and pred.fen == chess.STARTING_BOARD_FEN


def test_repair_legal_board_unchanged_and_impossible_returns_none():
    probs = one_hot(chess.Board())
    assert np.array_equal(repair(probs, []), probs)
    # adjacent kings with no alternative class anywhere: cannot be fixed by switching
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece.from_symbol("K"))
    board.set_piece_at(chess.E2, chess.Piece.from_symbol("k"))
    hard = np.zeros((64, NUM_CLASSES))
    for sq, lab in enumerate(board_to_labels(board)):
        hard[sq, CLASS_TO_INDEX[lab]] = 1.0
    assert repair(hard, ["kings adjacent"]) is None
    pred = assemble(hard, repair=True)
    assert not pred.legal and any("repair failed" in s for s in pred.issues)


def test_kings_adjacent_illegal():
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("K"))
    board.set_piece_at(chess.F5, chess.Piece.from_symbol("k"))
    assert any("kings adjacent" in s for s in check_legality(board))
    board.remove_piece_at(chess.F5)
    board.set_piece_at(chess.G6, chess.Piece.from_symbol("k"))
    assert check_legality(board) == []


def test_check_rules_with_and_without_turn():
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece.from_symbol("K"))
    board.set_piece_at(chess.E8, chess.Piece.from_symbol("k"))
    board.set_piece_at(chess.E5, chess.Piece.from_symbol("r"))  # black rook checks white king
    assert check_legality(board) == []                       # fine if white is to move
    assert check_legality(board, turn=chess.WHITE) == []
    assert any("white (not to move) is in check" in s for s in check_legality(board, turn=chess.BLACK))
    board.set_piece_at(chess.A8, chess.Piece.from_symbol("R"))  # now black is in check too
    assert any("both kings in check" in s for s in check_legality(board))


def test_promotion_aware_piece_caps():
    board = chess.Board()
    board.set_piece_at(chess.D4, chess.Piece.from_symbol("Q"))   # 2 queens, 8 pawns
    assert any("promoted pieces" in s for s in check_legality(board))
    board.remove_piece_at(chess.A2)                              # one pawn missing -> ok
    assert check_legality(board) == []
