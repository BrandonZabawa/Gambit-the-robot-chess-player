import chess
import numpy as np
import pytest

from gambit_vision.labels import CLASS_TO_INDEX, NUM_CLASSES, board_to_labels
from gambit_vision.move_detection import (
    State, StableFrameDetector, changed_squares, evaluate_on_game, infer_move,
)


def one_hot(board: chess.Board, noise: float = 0.0) -> np.ndarray:
    probs = np.full((64, NUM_CLASSES), noise / NUM_CLASSES)
    for sq, lab in enumerate(board_to_labels(board)):
        probs[sq, CLASS_TO_INDEX[lab]] += 1.0 - noise
    return probs


def after(fen: str, uci: str) -> tuple[chess.Board, np.ndarray]:
    board = chess.Board(fen)
    nxt = board.copy()
    nxt.push_uci(uci)
    return board, one_hot(nxt)


@pytest.mark.parametrize("fen,uci,san", [
    (chess.STARTING_FEN, "e2e4", "e4"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "O-O"),
    ("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6", "exd6"),
    ("8/P7/8/8/8/8/k6K/8 w - - 0 1", "a7a8q", "a8=Q+"),
])
def test_infer_move(fen, uci, san):
    board, probs = after(fen, uci)
    res = infer_move(board, probs)
    assert res.move == chess.Move.from_uci(uci)
    assert res.san == san
    assert not res.ambiguous and res.margin > 0


def test_castling_queenside_distinct_from_kingside():
    board, probs = after("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1")
    assert infer_move(board, probs).san == "O-O-O"


def test_unchanged_board_is_no_move():
    board = chess.Board()
    res = infer_move(board, one_hot(board))
    assert res.move is None and res.san is None and not res.ambiguous


def test_noisy_probs_still_pick_move():
    board, _ = after(chess.STARTING_FEN, "g1f3")
    nxt = board.copy()
    nxt.push_uci("g1f3")
    assert infer_move(board, one_hot(nxt, noise=0.5)).move == chess.Move.from_uci("g1f3")


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        infer_move(chess.Board(), np.zeros((64, 12)))


def test_changed_squares():
    board = chess.Board()
    prev = board_to_labels(board)
    board.push_uci("e2e4")
    assert changed_squares(prev, board_to_labels(board)) == ["e2", "e4"]


def test_stable_frame_detector():
    rng = np.random.default_rng(0)
    still_a = np.full((64, 64), 100.0)
    still_b = still_a.copy()
    still_b[10:20, 10:20] = 200  # a piece moved
    frames = (
        [still_a + rng.normal(0, 0.5, still_a.shape) for _ in range(5)]       # static
        + [rng.uniform(0, 255, still_a.shape) for _ in range(4)]               # hand
        + [still_b + rng.normal(0, 0.5, still_a.shape) for _ in range(12)]     # static
    )
    det = StableFrameDetector(diff_threshold=4.0, settle_frames=5)
    fired, refs = [], []
    for i, f in enumerate(frames):
        if det.update(f):
            fired.append(i)
            refs.append(det.reference_frame)
    # first still frame after motion is index 9 -> 1 diff vs hand frame (moving),
    # then still diffs from index 10; 5 in a row -> fires at index 14
    assert fired == [14]
    assert det.stable_frame is frames[14]
    assert refs[0] is frames[4]  # last still frame before the hand appeared
    assert det.state == State.IDLE


@pytest.mark.skip(reason="TODO(M5): needs a recorded game (PGN + frames)")
def test_evaluate_on_game_perfect_classifier():
    # Idea: render/collect frames for a short PGN, use a board_proba that returns one-hot
    # probabilities of the true position, expect move_accuracy == 1.0.
    evaluate_on_game("data/games/g01.pgn", "data/games/g01/frames")


@pytest.mark.skip(reason="TODO(M5): needs a recorded game (PGN + frames)")
def test_evaluate_on_game_reports_special_moves():
    pass
