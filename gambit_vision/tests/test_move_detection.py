import chess
import numpy as np
import pytest

from gambit_vision.labels import CLASS_TO_INDEX, NUM_CLASSES, board_to_labels
from gambit_vision.move_detection import (
    MoveTracker, State, StableFrameDetector, changed_squares, changed_squares_between,
    evaluate_on_game, frame_difference, infer_move, render_board, simulate_game_frames,
)

# 15 plies with en passant (exd6), a promotion (cxb8=Q) and castling on both sides.
GAME_PGN = "1. e4 Nf6 2. e5 d5 3. exd6 e6 4. dxc7 Be7 5. cxb8=Q Rxb8 6. Nf3 O-O 7. Be2 b6 8. O-O *"



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


# --------------------------------------------------------------- detector extensions

def test_blur_downsample_suppresses_sensor_noise():
    rng = np.random.default_rng(1)
    a = np.full((64, 64), 100.0) + rng.normal(0, 5, (64, 64))
    b = np.full((64, 64), 100.0) + rng.normal(0, 5, (64, 64))
    raw = frame_difference(a, b)
    smooth = frame_difference(a, b, downsample=4, blur_sigma=1.0)
    assert smooth < raw / 3


def test_per_square_changed_squares():
    before = chess.Board()
    after_ = before.copy()
    after_.push_uci("g1f3")
    sq = 8
    changed = changed_squares_between(render_board(before, sq), render_board(after_, sq),
                                      square_px=sq)
    assert changed == ["g1", "f3"]  # square-index order
    # same with a margin and the detector's downsampled geometry
    pad = lambda img: np.pad(img, 16, constant_values=0)  # noqa: E731
    det = StableFrameDetector(settle_frames=3, square_px=sq * 2, margin_px=16, downsample=2)
    big = lambda b: pad(np.kron(render_board(b, sq), np.ones((2, 2))))  # noqa: E731
    rng = np.random.default_rng(0)
    frames = [big(before)] * 3 + [rng.uniform(0, 255, big(before).shape) for _ in range(3)] \
        + [big(after_)] * 5
    fired = [i for i, f in enumerate(frames) if det.update(f)]
    assert len(fired) == 1
    assert det.changed_squares() == ["g1", "f3"]


def test_hand_resting_on_board_delays_stable_event():
    rng = np.random.default_rng(0)
    board = np.full((64, 64), 100.0)
    hand = board.copy()
    hand[0:40, 0:40] = 220  # a big still object (hand) covering ~40% of the board
    frames = ([board] * 3 + [rng.uniform(0, 255, board.shape) for _ in range(3)]
              + [hand] * 8 + [rng.uniform(0, 255, board.shape) for _ in range(3)]
              + [board] * 6)
    det = StableFrameDetector(settle_frames=4, hand_timeout_frames=100)
    fired = [i for i, f in enumerate(frames) if det.update(f)]
    # no event while the hand rests (frames 6..13); one event after it left
    assert fired == [21]
    # without the hand check it would have fired while the hand was resting
    det = StableFrameDetector(settle_frames=4, hand_check=False)
    assert [i for i, f in enumerate(frames) if det.update(f)][0] < 14


# --------------------------------------------------------------- infer_move extensions

def test_unknown_side_to_move():
    true = chess.Board()
    true.push_uci("e2e4")
    wrong_turn = true.copy(stack=False)
    wrong_turn.turn = chess.WHITE  # tracker believes white is to move
    nxt = true.copy()
    nxt.push_uci("e7e5")
    res = infer_move(wrong_turn, one_hot(nxt), unknown_turn=True)
    assert res.move == chess.Move.from_uci("e7e5")
    assert res.turn == chess.BLACK and res.turn_flipped and not res.ambiguous
    # without unknown_turn the black move is not even a candidate
    assert infer_move(wrong_turn, one_hot(nxt)).move != chess.Move.from_uci("e7e5")


def test_two_ply_candidates():
    board = chess.Board()
    nxt = board.copy()
    nxt.push_uci("e2e4")
    nxt.push_uci("c7c5")
    single = infer_move(board, one_hot(nxt, noise=0.05))
    assert not single.two_ply_searched
    res = infer_move(board, one_hot(nxt, noise=0.05), two_ply_threshold=4.0)
    assert res.two_ply_searched and res.deficit > 4.0
    assert res.moves == [chess.Move.from_uci("e2e4"), chess.Move.from_uci("c7c5")]
    assert res.san == "e4 c5" and not res.ambiguous
    # a normal single move does not trigger the (expensive) pair search
    one = board.copy()
    one.push_uci("d2d4")
    assert not infer_move(board, one_hot(one, 0.05), two_ply_threshold=4.0).two_ply_searched


# --------------------------------------------------------------- tracker / evaluation

def test_evaluate_on_game_simulated_special_moves(tmp_path):
    sim = simulate_game_frames(GAME_PGN, noise=0.05, seed=0)
    m = evaluate_on_game(GAME_PGN, sim.frames, sim.board_proba, frame_plies=sim.frame_plies,
                         out_dir=tmp_path)
    assert m["n_moves"] == 15
    assert m["move_accuracy"] == 1.0 and m["game_completed"]
    assert m["missed"] == m["spurious"] == m["wrong"] == m["ambiguous"] == 0
    sp = m["special_moves"]
    assert sp["castling"] == {"total": 2, "correct": 2, "accuracy": 1.0}
    assert sp["en_passant"]["correct"] == sp["en_passant"]["total"] == 1
    assert sp["promotion"]["correct"] == sp["promotion"]["total"] == 1
    assert (tmp_path / "move_metrics.json").exists()


def test_tracker_ignores_spurious_motion():
    sim = simulate_game_frames(GAME_PGN, noise=0.05, seed=1, spurious_motion_after=[0, 5])
    tracker = MoveTracker(sim.board_proba)
    events = tracker.run(sim.frames)
    assert len(events) == 15 and len(tracker.no_move_events) == 2
    assert tracker.board.board_fen() == sim.boards[-1].board_fen()
    m = evaluate_on_game(GAME_PGN, sim.frames, sim.board_proba)  # likelihood alignment
    assert m["move_accuracy"] == 1.0 and m["spurious"] == 0


def test_missed_stable_frame_recovered_by_two_ply():
    # no stable pause between 6. Nf3 and 6... O-O
    sim = simulate_game_frames(GAME_PGN, noise=0.05, seed=2, skip_pause_after=[11])
    m = evaluate_on_game(GAME_PGN, sim.frames, sim.board_proba, frame_plies=sim.frame_plies)
    assert m["move_accuracy"] == 1.0 and m["game_completed"]
    assert any(e["detected"] == "Nf3 O-O" for e in m["events"])
    # with the two-ply search disabled the two skipped plies are lost
    m = evaluate_on_game(GAME_PGN, sim.frames, sim.board_proba, frame_plies=sim.frame_plies,
                         two_ply_threshold=None)
    assert m["move_accuracy"] < 1.0 and not m["game_completed"]
    assert m["missed"] + m["wrong"] >= 1


def test_tracker_unknown_side_to_move_mid_game():
    pgn = '[FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"]\n\n2... Nc6 3. Bb5 a6 *'
    sim = simulate_game_frames(pgn, noise=0.05, seed=3)
    start = sim.boards[0].copy()
    start.turn = chess.WHITE  # wrong: actually black to move
    tracker = MoveTracker(sim.board_proba, start_board=start, unknown_turn=True)
    events = tracker.run(sim.frames)
    assert [e.result.san for e in events] == ["Nc6", "Bb5", "a6"]
    assert events[0].result.turn == chess.BLACK and events[0].result.turn_flipped
    assert tracker.board.board_fen() == sim.boards[-1].board_fen()


def test_evaluate_on_game_from_frames_directory(tmp_path):
    cv2 = pytest.importorskip("cv2")
    sim = simulate_game_frames(GAME_PGN, noise=0.05, seed=4)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i, f in enumerate(sim.frames):  # unpadded names: natural sort must handle 2 < 10
        cv2.imwrite(str(frames_dir / f"frame_{i}.png"), np.clip(f, 0, 255).astype(np.uint8))
    pgn_file = tmp_path / "game.pgn"
    pgn_file.write_text(GAME_PGN)
    m = evaluate_on_game(pgn_file, frames_dir, sim.board_proba, out_dir=tmp_path / "res")
    assert m["move_accuracy"] == 1.0 and m["game_completed"]


def test_ambiguous_result_asks_callback():
    board = chess.Board()
    # uninformative probabilities -> every candidate ties -> ambiguous
    flat = np.full((64, NUM_CLASSES), 1.0 / NUM_CLASSES)
    asked = []

    def ask(b, res):
        asked.append(res)
        return chess.Move.from_uci("d2d4")

    det = StableFrameDetector(settle_frames=2, hand_check=False)
    tracker = MoveTracker(lambda f: flat, start_board=board, detector=det,
                          two_ply_threshold=None, on_ambiguous=ask)
    rng = np.random.default_rng(0)
    still = np.full((32, 32), 50.0)
    frames = [still] * 2 + [rng.uniform(0, 255, still.shape) for _ in range(3)] + [still] * 3
    events = tracker.run(frames)
    assert asked and len(events) == 1 and events[0].result.san == "d4"
    # without a callback the ambiguous frame is recorded and the board kept
    tracker = MoveTracker(lambda f: flat, detector=StableFrameDetector(settle_frames=2,
                          hand_check=False), two_ply_threshold=None)
    assert tracker.run(frames) == [] and len(tracker.ambiguous_events) == 1
    assert tracker.board == chess.Board()


def test_two_ply_genuinely_ambiguous_pair_is_flagged():
    # Missing the frame after 2... d5 leaves "d5 exd6 e.p." and "d6 exd6" with the SAME
    # final placement: the tracker must flag this instead of guessing.
    board = chess.Board()
    for uci in ("e2e4", "g8f6", "e4e5"):
        board.push_uci(uci)
    nxt = board.copy()
    nxt.push_uci("d7d5")
    nxt.push_uci("e5d6")
    res = infer_move(board, one_hot(nxt, 0.05), two_ply_threshold=4.0)
    assert res.ambiguous and res.margin == 0.0
    pairs = {res.san}
    alt = board.copy()
    alt.push(res.runner_up[0])
    pairs.add(board.san(res.runner_up[0]) + " " + alt.san(res.runner_up[1]))
    assert pairs == {"d5 exd6", "d6 exd6"}
