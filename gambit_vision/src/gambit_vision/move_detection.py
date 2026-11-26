"""M5 (optional): move detection from a stream of warped top-down board images.

Pipeline per move:
    1. StableFrameDetector watches frame-to-frame differences and fires once the board
       is still again after a period of motion (the player's hand came in and left).
    2. The stable frame is cut into 64 crops and classified -> (64, 13) probabilities.
    3. infer_move() scores every legal move of the previous position (plus "no move")
       against those probabilities and returns the most likely one. Because candidates
       come from python-chess, castling, en passant and promotion need no special cases.

Status: infer_move / changed_squares are implemented; StableFrameDetector has simple
working logic but untuned thresholds; MoveTracker and evaluate_on_game are skeletons.
See docs/M5_TODO.md. This module must import without torch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import numpy as np

from gambit_vision.labels import CLASS_TO_INDEX, NUM_CLASSES, SQUARE_NAMES, board_to_labels

if TYPE_CHECKING:  # board_assembly is M4; import lazily at runtime
    from gambit_vision.board_assembly import BoardPrediction  # noqa: F401


# --------------------------------------------------------------------------- stable frames

class State(str, Enum):
    IDLE = "idle"          # board static, nothing happening
    MOTION = "motion"      # large frame differences (hand over the board)
    SETTLING = "settling"  # differences small again, waiting for settle_frames in a row
    STABLE = "stable"      # emitted for exactly one frame after settling; then back to IDLE


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two frames, in [0, 255] intensity units.

    Color frames are averaged to grayscale first. TODO(M5): consider downsampling or a
    Gaussian blur first to suppress sensor noise / flicker; try a per-square max instead
    of a global mean so a small hand at the board edge is not averaged away.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"frame shapes differ: {a.shape} vs {b.shape}")
    if a.ndim == 3:
        a, b = a.mean(axis=2), b.mean(axis=2)
    return float(np.abs(a - b).mean())


class StableFrameDetector:
    """State machine IDLE -> MOTION -> SETTLING -> STABLE -> IDLE.

    Feed frames with update(frame); it returns True exactly once per motion episode, on the
    frame where the board has been still for `settle_frames` consecutive frames after at
    least `min_motion_frames` frames of motion. `stable_frame` / `reference_frame` hold the
    frames after and before the episode.

    Defaults are guesses. TODO(M5): tune diff_threshold / settle_frames on recorded
    video (plot frame_difference over time for a few games), and decide whether to add
    an explicit hand detector (e.g. skin-color mask or "occupancy outside the start/end
    squares") so slow hand movements are not mistaken for a still board.
    """

    def __init__(self, diff_threshold: float = 4.0, settle_frames: int = 10,
                 min_motion_frames: int = 2) -> None:
        self.diff_threshold = diff_threshold
        self.settle_frames = settle_frames
        self.min_motion_frames = min_motion_frames
        self.reset()

    def reset(self) -> None:
        self.state = State.IDLE
        self.prev_frame: np.ndarray | None = None
        self.reference_frame: np.ndarray | None = None  # last still frame before motion
        self.stable_frame: np.ndarray | None = None     # frame at which STABLE fired
        self.last_diff = 0.0
        self._motion_count = 0
        self._still_count = 0

    def update(self, frame: np.ndarray) -> bool:
        if self.prev_frame is None:
            self.prev_frame = self.reference_frame = frame
            return False
        diff = frame_difference(self.prev_frame, frame)
        self.last_diff = diff
        moving = diff > self.diff_threshold
        fired = False

        if self.state in (State.IDLE, State.STABLE):
            if moving:
                self.state, self._motion_count = State.MOTION, 1
            else:
                self.state = State.IDLE
                self.reference_frame = frame
        elif self.state == State.MOTION:
            if moving:
                self._motion_count += 1
            elif self._motion_count >= self.min_motion_frames:
                self.state, self._still_count = State.SETTLING, 1
            else:  # a one-frame blip (noise, flicker): ignore it
                self.state = State.IDLE
        elif self.state == State.SETTLING:
            if moving:
                self.state = State.MOTION
                self._motion_count += 1
            else:
                self._still_count += 1

        if self.state == State.SETTLING and self._still_count >= self.settle_frames:
            self.state = State.STABLE
            self.stable_frame = frame
            fired = True

        self.prev_frame = frame
        return fired


# --------------------------------------------------------------------------- move inference

@dataclass
class MoveResult:
    move: chess.Move | None        # None = board judged unchanged
    san: str | None                # SAN in the previous position, None for "no move"
    score: float                   # log-likelihood of the best candidate
    runner_up: chess.Move | None   # second-best candidate (None may mean "no move")
    margin: float                  # score(best) - score(runner_up), >= 0
    ambiguous: bool                # margin below threshold -> ask for confirmation / wait
    ranking: list[tuple[chess.Move | None, float]] = field(default_factory=list, repr=False)


def placement_log_likelihood(labels: Sequence[str], log_probs: np.ndarray) -> float:
    """Sum over squares of log P(square has this label)."""
    idx = np.fromiter((CLASS_TO_INDEX[x] for x in labels), dtype=np.int64, count=64)
    return float(log_probs[np.arange(64), idx].sum())


def infer_move(prev_board: chess.Board, square_probs_after: np.ndarray, *,
               include_null: bool = True, ambiguity_margin: float = 2.0,
               eps: float = 1e-6) -> MoveResult:
    """Most likely legal move from prev_board given (64, 13) probabilities of the new frame.

    Each candidate (every legal move, plus "no move" if include_null) is pushed onto a copy
    of prev_board and scored by the log-likelihood of the resulting piece placement, treating
    squares as independent. `ambiguity_margin` is in nats: 2.0 means the best candidate is
    ~7x more likely than the runner-up.

    TODO(M5): if the side to move is uncertain (e.g. tracking started mid-game), also try
    the null move to flip turns; consider two-ply candidates when a stable frame was missed.
    """
    probs = np.asarray(square_probs_after, dtype=np.float64)
    if probs.shape != (64, NUM_CLASSES):
        raise ValueError(f"expected (64, {NUM_CLASSES}) probabilities, got {probs.shape}")
    log_probs = np.log(np.clip(probs, eps, 1.0))

    ranking: list[tuple[chess.Move | None, float]] = []
    if include_null:
        ranking.append((None, placement_log_likelihood(board_to_labels(prev_board), log_probs)))
    board = prev_board.copy(stack=False)
    for move in prev_board.legal_moves:
        board.push(move)
        ranking.append((move, placement_log_likelihood(board_to_labels(board), log_probs)))
        board.pop()
    if not ranking:  # game over and include_null=False
        return MoveResult(None, None, float("-inf"), None, 0.0, True)

    ranking.sort(key=lambda t: t[1], reverse=True)
    best, best_score = ranking[0]
    runner_up, second = ranking[1] if len(ranking) > 1 else (None, float("-inf"))
    margin = best_score - second
    return MoveResult(
        move=best,
        san=prev_board.san(best) if best is not None else None,
        score=best_score,
        runner_up=runner_up,
        margin=margin,
        ambiguous=margin < ambiguity_margin,
        ranking=ranking,
    )


def changed_squares(prev_labels: Sequence[str], new_labels: Sequence[str]) -> list[str]:
    """Names of squares whose label differs, in square-index order (a1 first)."""
    if len(prev_labels) != 64 or len(new_labels) != 64:
        raise ValueError("expected 64 labels each")
    return [SQUARE_NAMES[i] for i, (a, b) in enumerate(zip(prev_labels, new_labels)) if a != b]


# --------------------------------------------------------------------------- tracking

ProbaFn = Callable[[np.ndarray], np.ndarray]  # warped frame -> (64, 13) probabilities


@dataclass
class MoveEvent:
    frame_index: int
    result: MoveResult
    fen_after: str


class MoveTracker:
    """Detector + classifier + infer_move over a frame stream, maintaining the game board.

    `board_proba` maps a warped top-down frame to (64, 13) probabilities; build it from
    capture.py's crop function + an M3 model's predict_proba.
    """

    def __init__(self, board_proba: ProbaFn, start_board: chess.Board | None = None,
                 detector: StableFrameDetector | None = None,
                 ambiguity_margin: float = 2.0) -> None:
        self.board_proba = board_proba
        self.board = start_board.copy() if start_board is not None else chess.Board()
        self.detector = detector or StableFrameDetector()
        self.ambiguity_margin = ambiguity_margin
        self.events: list[MoveEvent] = []
        self._frame_index = 0

    def process_frame(self, frame: np.ndarray) -> MoveEvent | None:
        """Feed one warped frame; return a MoveEvent when a move was recognized."""
        self._frame_index += 1
        if not self.detector.update(frame):
            return None
        # TODO(M5):
        #   probs = self.board_proba(frame)
        #   result = infer_move(self.board, probs, ambiguity_margin=self.ambiguity_margin)
        #   - result.move is None -> hand passed over / piece adjusted; no event
        #   - result.ambiguous -> keep the frame, wait for the next stable frame or flag
        #     for the user/robot instead of pushing a guess
        #   - else push onto self.board, append MoveEvent(self._frame_index, result,
        #     self.board.fen()) to self.events and return it
        #   Optionally cross-check with board_assembly.assemble(probs) (lazy import).
        raise NotImplementedError("MoveTracker.process_frame: see TODO(M5)")

    def run(self, frames: Iterable[np.ndarray]) -> list[MoveEvent]:
        """Process a whole stream (e.g. frames read from a video and warped by homography)."""
        for frame in frames:
            self.process_frame(frame)
        return self.events


def evaluate_on_game(pgn_path: str | Path, frames_dir: str | Path,
                     board_proba: ProbaFn | None = None, **tracker_kwargs) -> dict:
    """Replay one recorded game through MoveTracker and score it against its PGN.

    Inputs: the game's PGN (ground truth) and a directory of warped frames in time order
    (e.g. data/crops/<session>/frames/ or frames extracted from the game video).

    Returns a metrics dict (also written to results/<run_name>/move_metrics.json by the CLI):
        n_moves             plies in the PGN
        move_accuracy       fraction of PGN plies recognized correctly, aligned in order
                            (a missed or wrong move counts as an error; after an error the
                            tracker is resynced to the true position so one mistake does
                            not cascade - report the unsynced "game completed" flag too)
        missed              PGN plies with no detected move
        spurious            detected moves with no corresponding ply
        ambiguous           detections flagged ambiguous
        special_moves       accuracy restricted to castling / en passant / promotion plies
        game_completed      True if the tracker reproduced the full game without resync

    TODO(M5): load PGN with chess.pgn.read_game, load frames (sorted by name), build the
    tracker, align detected moves with mainline moves, compute the numbers above.
    """
    raise NotImplementedError("evaluate_on_game: see docstring TODO(M5)")
