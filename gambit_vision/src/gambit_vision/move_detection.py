"""M5 (optional): move detection from a stream of warped top-down board images.

Pipeline per move:
    1. StableFrameDetector watches frame-to-frame differences (on blurred, downsampled
       grayscale frames) and fires once the board is still again after a period of motion
       (the player's hand came in and left).
    2. The stable frame is cut into 64 crops and classified -> (64, 13) probabilities.
    3. infer_move() scores every legal move of the previous position (plus "no move")
       against those probabilities and returns the most likely one. Because candidates
       come from python-chess, castling, en passant and promotion need no special cases.
       Optional extensions: unknown side to move (try both colours) and two-ply candidates
       (a stable frame between two plies was missed).
    4. MoveTracker glues 1-3 together and keeps the game board; evaluate_on_game() replays
       a recorded game against its PGN and reports move-level metrics.

simulate_game_frames() produces synthetic frames + a fake classifier for a PGN, so the
whole tracker can be tested without a camera (see tests/test_move_detection.py).

Thresholds are constructor parameters with defaults chosen on synthetic data; they must be
tuned on real footage (see docs/M5_TODO.md). This module must import without torch; cv2 is
imported lazily (only for reading images/videos and the optional skin mask).

Image orientation convention (top-down warped image): a8 is the top-left square, h1 the
bottom-right one (white at the bottom, like a printed diagram), with `margin_px` pixels of
border around the 8 x 8 squares. Pass white_at_bottom=False if the warp is rotated 180 deg.
"""

from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Union

import chess
import chess.pgn
import numpy as np

from gambit_vision.labels import CLASS_TO_INDEX, NUM_CLASSES, SQUARE_NAMES, board_to_labels

if TYPE_CHECKING:  # board_assembly is M4; import lazily at runtime
    from gambit_vision.board_assembly import BoardPrediction  # noqa: F401


# --------------------------------------------------------------------------- preprocessing

def _to_gray(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    return f.mean(axis=2) if f.ndim == 3 else f


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur in plain numpy (edge-replicated borders)."""
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(img, pad, mode="edge")
        n = img.shape[axis]
        out = np.zeros_like(img)
        for k, w in enumerate(kernel):  # sum of shifted copies = 1-D convolution
            sl = [slice(None), slice(None)]
            sl[axis] = slice(k, k + n)
            out += w * padded[tuple(sl)]
        img = out
    return img


def preprocess_frame(frame: np.ndarray, downsample: int = 1, blur_sigma: float = 0.0) -> np.ndarray:
    """Grayscale -> block-average downsample by `downsample` -> Gaussian blur (in output pixels).

    Both steps suppress sensor noise and flicker so that only real changes (a hand, a moved
    piece) produce large differences; downsampling also makes everything after it cheaper.
    """
    f = _to_gray(frame)
    d = int(downsample)
    if d > 1:
        h, w = (f.shape[0] // d) * d, (f.shape[1] // d) * d
        f = f[:h, :w].reshape(h // d, d, w // d, d).mean(axis=(1, 3))
    if blur_sigma > 0:
        f = _gaussian_blur(f, blur_sigma)
    return f


def frame_difference(a: np.ndarray, b: np.ndarray, *, downsample: int = 1,
                     blur_sigma: float = 0.0) -> float:
    """Mean absolute difference between two frames, in [0, 255] intensity units.

    Color frames are averaged to grayscale first; optionally downsampled and blurred
    (see preprocess_frame). With the defaults this is the raw per-pixel mean difference.
    """
    if np.shape(a) != np.shape(b):
        raise ValueError(f"frame shapes differ: {np.shape(a)} vs {np.shape(b)}")
    pa = preprocess_frame(a, downsample, blur_sigma)
    pb = preprocess_frame(b, downsample, blur_sigma)
    return float(np.abs(pa - pb).mean())


def _per_square_mean(img: np.ndarray, square_px: float, margin_px: float = 0.0,
                     white_at_bottom: bool = True) -> np.ndarray:
    """Mean of a (H, W) image over each board square -> (64,) in square-index order (a1=0).

    square_px / margin_px are in the pixel units of `img` (may be fractional after
    downsampling; square borders are rounded to whole pixels).
    """
    edges = np.round(margin_px + square_px * np.arange(9)).astype(int)
    if edges[-1] > min(img.shape[:2]):
        raise ValueError(f"image {img.shape} too small for 8 squares of {square_px}px "
                         f"+ margin {margin_px}px")
    out = np.zeros(64, dtype=np.float64)
    for r in range(8):          # image row, top to bottom
        for c in range(8):      # image column, left to right
            cell = img[edges[r]:edges[r + 1], edges[c]:edges[c + 1]]
            if white_at_bottom:
                rank, file = 7 - r, c
            else:
                rank, file = r, 7 - c
            out[chess.square(file, rank)] = cell.mean()
    return out


def square_differences(a: np.ndarray, b: np.ndarray, square_px: int, margin_px: int = 0, *,
                       downsample: int = 1, blur_sigma: float = 0.0,
                       white_at_bottom: bool = True) -> np.ndarray:
    """Per-square mean absolute difference between two warped top-down frames.

    `square_px` / `margin_px` describe the full-resolution warped image (configs/camera.yaml
    `warp:`). Returns a (64,) array in python-chess square order (a1 first).
    """
    if np.shape(a) != np.shape(b):
        raise ValueError(f"frame shapes differ: {np.shape(a)} vs {np.shape(b)}")
    d = max(1, int(downsample))
    diff = np.abs(preprocess_frame(a, d, blur_sigma) - preprocess_frame(b, d, blur_sigma))
    return _per_square_mean(diff, square_px / d, margin_px / d, white_at_bottom)


def changed_squares_between(a: np.ndarray, b: np.ndarray, square_px: int, margin_px: int = 0,
                            threshold: float = 10.0, **kwargs) -> list[str]:
    """Names of squares whose mean absolute difference exceeds `threshold` (a1 order)."""
    d = square_differences(a, b, square_px, margin_px, **kwargs)
    return [SQUARE_NAMES[i] for i in np.flatnonzero(d > threshold)]


def skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of "skin-ish" pixels using a fixed HSV box (OpenCV H in 0..180).

    Deliberately crude: H in [0, 25] or [165, 180], S in [40, 200], V >= 60. Wooden boards
    and brown pieces also fall in this range, which is why StableFrameDetector only counts
    skin pixels that are NEW compared with the reference (pre-motion) frame.
    Tune the box on real footage (lighting and skin tones vary).
    """
    import cv2  # lazy

    img = np.asarray(frame_bgr)
    if img.ndim != 3:
        return np.zeros(img.shape[:2], dtype=bool)
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 25) | (h >= 165)) & (s >= 40) & (s <= 200) & (v >= 60)


# --------------------------------------------------------------------------- stable frames

class State(str, Enum):
    IDLE = "idle"          # board static, nothing happening
    MOTION = "motion"      # large frame differences (hand over the board)
    SETTLING = "settling"  # differences small again, waiting for settle_frames in a row
    STABLE = "stable"      # emitted for exactly one frame after settling; then back to IDLE


class StableFrameDetector:
    """State machine IDLE -> MOTION -> SETTLING -> STABLE -> IDLE.

    Feed frames with update(frame); it returns True exactly once per motion episode, on the
    frame where the board has been still for `settle_frames` consecutive frames after at
    least `min_motion_frames` frames of motion. `stable_frame` / `reference_frame` hold the
    frames after and before the episode.

    Motion statistic: frames are converted to gray, block-downsampled by `downsample` and
    blurred (`blur_sigma`, in downsampled pixels). Then
      - without board geometry: mean absolute difference over the whole image;
      - with `square_px` (warped image geometry from configs/camera.yaml): the MAX over the
        64 per-square mean differences, so a small hand at the board edge is not averaged
        away. `last_square_diff` holds the per-square values.

    Hand-presence heuristic (checked only at the moment the detector would fire, comparing
    the candidate stable frame with the reference frame from before the motion):
      - changed-area test: if more than `max_changed_fraction` of the (downsampled) pixels
        differ by more than `pixel_change_threshold`, something big is still on the board.
        A legal move changes at most 4 squares (castling) = 6% of the board, a hand/arm
        resting on it usually much more.
      - optional skin test (`skin_check=True`, BGR colour frames only): more than
        `skin_fraction` of the pixels are skin-coloured now but were not in the reference.
    If a hand is detected the detector keeps waiting; after `hand_timeout_frames` still
    frames it fires anyway (a permanent lighting change must not block it forever).

    Defaults were chosen on synthetic frames: tune diff_threshold (between the noise floor
    and hand motion; plot `last_diff` over a recorded game) and settle_frames (~0.3-0.5 s).
    """

    def __init__(self, diff_threshold: float = 4.0, settle_frames: int = 10,
                 min_motion_frames: int = 2, *, downsample: int = 4, blur_sigma: float = 1.0,
                 square_px: int | None = None, margin_px: int = 0,
                 white_at_bottom: bool = True, square_change_threshold: float = 10.0,
                 hand_check: bool = True, pixel_change_threshold: float = 25.0,
                 max_changed_fraction: float = 0.2, skin_check: bool = False,
                 skin_fraction: float = 0.02, hand_timeout_frames: int | None = None) -> None:
        self.diff_threshold = diff_threshold
        self.settle_frames = settle_frames
        self.min_motion_frames = min_motion_frames
        self.downsample = max(1, int(downsample))
        self.blur_sigma = blur_sigma
        self.square_px = square_px
        self.margin_px = margin_px
        self.white_at_bottom = white_at_bottom
        self.square_change_threshold = square_change_threshold
        self.hand_check = hand_check
        self.pixel_change_threshold = pixel_change_threshold
        self.max_changed_fraction = max_changed_fraction
        self.skin_check = skin_check
        self.skin_fraction = skin_fraction
        self.hand_timeout_frames = (hand_timeout_frames if hand_timeout_frames is not None
                                    else 4 * settle_frames)
        self.reset()

    def reset(self) -> None:
        self.state = State.IDLE
        self.prev_frame: np.ndarray | None = None
        self.reference_frame: np.ndarray | None = None  # last still frame before motion
        self.stable_frame: np.ndarray | None = None     # frame at which STABLE fired
        self.last_diff = 0.0
        self.last_square_diff: np.ndarray | None = None
        self.hand_detected = False                      # last hand check result
        self._prev_p: np.ndarray | None = None          # preprocessed prev_frame
        self._ref_p: np.ndarray | None = None           # preprocessed reference_frame
        self._stable_p: np.ndarray | None = None        # preprocessed stable_frame
        self._event_ref_p: np.ndarray | None = None     # preprocessed reference at that event
        self._motion_count = 0
        self._still_count = 0

    # -- helpers
    def _prep(self, frame: np.ndarray) -> np.ndarray:
        return preprocess_frame(frame, self.downsample, self.blur_sigma)

    def _square_means(self, diff: np.ndarray) -> np.ndarray:
        d = self.downsample
        return _per_square_mean(diff, self.square_px / d, self.margin_px / d,
                                self.white_at_bottom)

    def _motion_statistic(self, p_prev: np.ndarray, p_cur: np.ndarray) -> float:
        diff = np.abs(p_cur - p_prev)
        if self.square_px:
            self.last_square_diff = self._square_means(diff)
            return float(self.last_square_diff.max())
        return float(diff.mean())

    def hand_present(self, frame: np.ndarray, frame_p: np.ndarray | None = None) -> bool:
        """Heuristic: is something large (a hand) on the board compared with the reference?"""
        if self.reference_frame is None:
            return False
        frame_p = self._prep(frame) if frame_p is None else frame_p
        ref_p = self._ref_p if self._ref_p is not None else self._prep(self.reference_frame)
        changed = np.abs(frame_p - ref_p) > self.pixel_change_threshold
        if changed.mean() > self.max_changed_fraction:
            return True
        if self.skin_check and np.ndim(frame) == 3:
            new_skin = skin_mask(frame) & ~skin_mask(self.reference_frame)
            if new_skin.mean() > self.skin_fraction:
                return True
        return False

    def changed_squares(self, threshold: float | None = None) -> list[str]:
        """Squares that differ before/after the last stable event (needs square_px).

        Compares stable_frame with the reference frame as it was when the event fired
        (reference_frame itself moves on while the board stays idle).

        A cheap cross-check for infer_move: a normal move changes 2 squares, castling and
        en passant 3-4. Many changed squares hint at a bumped board or a lighting change.
        """
        if self.square_px is None:
            raise ValueError("changed_squares needs square_px (warped board geometry)")
        if self._stable_p is None or self._event_ref_p is None:
            return []
        thr = self.square_change_threshold if threshold is None else threshold
        d = self._square_means(np.abs(self._stable_p - self._event_ref_p))
        return [SQUARE_NAMES[i] for i in np.flatnonzero(d > thr)]

    # -- main entry point
    def update(self, frame: np.ndarray) -> bool:
        p = self._prep(frame)
        if self.prev_frame is None:
            self.prev_frame = self.reference_frame = frame
            self._prev_p = self._ref_p = p
            return False
        if p.shape != self._prev_p.shape:
            raise ValueError(f"frame shapes differ: {np.shape(self.prev_frame)} vs {np.shape(frame)}")
        diff = self._motion_statistic(self._prev_p, p)
        self.last_diff = diff
        moving = diff > self.diff_threshold
        fired = False

        if self.state in (State.IDLE, State.STABLE):
            if moving:
                self.state, self._motion_count = State.MOTION, 1
            else:
                self.state = State.IDLE
                self.reference_frame, self._ref_p = frame, p
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
            self.hand_detected = self.hand_check and self.hand_present(frame, p)
            if not self.hand_detected or self._still_count >= self.hand_timeout_frames:
                self.state = State.STABLE
                self.stable_frame, self._stable_p = frame, p
                self._event_ref_p = self._ref_p
                fired = True

        self.prev_frame, self._prev_p = frame, p
        return fired


# --------------------------------------------------------------------------- move inference

Candidate = Union[chess.Move, None, tuple]  # single move, "no move", or (move1, move2)


@dataclass
class MoveResult:
    move: chess.Move | None        # None = board judged unchanged; first ply for two-ply
    san: str | None                # SAN in the previous position ("e4 e5" for two plies)
    score: float                   # log-likelihood of the best candidate (minus penalties)
    runner_up: Candidate           # second-best candidate (None may mean "no move")
    margin: float                  # score(best) - score(runner_up), >= 0
    ambiguous: bool                # margin below threshold -> ask for confirmation / wait
    ranking: list[tuple[Candidate, float]] = field(default_factory=list, repr=False)
    moves: list[chess.Move] = field(default_factory=list)  # all plies (0, 1 or 2)
    sans: list[str] = field(default_factory=list)
    turn: chess.Color | None = None   # colour that made the (first) move; None for no move
    turn_flipped: bool = False        # True if the move was found with the turn flipped
    two_ply_searched: bool = False    # two-ply candidates were scored
    deficit: float = 0.0              # upper bound - best single-ply score (>= 0), in nats


def placement_log_likelihood(labels: Sequence[str], log_probs: np.ndarray) -> float:
    """Sum over squares of log P(square has this label)."""
    idx = np.fromiter((CLASS_TO_INDEX[x] for x in labels), dtype=np.int64, count=64)
    return float(log_probs[np.arange(64), idx].sum())


def _flip_turn(board: chess.Board) -> chess.Board:
    b = board.copy(stack=False)
    b.turn = not b.turn
    b.ep_square = None  # an en passant right only exists for the side that was to move
    return b


def infer_move(prev_board: chess.Board, square_probs_after: np.ndarray, *,
               include_null: bool = True, ambiguity_margin: float = 2.0,
               eps: float = 1e-6, unknown_turn: bool = False,
               two_ply_threshold: float | None = None,
               two_ply_penalty: float = 1.0) -> MoveResult:
    """Most likely legal move from prev_board given (64, 13) probabilities of the new frame.

    Each candidate (every legal move, plus "no move" if include_null) is pushed onto a copy
    of prev_board and scored by the log-likelihood of the resulting piece placement, treating
    squares as independent. `ambiguity_margin` is in nats: 2.0 means the best candidate is
    ~7x more likely than the runner-up.

    unknown_turn: the side to move is not trusted (tracking started mid-game). Legal moves
        of BOTH colours are scored (the turn is flipped on a copy, en passant cleared) and
        the result reports which colour moved in `turn` / `turn_flipped`.
    two_ply_threshold: if not None, and the best single-ply candidate explains the frame
        poorly - its score is more than this many nats below the upper bound
        sum_sq max_c log p (the score of the per-square argmax placement) - then all legal
        move PAIRS are scored too (a stable frame between two plies was missed). Pairs pay
        `two_ply_penalty` nats (prior: missing a stable frame is rarer than not).
        Ranking entries for pairs use a (move1, move2) tuple as the candidate.
    """
    probs = np.asarray(square_probs_after, dtype=np.float64)
    if probs.shape != (64, NUM_CLASSES):
        raise ValueError(f"expected (64, {NUM_CLASSES}) probabilities, got {probs.shape}")
    log_probs = np.log(np.clip(probs, eps, 1.0))

    starts = [prev_board.copy(stack=False)]
    if unknown_turn:
        flipped = _flip_turn(prev_board)
        if flipped.is_valid():  # e.g. not valid if the side NOT to move would be in check
            starts.append(flipped)

    # each entry: (candidate key, list of moves, start board, score)
    cands: list[tuple[Candidate, list[chess.Move], chess.Board, float]] = []
    if include_null:
        cands.append((None, [], starts[0],
                      placement_log_likelihood(board_to_labels(prev_board), log_probs)))
    for start in starts:
        board = start.copy(stack=False)
        for move in start.legal_moves:
            board.push(move)
            cands.append((move, [move], start,
                          placement_log_likelihood(board_to_labels(board), log_probs)))
            board.pop()

    upper = float(log_probs.max(axis=1).sum())
    best_single = max((c[3] for c in cands), default=float("-inf"))
    deficit = upper - best_single
    two_ply = two_ply_threshold is not None and deficit > two_ply_threshold
    if two_ply:
        for start in starts:
            board = start.copy(stack=False)
            for m1 in start.legal_moves:
                board.push(m1)
                for m2 in board.legal_moves:
                    board.push(m2)
                    score = placement_log_likelihood(board_to_labels(board), log_probs)
                    cands.append(((m1, m2), [m1, m2], start, score - two_ply_penalty))
                    board.pop()
                board.pop()

    if not cands:  # game over and include_null=False
        return MoveResult(None, None, float("-inf"), None, 0.0, True, deficit=deficit)

    cands.sort(key=lambda t: t[3], reverse=True)  # stable: "no move" wins exact ties
    key, moves, start, best_score = cands[0]
    runner_up, second = (cands[1][0], cands[1][3]) if len(cands) > 1 else (None, float("-inf"))
    margin = best_score - second

    sans: list[str] = []
    b = start.copy(stack=False)
    for m in moves:
        sans.append(b.san(m))
        b.push(m)
    return MoveResult(
        move=moves[0] if moves else None,
        san=" ".join(sans) if sans else None,
        score=best_score,
        runner_up=runner_up,
        margin=margin,
        ambiguous=margin < ambiguity_margin,
        ranking=[(c[0], c[3]) for c in cands],
        moves=list(moves),
        sans=sans,
        turn=start.turn if moves else None,
        turn_flipped=bool(moves) and start.turn != prev_board.turn,
        two_ply_searched=two_ply,
        deficit=deficit,
    )


def changed_squares(prev_labels: Sequence[str], new_labels: Sequence[str]) -> list[str]:
    """Names of squares whose label differs, in square-index order (a1 first)."""
    if len(prev_labels) != 64 or len(new_labels) != 64:
        raise ValueError("expected 64 labels each")
    return [SQUARE_NAMES[i] for i, (a, b) in enumerate(zip(prev_labels, new_labels)) if a != b]


# --------------------------------------------------------------------------- frame sources

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _natural_key(path: Path) -> list:
    """Sort key so that frame_2.png comes before frame_10.png."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def iter_frames(source: str | Path | Iterable[np.ndarray]) -> Iterator[np.ndarray]:
    """Yield frames from a directory of images (natural sort order = time order), a video
    file (cv2.VideoCapture) or any iterable of arrays."""
    if not isinstance(source, (str, Path)):
        yield from source
        return
    import cv2  # lazy

    path = Path(source)
    if path.is_dir():
        files = sorted((p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS),
                       key=_natural_key)
        if not files:
            raise FileNotFoundError(f"no image frames in {path}")
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise OSError(f"could not read {p}")
            yield img
    elif path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"could not open video {path}")
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                yield img
        finally:
            cap.release()
    else:
        raise FileNotFoundError(path)


# --------------------------------------------------------------------------- tracking

ProbaFn = Callable[[np.ndarray], np.ndarray]  # warped frame -> (64, 13) probabilities
AskFn = Callable[[chess.Board, MoveResult], "chess.Move | None"]


@dataclass
class MoveEvent:
    frame_index: int
    result: MoveResult
    fen_after: str
    kind: str = "move"             # "move" (accepted), "ambiguous" or "no_move"
    probs: np.ndarray | None = field(default=None, repr=False)


class MoveTracker:
    """Detector + classifier + infer_move over a frame stream, maintaining the game board.

    `board_proba` maps a warped top-down frame to (64, 13) probabilities; build it from
    capture.py's crop function + an M3 model's predict_proba.

    On every stable frame:
      - best candidate is "no move" (hand passed over, a piece was adjusted) -> recorded in
        `no_move_events`, board unchanged;
      - result ambiguous -> `on_ambiguous(board, result)` is asked if given (e.g. the robot
        UI asks the player); it may return the confirmed legal move. Otherwise the event is
        recorded in `ambiguous_events` and the board is kept: the next stable frame is
        compared with the same board (two-ply search can then catch up);
      - else the move(s) are pushed and a MoveEvent is appended to `events` and returned.
    `history` holds every stable-frame event of any kind, in order.

    unknown_turn: do not trust whose turn it is until the first accepted move (then the
    turn is known). two_ply_threshold / two_ply_penalty: see infer_move; None disables.
    """

    def __init__(self, board_proba: ProbaFn, start_board: chess.Board | None = None,
                 detector: StableFrameDetector | None = None,
                 ambiguity_margin: float = 2.0, *, unknown_turn: bool = False,
                 two_ply_threshold: float | None = 4.0, two_ply_penalty: float = 1.0,
                 on_ambiguous: AskFn | None = None) -> None:
        self.board_proba = board_proba
        self.board = start_board.copy() if start_board is not None else chess.Board()
        self.detector = detector or StableFrameDetector()
        self.ambiguity_margin = ambiguity_margin
        self.unknown_turn = unknown_turn
        self.two_ply_threshold = two_ply_threshold
        self.two_ply_penalty = two_ply_penalty
        self.on_ambiguous = on_ambiguous
        self.events: list[MoveEvent] = []
        self.ambiguous_events: list[MoveEvent] = []
        self.no_move_events: list[MoveEvent] = []
        self.history: list[MoveEvent] = []
        self._frame_index = 0

    def _ask(self, result: MoveResult) -> MoveResult | None:
        """Let the caller confirm an ambiguous result; returns an accepted result or None."""
        if self.on_ambiguous is None:
            return None
        move = self.on_ambiguous(self.board.copy(), result)
        if move is None:
            return None
        board = self.board
        if move not in board.legal_moves and self.unknown_turn:
            board = _flip_turn(self.board)
        if move not in board.legal_moves:
            raise ValueError(f"on_ambiguous returned an illegal move {move}")
        san = board.san(move)
        return MoveResult(move=move, san=san, score=result.score, runner_up=result.move,
                          margin=float("inf"), ambiguous=False, ranking=result.ranking,
                          moves=[move], sans=[san], turn=board.turn,
                          turn_flipped=board.turn != self.board.turn)

    def _apply(self, result: MoveResult) -> None:
        if result.turn is not None and result.turn != self.board.turn:
            # the side to move was wrong (unknown_turn): flip before pushing
            self.board.turn = result.turn
            self.board.ep_square = None
        for move in result.moves:
            self.board.push(move)
        self.unknown_turn = False  # after one real move the turn is known

    def _record(self, kind: str, result: MoveResult, probs: np.ndarray) -> MoveEvent:
        event = MoveEvent(self._frame_index, result, self.board.fen(), kind, probs)
        {"move": self.events, "ambiguous": self.ambiguous_events,
         "no_move": self.no_move_events}[kind].append(event)
        self.history.append(event)
        return event

    def process_frame(self, frame: np.ndarray) -> MoveEvent | None:
        """Feed one warped frame; return a MoveEvent when a move was recognized."""
        self._frame_index += 1
        if not self.detector.update(frame):
            return None
        probs = np.asarray(self.board_proba(frame), dtype=np.float64)
        result = infer_move(self.board, probs, ambiguity_margin=self.ambiguity_margin,
                            unknown_turn=self.unknown_turn,
                            two_ply_threshold=self.two_ply_threshold,
                            two_ply_penalty=self.two_ply_penalty)
        if result.ambiguous:
            confirmed = self._ask(result)
            if confirmed is None:  # keep the board, wait for the next stable frame
                self._record("ambiguous", result, probs)
                return None
            result = confirmed
        if not result.moves:  # spurious motion: hand hovered, piece adjusted
            self._record("no_move", result, probs)
            return None
        self._apply(result)
        return self._record("move", result, probs)

    def run(self, frames: str | Path | Iterable[np.ndarray],
            on_event: Callable[[MoveEvent], None] | None = None) -> list[MoveEvent]:
        """Process a whole stream: a directory of frames, a video file or an iterable of
        (already warped) frames. `on_event` is called for every accepted move."""
        for frame in iter_frames(frames):
            event = self.process_frame(frame)
            if event is not None and on_event is not None:
                on_event(event)
        return self.events


# --------------------------------------------------------------------------- evaluation

def _load_game(pgn: str | Path | chess.pgn.Game) -> chess.pgn.Game:
    """Accept a chess.pgn.Game, a path to a .pgn file, or PGN text."""
    if isinstance(pgn, chess.pgn.Game):
        return pgn
    text = str(pgn)
    is_file = False
    try:
        is_file = isinstance(pgn, Path) or ("\n" not in text and Path(text).is_file())
    except OSError:  # e.g. PGN text too long to be a file name
        pass
    if is_file:
        text = Path(pgn).read_text()
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise ValueError(f"no game found in {pgn!r}")
    if game.errors:
        raise ValueError(f"PGN errors: {game.errors}")
    return game


def _special_kind(board: chess.Board, move: chess.Move) -> str | None:
    if board.is_castling(move):
        return "castling"
    if board.is_en_passant(move):
        return "en_passant"
    if move.promotion is not None:
        return "promotion"
    return None


def evaluate_on_game(pgn_path: str | Path | chess.pgn.Game,
                     frames_dir: str | Path | Iterable[np.ndarray],
                     board_proba: ProbaFn | None = None, *,
                     out_dir: str | Path | None = None,
                     frame_plies: Sequence[int] | None = None,
                     lookahead: int = 4, **tracker_kwargs) -> dict:
    """Replay one recorded game through MoveTracker and score it against its PGN.

    Inputs: the game's PGN (ground truth; path, PGN text or chess.pgn.Game), the frames (a
    directory of warped frames whose names sort in time order, a video file, or an iterable
    of arrays) and `board_proba` (frame -> (64, 13) probabilities; for a raw video it must
    also do the warp). `tracker_kwargs` go to MoveTracker (detector, two_ply_threshold, ...).

    Ground truth for each stable frame = how many plies of the PGN had been played when it
    was taken. If `frame_plies` (true ply count per frame, e.g. manifest `move_idx`) is
    given it is used; otherwise it is estimated as the PGN position among the next
    `lookahead` plies that best matches the frame's probabilities (fine when the
    classifier is decent; prefer frame_plies when available).

    Scoring (tracker at ply k, the stable frame shows ply j):
      - accepted detection equal to PGN plies k, k+1, ... -> those plies are correct;
      - wrong detection -> one error (+ missed for further plies it skipped), or spurious
        if nothing had happened (j <= k); the tracker is then RESYNCED to the true position
        so one mistake does not cascade (game_completed becomes False);
      - no detection although j > k -> the tracker may still catch up (ambiguous frame,
        then two-ply search); if it falls further behind than it can recover, the skipped
        plies count as missed and it is resynced. Plies never reached by the end are missed.

    Returns (and writes to out_dir/move_metrics.json when out_dir is given):
        n_moves, correct, move_accuracy (= correct / n_moves), wrong, missed, spurious,
        ambiguous (stable frames flagged ambiguous), resyncs,
        special_moves {castling|en_passant|promotion: {total, correct, accuracy}},
        game_completed (full game reproduced without any resync), events (per stable frame).
    """
    if board_proba is None:
        raise ValueError("evaluate_on_game needs board_proba (frame -> (64, 13) probs): "
                         "warp + crop the frame and run an M3 model's predict_proba")
    game = _load_game(pgn_path)
    start = game.board()
    true_moves = list(game.mainline_moves())
    n = len(true_moves)
    boards = [start.copy()]
    kinds: list[str | None] = []
    for m in true_moves:
        kinds.append(_special_kind(boards[-1], m))
        b = boards[-1].copy()
        b.push(m)
        boards.append(b)

    tracker = MoveTracker(board_proba, start_board=start, **tracker_kwargs)
    max_lag = 1 if tracker.two_ply_threshold is not None else 0  # how far it can catch up
    k = 0  # number of true plies the tracker's board is at
    correct = wrong = missed = spurious = n_ambiguous = resyncs = 0
    ply_correct = [False] * n
    event_log: list[dict] = []

    def resync(j: int) -> None:
        nonlocal k, resyncs
        tracker.board = boards[j].copy()
        tracker.unknown_turn = False
        k = j
        resyncs += 1

    for frame in iter_frames(frames_dir):
        n_hist = len(tracker.history)
        tracker.process_frame(frame)
        if len(tracker.history) == n_hist:
            continue  # not a stable frame
        ev = tracker.history[-1]
        if frame_plies is not None:
            truth = int(frame_plies[ev.frame_index - 1])
        else:
            log_probs = np.log(np.clip(ev.probs, 1e-6, 1.0))
            truth = max(range(k, min(n, k + lookahead) + 1), key=lambda j: (
                placement_log_likelihood(board_to_labels(boards[j]), log_probs), -j))
        truth = min(max(truth, k), n)
        k_before = k
        detected = ev.result.moves if ev.kind == "move" else []
        n_ambiguous += ev.kind == "ambiguous"

        if not detected:
            if truth - k > max_lag:
                missed += truth - k
                resync(truth)
        else:
            p = 0  # length of the correct prefix of the detection
            while p < len(detected) and k + p < n and detected[p] == true_moves[k + p]:
                ply_correct[k + p] = True
                p += 1
            correct += p
            if p == len(detected):
                k += p
                if truth - k > max_lag:
                    missed += truth - k
                    resync(truth)
            else:
                if truth <= k + p:
                    spurious += 1
                else:
                    wrong += 1
                    missed += truth - (k + p) - 1
                resync(max(truth, k + p))
        event_log.append({
            "frame_index": ev.frame_index, "kind": ev.kind, "detected": ev.result.san,
            "margin": None if math.isinf(ev.result.margin) else round(ev.result.margin, 3),
            "true_ply_before": k_before, "true_ply_frame": truth,
        })
    missed += n - k

    special: dict[str, dict] = {}
    for kind in ("castling", "en_passant", "promotion"):
        idx = [i for i, kd in enumerate(kinds) if kd == kind]
        c = sum(ply_correct[i] for i in idx)
        special[kind] = {"total": len(idx), "correct": c,
                         "accuracy": (c / len(idx)) if idx else None}

    metrics = {
        "n_moves": n,
        "correct": correct,
        "move_accuracy": correct / n if n else 1.0,
        "wrong": wrong,
        "missed": missed,
        "spurious": spurious,
        "ambiguous": n_ambiguous,
        "resyncs": resyncs,
        "special_moves": special,
        "game_completed": resyncs == 0 and correct == n,
        "n_stable_events": len(tracker.history),
        "events": event_log,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "move_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# --------------------------------------------------------------------------- simulation

# Synthetic "camera": each square is drawn as a flat patch whose intensity encodes its
# label, so a fake classifier can read the position back from any still frame.
_SIM_BASE, _SIM_STEP = 30.0, 15.0


def render_board(board: chess.Board, square_px: int = 8) -> np.ndarray:
    """Synthetic top-down image (a8 top-left): square intensity = 30 + 15 * class index."""
    img = np.zeros((8 * square_px, 8 * square_px), dtype=np.float32)
    for sq, lab in enumerate(board_to_labels(board)):
        r, c = 7 - chess.square_rank(sq), chess.square_file(sq)
        img[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = \
            _SIM_BASE + _SIM_STEP * CLASS_TO_INDEX[lab]
    return img


@dataclass
class SimulatedGame:
    frames: list[np.ndarray]
    board_proba: ProbaFn            # frame -> noisy one-hot probabilities of what it shows
    frame_plies: list[int]          # true number of plies played at each frame
    boards: list[chess.Board]       # true position after 0..n plies
    square_px: int


def simulate_game_frames(pgn: str | Path | chess.pgn.Game, noise: float = 0.1, *,
                         seed: int = 0, square_px: int = 8, still_frames: int = 15,
                         motion_frames: int = 6, sensor_noise: float = 1.0,
                         hand_fraction: float = 0.4,
                         skip_pause_after: Iterable[int] = (),
                         spurious_motion_after: Iterable[int] = ()) -> SimulatedGame:
    """Synthetic frame stream + fake classifier for a PGN, to test the tracker without a camera.

    Stream: still frames of the start position, then for every ply `motion_frames` "hand"
    frames (a random-noise blob covering `hand_fraction` of the board at a random place;
    the position under it switches half way) followed by `still_frames` still frames of the
    new position. Still frames carry Gaussian sensor noise (`sensor_noise`, intensity units).

    skip_pause_after: ply numbers (1 = after the first move) with NO still pause, so the
        next ply's motion follows immediately -> a missed stable frame.
    spurious_motion_after: ply numbers (0 = before the first move) after which an extra
        motion episode happens without any move (hand hovering) -> should give "no move".

    board_proba(frame) reads each square's mean intensity back to a label (a perfect
    "classifier") and returns (1 - noise) * one_hot + noise * Dirichlet(1) per square.
    """
    game = _load_game(pgn)
    rng = np.random.default_rng(seed)
    skip, hover = set(skip_pause_after), set(spurious_motion_after)
    boards = [game.board()]
    for m in game.mainline_moves():
        b = boards[-1].copy()
        b.push(m)
        boards.append(b)
    size = 8 * square_px
    frames: list[np.ndarray] = []
    plies: list[int] = []

    def still(ply: int, count: int) -> None:
        img = render_board(boards[ply], square_px)
        for _ in range(count):
            frames.append(img + rng.normal(0, sensor_noise, img.shape).astype(np.float32))
            plies.append(ply)

    def motion(before: int, after: int) -> None:
        side = int(round(size * math.sqrt(hand_fraction)))
        for i in range(motion_frames):
            ply = before if i < motion_frames // 2 else after
            img = render_board(boards[ply], square_px)
            y, x = rng.integers(0, size - side + 1, size=2)
            img[y:y + side, x:x + side] = rng.uniform(0, 255, (side, side))
            frames.append(img)
            plies.append(ply)

    still(0, still_frames)
    for ply in range(len(boards)):
        if ply in hover:
            motion(ply, ply)
            still(ply, still_frames)
        if ply + 1 < len(boards):
            motion(ply, ply + 1)
            if ply + 1 not in skip:
                still(ply + 1, still_frames)

    def board_proba(frame: np.ndarray) -> np.ndarray:
        means = _per_square_mean(_to_gray(frame), square_px)
        cls = np.clip(np.round((means - _SIM_BASE) / _SIM_STEP), 0, NUM_CLASSES - 1).astype(int)
        probs = noise * rng.dirichlet(np.ones(NUM_CLASSES), size=64)
        probs[np.arange(64), cls] += 1.0 - noise
        return probs

    return SimulatedGame(frames, board_proba, plies, boards, square_px)
