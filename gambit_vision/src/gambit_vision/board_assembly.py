"""M4: combine 64 square predictions into a board state.

Input: a (64, 13) probability array from any square classifier (rows in python-chess
square order a1..h8, columns in labels.CLASSES order). See docs/contracts.md.
Output: a BoardPrediction (FEN, per-square labels/confidence, low-confidence squares,
legality verdict).

Pipeline:
    probs --argmax--> labels --check_legality--> issues
    (optional) illegal? --repair--> most plausible legal labels nearby
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from gambit_vision.labels import CLASS_TO_INDEX, CLASSES, NUM_CLASSES, SQUARE_NAMES, labels_to_board

# Hard caps per piece letter (8 pawns can promote, so e.g. at most 1 + 8 = 9 queens).
MAX_PIECES = {"K": 1, "Q": 9, "R": 10, "B": 10, "N": 10, "P": 8}
MAX_PER_COLOR = 16
# Pieces each side starts with. Anything above this must come from a promotion, and every
# promotion uses up one pawn: sum(extra pieces) <= 8 - pawns. Tighter than the fixed caps.
START_COUNTS = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 2, chess.KNIGHT: 2}

COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}
KING_CLASS = {chess.WHITE: CLASS_TO_INDEX["K"], chess.BLACK: CLASS_TO_INDEX["k"]}
PAWN_CLASSES = [CLASS_TO_INDEX["P"], CLASS_TO_INDEX["p"]]
BACK_RANK_SQUARES = [sq for sq in chess.SQUARES if chess.square_rank(sq) in (0, 7)]


@dataclass
class BoardPrediction:
    fen: str                       # piece placement only (board_fen), e.g. "rnbqkbnr/pppppppp/8/..."
    labels: list[str]              # 64 labels, square-index order
    confidence: list[float]        # 64 probabilities of the output label (= max-prob unless repaired)
    low_confidence: list[str] = field(default_factory=list)  # square names below threshold
    legal: bool = True
    issues: list[str] = field(default_factory=list)          # human-readable problems
    repaired: bool = False                                   # True if repair() changed labels
    changed_squares: list[str] = field(default_factory=list)  # squares changed by repair()


# --------------------------------------------------------------------------- legality
def _problems(board: chess.Board, turn: chess.Color | None = None) -> list[tuple[str, int]]:
    """All plausibility problems as (message, severity) pairs.

    Severity is roughly "how many squares must change to fix it". repair() minimizes the
    summed severity; check_legality() only reports the messages.
    """
    out: list[tuple[str, int]] = []
    kings = {}
    for color, name in COLOR_NAMES.items():
        n_kings = len(board.pieces(chess.KING, color))
        if n_kings != 1:
            out.append((f"{name} has {n_kings} kings (expected 1)", abs(n_kings - 1)))
        else:
            kings[color] = board.king(color)

        total = sum(len(board.pieces(pt, color)) for pt in chess.PIECE_TYPES)
        if total > MAX_PER_COLOR:
            out.append((f"{name} has {total} pieces (max {MAX_PER_COLOR})", total - MAX_PER_COLOR))

        for pt in chess.PIECE_TYPES:
            symbol = chess.piece_symbol(pt).upper()
            n = len(board.pieces(pt, color))
            if symbol != "K" and n > MAX_PIECES[symbol]:
                out.append((f"{name} has {n} {symbol} (max {MAX_PIECES[symbol]})", n - MAX_PIECES[symbol]))

        # Promotions: every piece beyond the starting set needs one missing pawn.
        n_pawns = len(board.pieces(chess.PAWN, color))
        extra = sum(max(0, len(board.pieces(pt, color)) - n0) for pt, n0 in START_COUNTS.items())
        missing_pawns = max(0, 8 - n_pawns)
        if extra > missing_pawns:
            out.append((f"{name} has {extra} promoted pieces but only {missing_pawns} missing pawns",
                        extra - missing_pawns))

    back_ranks = chess.BB_RANK_1 | chess.BB_RANK_8
    for sq in chess.SquareSet(board.pawns & back_ranks):
        out.append((f"pawn on back rank at {chess.square_name(sq)}", 1))

    # The king-safety checks need exactly one king per side.
    if len(kings) == 2:
        wk, bk = kings[chess.WHITE], kings[chess.BLACK]
        if chess.square_distance(wk, bk) <= 1:
            out.append((f"kings adjacent ({chess.square_name(wk)}, {chess.square_name(bk)})", 1))
        in_check = {
            chess.WHITE: board.is_attacked_by(chess.BLACK, wk),
            chess.BLACK: board.is_attacked_by(chess.WHITE, bk),
        }
        if turn is None:
            # board_fen has no side to move. "Side not to move is in check" can then only be
            # proven if BOTH kings are in check (whoever just moved left their king attacked).
            if in_check[chess.WHITE] and in_check[chess.BLACK]:
                out.append(("both kings in check", 1))
        elif in_check[not turn]:
            out.append((f"{COLOR_NAMES[not turn]} (not to move) is in check", 1))
    return out


def check_legality(board: chess.Board, turn: chess.Color | None = None) -> list[str]:
    """Plausibility checks on piece placement. Empty list = no problems found.

    turn: side to move if known (e.g. from the game record); None = unknown, then only
    turn-independent checks are applied.
    """
    return [msg for msg, _ in _problems(board, turn)]


def _score(labels_idx: np.ndarray, turn: chess.Color | None) -> int:
    board = labels_to_board([CLASSES[i] for i in labels_idx])
    return sum(sev for _, sev in _problems(board, turn))


# --------------------------------------------------------------------------- repair
def repair(
    probs: np.ndarray,
    issues: list[str] | None = None,
    max_steps: int = 8,
    turn: chess.Color | None = None,
) -> np.ndarray | None:
    """Constrained fixing of an illegal board.

    Works on a boolean mask `allowed` (64, 13): each step forbids some classes on some
    squares, and the label of a square is always its most probable *allowed* class.

      1. Kings: each color gets exactly one king. If there are several, keep the one with
         the highest king probability; if there is none, put it on the square with the
         highest king probability. All other squares are forbidden that king class
         (so they fall back to their best non-king class).
      2. Pawns can never stand on rank 1 or 8: forbid P/p there (-> best non-pawn class).
      3. Greedy: while problems remain (at most `max_steps` times), try switching every
         square to its next-best allowed class; among the switches that reduce the total
         violation score, apply the cheapest one (smallest drop in log-probability).

    Returns the masked + renormalized (64, 13) probs (argmax gives the repaired labels),
    or None if no legal board was reached. If `issues` is an empty list the board is
    taken as legal and a copy of probs is returned.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (64, NUM_CLASSES):
        raise ValueError(f"expected probs of shape (64, {NUM_CLASSES}), got {probs.shape}")
    if issues is not None and not issues:
        return probs.copy()

    logp = np.log(probs + 1e-12)
    allowed = np.ones_like(probs, dtype=bool)

    def current() -> np.ndarray:
        """Best allowed class per square."""
        return np.where(allowed, probs, -1.0).argmax(axis=1)

    # 1. exactly one king per color
    taken: set[int] = set()
    for color in (chess.WHITE, chess.BLACK):
        k = KING_CLASS[color]
        labels_idx = current()
        squares = [sq for sq in range(64) if labels_idx[sq] == k and sq not in taken]
        if len(squares) == 1:
            king_sq = squares[0]  # already fine, do not move it
        else:  # 0 or >1 kings: most king-like square not used by the other king
            king_sq = max((sq for sq in range(64) if sq not in taken), key=lambda sq: probs[sq, k])
        allowed[:, k] = False
        allowed[king_sq, :] = False
        allowed[king_sq, k] = True  # force the king onto king_sq
        taken.add(king_sq)

    # 2. no pawns on the back ranks
    for sq in BACK_RANK_SQUARES:
        allowed[sq, PAWN_CLASSES] = False

    # 3. greedy fix of whatever is left
    labels_idx = current()
    score = _score(labels_idx, turn)
    for _ in range(max_steps):
        if score == 0:
            break
        best = None  # (cost, square, new_class, new_score)
        for sq in range(64):
            options = np.flatnonzero(allowed[sq])
            options = options[options != labels_idx[sq]]
            if options.size == 0:
                continue  # forced square (a king) or nothing left to try
            nxt = options[np.argmax(probs[sq, options])]
            cost = logp[sq, labels_idx[sq]] - logp[sq, nxt]
            trial = labels_idx.copy()
            trial[sq] = nxt
            new_score = _score(trial, turn)
            if new_score < score and (best is None or cost < best[0]):
                best = (cost, sq, nxt, new_score)
        if best is None:
            return None  # no single switch helps -> give up
        _, sq, _, score = best
        allowed[sq, labels_idx[sq]] = False  # square falls to its next-best class
        labels_idx = current()
    if score != 0:
        return None

    out = np.where(allowed, probs, 0.0)
    out[np.arange(64), labels_idx] += 1e-12  # keep argmax well-defined if all mass was masked
    return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- assemble
def assemble(
    probs: np.ndarray,
    threshold: float = 0.6,
    expected_board: chess.Board | None = None,
    repair: bool = False,
    turn: chess.Color | None = None,
) -> BoardPrediction:
    """Turn (64, 13) class probabilities into a BoardPrediction.

    threshold: squares whose output-label probability is below this are flagged as
        low-confidence (config: assembly.low_confidence_threshold).
    expected_board: optional prior (e.g. the previous position from M5). Only used to
        report disagreements in `issues` (M5 does the move-level reasoning).
    repair: if True and the argmax board is illegal, run repair(). On success the
        repaired labels are returned with legal=True, repaired=True, `changed_squares`,
        and `issues` lists the original problems ("fixed: ...") and each change.
    turn: side to move if known; enables the "side not to move is in check" check.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (64, NUM_CLASSES):
        raise ValueError(f"expected probs of shape (64, {NUM_CLASSES}), got {probs.shape}")

    idx = probs.argmax(axis=1)
    issues = check_legality(labels_to_board([CLASSES[i] for i in idx]), turn)
    legal = not issues  # expected_board mismatch below is informative, not illegal
    repaired, changed = False, []

    if repair and not legal:
        fixed = _repair(probs, issues, turn=turn)
        if fixed is None:
            issues.append("repair failed: no legal board found within the step limit")
        else:
            new_idx = fixed.argmax(axis=1)
            moved = [sq for sq in range(64) if new_idx[sq] != idx[sq]]
            changed = [SQUARE_NAMES[sq] for sq in moved]
            issues = [f"fixed: {s}" for s in issues] + [
                f"repaired {SQUARE_NAMES[sq]}: {CLASSES[idx[sq]]} -> {CLASSES[new_idx[sq]]} "
                f"(p {probs[sq, idx[sq]]:.2f} -> {probs[sq, new_idx[sq]]:.2f})" for sq in moved]
            idx, legal, repaired = new_idx, True, True

    labels = [CLASSES[i] for i in idx]
    board = labels_to_board(labels)
    # Confidence = original probability of the label we output. Equals the max-prob, except
    # on repaired squares, where it is (honestly) lower and usually gets flagged.
    conf = probs[np.arange(64), idx]
    low = [SQUARE_NAMES[sq] for sq in range(64) if conf[sq] < threshold]

    if expected_board is not None:
        diff = [SQUARE_NAMES[sq] for sq in range(64)
                if board.piece_at(sq) != expected_board.piece_at(sq)]
        if diff:
            issues.append(f"differs from expected board at {', '.join(diff)}")

    return BoardPrediction(
        fen=board.board_fen(),
        labels=labels,
        confidence=conf.tolist(),
        low_confidence=low,
        legal=legal,
        issues=issues,
        repaired=repaired,
        changed_squares=changed,
    )


_repair = repair  # inside assemble() the `repair` flag shadows the function name
