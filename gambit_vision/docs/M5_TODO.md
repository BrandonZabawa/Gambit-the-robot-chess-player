# M5 (optional): move detection — remaining work

Code: `src/gambit_vision/move_detection.py`. Tests: `tests/test_move_detection.py`.

## Done
- `infer_move(prev_board, probs)`: scores every legal move (plus "no move") by the
  log-likelihood of the resulting placement under the (64, 13) probabilities. Castling,
  en passant and promotion come for free from python-chess. Tested on all of them.
- `changed_squares(prev_labels, new_labels)`.
- `StableFrameDetector`: IDLE -> MOTION -> SETTLING -> STABLE state machine on the mean
  absolute frame difference. Works on synthetic frames; thresholds are guesses.

## To do
- [ ] **Record games.** For each game: a video (or frame sequence) from the fixed
      overhead camera plus its PGN. Suggested layout:
      `data/games/<game_id>/{game.pgn, video.mp4 | frames/*.png}`. Include at least one
      castle, one en passant and one promotion across the set (set up positions if needed).
- [ ] **Tune the detector** — `move_detection.py:StableFrameDetector`, `frame_difference`.
      Plot `frame_difference` over time for 2-3 games; choose `diff_threshold` between the
      noise floor and hand motion, and `settle_frames` (~0.3-0.5 s of frames).
      Consider blur/downsampling, a per-square max diff, or a hand mask.
- [ ] **Frame -> probabilities** — build the `board_proba` callable: warp with
      `homography.py`, crop with M2's crop function, run an M3 model's `predict_proba`.
- [ ] **`MoveTracker.process_frame`** — fill in the TODO: call `infer_move`, skip
      "no move", hold ambiguous results instead of pushing a guess, push and record events.
      Optional: cross-check with `board_assembly.assemble(probs)` (lazy import).
- [ ] **`evaluate_on_game`** — load PGN (`chess.pgn.read_game`), run the tracker over the
      frames, align detected moves with the mainline, compute the metrics in its
      docstring. Remove the `skip` marks on the placeholder tests.
- [ ] Optional: handle a missed stable frame by scoring two-ply candidates; handle
      unknown side-to-move when starting mid-game.
- [ ] Register CLI commands in `cli.py` (`detect-moves`, `eval-moves`).

## What to report (recorded games)
Per game and pooled over all games, from `evaluate_on_game`:
- **Move accuracy** (correct plies / total plies, with resync after an error) and
  **games completed without error**.
- Missed, spurious and ambiguous detections.
- Accuracy on special moves (castling, en passant, promotion) separately.
- Ablation: move accuracy when the move is read from the per-frame argmax FEN
  (diff of two `assemble()` results) vs. legality-filtered `infer_move` — this shows
  what the legality prior buys over independent square classification.
- A few failure cases (frame + top-2 candidates with scores).
