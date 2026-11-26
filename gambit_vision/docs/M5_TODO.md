# M5 (optional): move detection — remaining work

Code and self-checks: `notebooks/move_detection.ipynb`. Run: `notebooks/run_move_detection.ipynb`.

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
- [ ] **Tune the detector** — `notebooks/move_detection.ipynb: StableFrameDetector`, `frame_difference`.
      Plot `frame_difference` over time for 2-3 games; choose `diff_threshold` between the
      noise floor and hand motion, and `settle_frames` (~0.3-0.5 s of frames).
      Consider blur/downsampling, a per-square max diff, or a hand mask.
- [ ] **Frame -> probabilities** — build the `board_proba` callable: warp with
      `notebooks/homography.ipynb: warp_board`, crop with `notebooks/capture.ipynb: extract_all_crops`, run an M3 model's `predict_proba`.
- [ ] **`MoveTracker.process_frame`** — fill in the TODO: call `infer_move`, skip
      "no move", hold ambiguous results instead of pushing a guess, push and record events.
      Optional: cross-check with `notebooks/board_assembly.ipynb: assemble(probs)` (`%run` it).
- [ ] **`evaluate_on_game`** — load PGN (`chess.pgn.read_game`), run the tracker over the
      frames, align detected moves with the mainline, compute the metrics in its
      docstring. Turn the TODO notes in its self-check into asserts.
- [ ] Optional: handle a missed stable frame by scoring two-ply candidates; handle
      unknown side-to-move when starting mid-game.
- [ ] Fill the TODO cells in `notebooks/run_move_detection.ipynb`.

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
