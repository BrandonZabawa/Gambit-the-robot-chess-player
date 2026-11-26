# M5 (optional): move detection — remaining work

Code: `src/gambit_vision/move_detection.py`. Tests: `tests/test_move_detection.py`
(all synthetic, no camera needed).

## Done
- [x] `infer_move(prev_board, probs)`: scores every legal move (plus "no move") by the
      log-likelihood of the resulting placement under the (64, 13) probabilities. Castling,
      en passant and promotion come for free from python-chess.
  - [x] `unknown_turn=True`: scores the moves of both colours (turn flipped on a copy) and
        reports the colour that moved (`turn`, `turn_flipped`). The tracker uses it until
        the first accepted move.
  - [x] Two-ply candidates (`two_ply_threshold`): if the best single move is more than
        that many nats worse than the per-square-argmax upper bound (`deficit`), all legal
        move pairs are scored too (with a `two_ply_penalty`). This recovers a missed
        stable frame. Pairs that end in the same placement (e.g. `d5 exd6 e.p.` vs
        `d6 exd6`) tie and are flagged as ambiguous, not guessed.
- [x] `changed_squares(prev_labels, new_labels)`, plus `square_differences` /
      `changed_squares_between` that work directly on warped images (square_px/margin_px
      from `configs/camera.yaml`; a8 is assumed top-left, see `white_at_bottom`).
- [x] `StableFrameDetector`: gray -> block downsample -> Gaussian blur before differencing;
      optional per-square max difference (`square_px=`) so a small hand at the edge is not
      averaged away; `changed_squares()` after an event; hand heuristic (changed area vs
      the pre-motion frame > 20%, optional "new skin pixels" HSV mask) with a timeout so a
      lighting change can't block it forever. All thresholds are constructor parameters.
- [x] `MoveTracker.process_frame` / `run`: accepted moves -> `events`; "no move" (spurious
      motion) -> `no_move_events`; ambiguous -> `on_ambiguous(board, result)` callback if
      given (e.g. ask the player), else `ambiguous_events` and the board is kept.
      `run()` takes a frames directory, a video file or an iterable of frames.
- [x] `evaluate_on_game(pgn, frames, board_proba, out_dir=..., frame_plies=...)`: move
      accuracy with resync after errors, missed / spurious / wrong / ambiguous counts,
      special-move breakdown, game_completed, per-event log; writes `move_metrics.json`.
- [x] `simulate_game_frames(pgn, noise=...)`: synthetic frames (hand blobs between plies,
      optional missed pauses / hovering hand) + a noisy-one-hot fake classifier. On the
      15-ply test game (castling x2, en passant, promotion) the tracker reaches move
      accuracy 1.0 up to noise 0.3, including a missed stable frame.

## To do (needs the student's hardware / data)
- [ ] **Record games.** For each game: a video (or frame sequence) from the fixed
      overhead camera plus its PGN. Suggested layout:
      `data/games/<game_id>/{game.pgn, video.mp4 | frames/*.png}`. Include at least one
      castle, one en passant and one promotion across the set (set up positions if needed).
      If frames come from capture sessions, pass the manifest's `move_idx` per frame as
      `frame_plies` to `evaluate_on_game` (exact ground truth alignment).
- [ ] **Frame -> probabilities** — build the `board_proba` callable: warp with
      `homography.py`, crop with M2's crop function, run an M3 model's `predict_proba`.
      Check that the warped image really has a8 top-left (else `white_at_bottom=False`).
- [ ] **Tune on real footage** — `StableFrameDetector(diff_threshold, settle_frames,
      downsample, blur_sigma, max_changed_fraction, skin_check)`: plot `det.last_diff`
      over time for 2-3 games; `diff_threshold` between the noise floor and hand motion,
      `settle_frames` ~0.3-0.5 s. Then `MoveTracker(ambiguity_margin, two_ply_threshold,
      two_ply_penalty)` — these are in nats and depend on how calibrated the model is.
- [ ] Wire `board_proba` into `cli.py eval-moves` (it currently calls `evaluate_on_game`
      without one, which raises a clear error) and optionally add `detect-moves`.

## What to report (recorded games)
Per game and pooled over all games, from `evaluate_on_game`:
- **Move accuracy** (correct plies / total plies, with resync after an error) and
  **games completed without error**.
- Missed, spurious and ambiguous detections.
- Accuracy on special moves (castling, en passant, promotion) separately.
- Ablation: move accuracy when the move is read from the per-frame argmax FEN
  (diff of two `assemble()` results) vs. legality-filtered `infer_move` — this shows
  what the legality prior buys over independent square classification. Also with
  `two_ply_threshold=None` vs. enabled.
- A few failure cases (frame + top-2 candidates with scores; `metrics["events"]`).
