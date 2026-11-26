import json
import math
import sys

import chess
import numpy as np
import pandas as pd
import pytest

from gambit_vision.chesscog_compare import compare, run_chesscog, run_chesscog_to_csv, squares_wrong
from gambit_vision.evaluate import (
    confusion_matrix_plot,
    evaluate_baseline,
    evaluate_model,
    full_board_accuracy,
    per_class_accuracy,
    per_square_accuracy,
)
from gambit_vision.labels import CLASS_TO_INDEX, CLASSES, NUM_CLASSES, board_to_labels


def test_per_square_accuracy():
    assert per_square_accuracy(["P", "empty", "k", "q"], ["P", "empty", "k", "Q"]) == 0.75
    assert math.isnan(per_square_accuracy([], []))
    with pytest.raises(ValueError):
        per_square_accuracy(["P"], ["P", "p"])


def test_full_board_accuracy_ignores_non_placement_fields():
    start = chess.STARTING_FEN
    after = chess.Board()
    after.push_san("e4")
    assert full_board_accuracy([start, start], [chess.STARTING_BOARD_FEN, after.board_fen()]) == 0.5


def test_confusion_matrix_plot(tmp_path):
    out = tmp_path / "cm.png"
    cm = confusion_matrix_plot(["P", "P", "empty"], ["P", "p", "empty"], out)
    assert out.exists() and cm.shape == (13, 13)


def test_squares_wrong():
    after = chess.Board()
    after.push_san("e4")
    assert squares_wrong(chess.STARTING_FEN, after.board_fen()) == 2
    assert squares_wrong(chess.STARTING_FEN, "") == 64


def test_compare(tmp_path):
    after = chess.Board()
    after.push_san("e4")
    ours = tmp_path / "ours.csv"
    theirs = tmp_path / "chesscog.csv"
    ours.write_text("session_id,frame_id,fen_true,fen_ours\n"
                    f"s1,0,{chess.STARTING_BOARD_FEN},{chess.STARTING_BOARD_FEN}\n")
    theirs.write_text(f"session_id,frame_id,fen_chesscog\ns1,0,{after.board_fen()}\n")
    df = compare(ours, theirs, tmp_path / "chesscog_vs_ours.csv")
    assert bool(df["correct_ours"][0]) and not bool(df["correct_chesscog"][0])
    assert df["squares_wrong_chesscog"][0] == 2


def test_per_class_accuracy():
    acc = per_class_accuracy(["P", "P", "empty"], ["P", "B", "empty"])
    assert acc["P"] == 0.5 and acc["empty"] == 1.0 and acc["K"] is None


# --------------------------------------------------------------------------- fake dataset
# Each fake crop encodes what the fake classifier should answer: red channel = 10 * index of
# the top class; green channel = 10 * index of a runner-up + 5 (0 = confident one-hot).
def _write_crop(path, top: int, second: int | None = None):
    import cv2

    img = np.zeros((20, 10, 3), np.uint8)       # BGR
    img[..., 2] = top * 10
    img[..., 1] = 0 if second is None else second * 10 + 5
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


class FakeClassifier:
    def predict_proba(self, crops):
        out = np.zeros((len(crops), NUM_CLASSES))
        for i, c in enumerate(crops):            # crops are RGB from dataset.read_crop
            top, g = int(c[0, 0, 0]) // 10, int(c[0, 0, 1])
            if g == 0:
                out[i, top] = 1.0
            else:
                out[i, top], out[i, (g - 5) // 10] = 0.6, 0.4
        return out


class FakeBaseline:
    targets = ("occupancy", "color3")

    def predict(self, crops, target="color3"):
        idx = [int(c[0, 0, 0]) // 10 for c in crops]
        labs = [CLASSES[i] for i in idx]
        if target == "occupancy":
            return np.array([lab != "empty" for lab in labs], int)
        return np.array([0 if lab == "empty" else (1 if lab.isupper() else 2) for lab in labs])


def _fake_dataset(root):
    """Session s1 with two frames of the start position.

    frame 000: every crop confidently correct.
    frame 001: the d1 queen crop looks like a king (0.6) with the queen as runner-up (0.4),
               so argmax gives two white kings and repair should fix it.
    """
    sess = root / "s1"
    rows = []
    labels = board_to_labels(chess.Board())
    for frame_id in ("000", "001"):
        for sq, lab in enumerate(labels):
            name = chess.square_name(sq)
            rel = f"squares/{frame_id}_{name}.png"
            if frame_id == "001" and name == "d1":
                _write_crop(sess / rel, CLASS_TO_INDEX["K"], CLASS_TO_INDEX["Q"])
            else:
                _write_crop(sess / rel, CLASS_TO_INDEX[lab])
            rows.append(dict(session_id="s1", game_id="g1", move_idx=0, frame_id=frame_id,
                             square=name, label=lab, path=rel, fen=chess.STARTING_FEN))
    # shuffle to check that evaluate sorts into square order
    pd.DataFrame(rows).sample(frac=1.0, random_state=0).to_csv(sess / "manifest.csv", index=False)


def _cfg(tmp_path, run_name):
    return {
        "data": {"crops_root": str(tmp_path / "crops"), "test_sessions": ["s1"]},
        "assembly": {"low_confidence_threshold": 0.6},
        "eval": {"run_name": run_name, "results_dir": str(tmp_path / "results")},
    }


def test_evaluate_model_writes_results(tmp_path):
    _fake_dataset(tmp_path / "crops")
    m = evaluate_model(_cfg(tmp_path, "fake"), FakeClassifier())
    out = tmp_path / "results" / "fake"
    for f in ("metrics.json", "per_square.csv", "boards.csv", "confusion_matrix.png", "failures.png"):
        assert (out / f).exists(), f

    assert m["n_boards"] == 2 and m["n_squares"] == 128
    assert m["per_square_accuracy"] == pytest.approx(127 / 128)
    assert m["occupancy_accuracy"] == 1.0 and m["color3_accuracy"] == 1.0
    assert m["per_class_accuracy"]["Q"] == 0.5 and m["per_class_accuracy"]["P"] == 1.0
    assert m["full_board_accuracy"] == 0.5 and m["legal_rate"] == 0.5
    assert m["n_illegal_boards"] == 1 and m["repair_rate"] == 1.0
    assert m["full_board_accuracy_repaired"] == 1.0 and m["legal_rate_repaired"] == 1.0
    assert m["errors_caught_by_flag"] == 0.0  # wrong d1 crop has p=0.6, not below threshold 0.6
    assert json.loads((out / "metrics.json").read_text())["n_boards"] == 2

    boards = pd.read_csv(out / "boards.csv", dtype={"frame_id": str})
    assert list(boards.columns[:6]) == ["session_id", "frame_id", "fen_true", "fen_pred",
                                        "fen_pred_repaired", "n_wrong"]
    b1 = boards.set_index("frame_id").loc["001"]
    assert b1["n_wrong"] == 1 and b1["fen_pred_repaired"] == chess.STARTING_BOARD_FEN

    per_sq = pd.read_csv(out / "per_square.csv", dtype={"frame_id": str})
    first = per_sq[per_sq["frame_id"] == "000"]["square"].tolist()
    assert first[:3] == ["a1", "b1", "c1"] and first[-1] == "h8"
    d1 = per_sq[(per_sq["frame_id"] == "001") & (per_sq["square"] == "d1")].iloc[0]
    assert d1["pred"] == "K" and d1["pred_repaired"] == "Q" and not d1["correct"]

    # boards.csv plugs straight into the chesscog comparison
    (tmp_path / "cc.csv").write_text(
        f"session_id,frame_id,fen_chesscog\ns1,000,{chess.STARTING_BOARD_FEN}\ns1,001,\n")
    df = compare(out / "boards.csv", tmp_path / "cc.csv", out / "chesscog_vs_ours.csv")
    assert df["correct_ours"].tolist() == [True, False]
    assert df["squares_wrong_chesscog"].tolist() == [0, 64]


def test_evaluate_baseline(tmp_path):
    _fake_dataset(tmp_path / "crops")
    m = evaluate_baseline(_cfg(tmp_path, "baseline"), FakeBaseline())
    out = tmp_path / "results" / "baseline"
    assert (out / "per_square.csv").exists() and (out / "confusion_matrix_color3.png").exists()
    assert m["occupancy_accuracy"] == 1.0 and m["n_boards"] == 2
    # the fake d1 crop says "K": still white + occupied, so color3 is right everywhere
    assert m["color3_accuracy"] == 1.0 and m["color3_full_board_accuracy"] == 1.0


def test_evaluate_model_requires_test_sessions(tmp_path):
    cfg = _cfg(tmp_path, "x")
    cfg["data"]["test_sessions"] = []
    with pytest.raises(ValueError, match="test_sessions"):
        evaluate_model(cfg, FakeClassifier())


# --------------------------------------------------------------------------- chesscog
def test_run_chesscog_missing_gives_install_help(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "chesscog", None)  # makes `import chesscog...` fail
    with pytest.raises(ImportError, match="pip install chesscog"):
        run_chesscog([tmp_path / "x.png"])


def test_run_chesscog_with_fake_recognizer(tmp_path):
    import cv2

    class FakeRecognizer:
        def predict(self, img, turn):
            assert img.shape == (8, 8, 3) and turn == chess.WHITE
            return chess.Board(), np.zeros((4, 2))

    frames = tmp_path / "crops" / "s1" / "raw"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "000.png"), np.zeros((8, 8, 3), np.uint8))
    paths = [frames / "000.png", frames / "missing.png"]
    assert run_chesscog(paths, recognizer=FakeRecognizer()) == [chess.STARTING_BOARD_FEN, ""]


def test_run_chesscog_to_csv(monkeypatch, tmp_path):
    import cv2

    import gambit_vision.chesscog_compare as cc

    frames = tmp_path / "crops" / "s1" / "raw"
    frames.mkdir(parents=True)
    cv2.imwrite(str(frames / "000.png"), np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(cc, "_load_recognizer", lambda: type(
        "R", (), {"predict": lambda self, img, turn: (chess.Board(), None)})())
    df = run_chesscog_to_csv(str(frames / "*.png"), tmp_path / "cc.csv", turn="white")
    assert df.to_dict("records") == [
        {"session_id": "s1", "frame_id": "000", "fen_chesscog": chess.STARTING_BOARD_FEN}]
    assert (tmp_path / "cc.csv").exists()
