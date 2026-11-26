import math

import chess
import pytest

from gambit_vision.chesscog_compare import compare, squares_wrong
from gambit_vision.evaluate import confusion_matrix_plot, full_board_accuracy, per_square_accuracy


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


@pytest.mark.skip(reason="TODO M4: needs dataset.load_manifest and a small fixture session")
def test_evaluate_model_writes_results(tmp_path):
    pass


@pytest.mark.skip(reason="TODO M4: run_chesscog() not implemented (needs chesscog install)")
def test_run_chesscog():
    pass
