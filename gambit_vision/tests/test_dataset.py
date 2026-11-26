from types import SimpleNamespace

import chess
import numpy as np
import pytest

from gambit_vision import dataset
from gambit_vision.labels import CLASS_TO_INDEX, board_to_labels


def write_session(root, session, n_frames=1, board=None):
    """Fake session: manifest only (64 rows per frame), crops are not written."""
    board = board or chess.Board()
    labels = board_to_labels(board)
    sdir = root / session
    sdir.mkdir(parents=True)
    lines = ["session_id,game_id,move_idx,frame_id,square,label,path,fen"]
    for f in range(n_frames):
        fid = f"{f:06d}"
        for sq in range(64):
            name = chess.SQUARE_NAMES[sq]
            lines.append(f"{session},g1,{f},{fid},{name},{labels[sq]},squares/{fid}_{name}.png,{board.fen()}")
    (sdir / "manifest.csv").write_text("\n".join(lines) + "\n")


def make_cfg(root, train, val=(), test=()):
    return SimpleNamespace(data=SimpleNamespace(
        crops_root=str(root), train_sessions=list(train), val_sessions=list(val),
        test_sessions=list(test)))


def test_load_manifest_labels(tmp_path):
    write_session(tmp_path, "s01", n_frames=2)
    df = dataset.load_manifest(tmp_path, ["s01"])
    assert len(df) == 128
    assert df["frame_id"].iloc[0] == "000000"  # kept as str
    e1 = df[(df.square == "e1") & (df.frame_id == "000000")].iloc[0]
    assert e1.label == "K" and e1.y == CLASS_TO_INDEX["K"] and e1.y_occ == 1 and e1.y_color3 == 1
    e8 = df[df.square == "e8"].iloc[0]
    assert e8.y_color3 == 2
    assert (df[df.label == "empty"].y == 0).all()
    assert e1.abs_path.endswith("s01/squares/000000_e1.png")


def test_list_sessions_and_all(tmp_path):
    write_session(tmp_path, "s02")
    write_session(tmp_path, "s01")
    assert dataset.list_sessions(tmp_path) == ["s01", "s02"]
    assert set(dataset.load_manifest(tmp_path)["session_id"]) == {"s01", "s02"}


def test_unknown_label_rejected(tmp_path):
    write_session(tmp_path, "s01")
    m = tmp_path / "s01" / "manifest.csv"
    m.write_text(m.read_text().replace(",K,", ",X,"))
    with pytest.raises(ValueError, match="unknown labels"):
        dataset.load_manifest(tmp_path, ["s01"])


def test_missing_session(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.load_manifest(tmp_path, ["nope"])


def test_split_by_session(tmp_path):
    for s in ("a", "b", "c"):
        write_session(tmp_path, s)
    splits = dataset.split_by_session(make_cfg(tmp_path, ["a", "b"], ["c"]))
    assert set(splits["train"].session_id) == {"a", "b"}
    assert set(splits["val"].session_id) == {"c"}
    assert len(splits["test"]) == 0


def test_split_overlap_rejected(tmp_path):
    write_session(tmp_path, "a")
    with pytest.raises(ValueError, match="both"):
        dataset.split_by_session(make_cfg(tmp_path, ["a"], ["a"]))


def test_class_weights(tmp_path):
    write_session(tmp_path, "s01")
    w = dataset.class_weights(dataset.load_manifest(tmp_path, ["s01"]))
    assert w.shape == (13,)
    assert w[CLASS_TO_INDEX["empty"]] < w[CLASS_TO_INDEX["K"]]  # 32 empties vs 1 king


def test_square_crop_dataset(tmp_path):
    pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    write_session(tmp_path, "s01")
    df = dataset.load_manifest(tmp_path, ["s01"]).head(3)
    for p in df.abs_path:
        (tmp_path / "s01" / "squares").mkdir(exist_ok=True)
        cv2.imwrite(p, np.zeros((200, 100, 3), np.uint8))
    ds = dataset.SquareCropDataset(df)
    img, y = ds[0]
    assert len(ds) == 3 and img.shape == (200, 100, 3) and y == df.y.iloc[0]


@pytest.mark.skip(reason="TODO M3")
def test_build_transforms_output_shape():
    """build_transforms(cfg, train=True/False) on a (200, 100, 3) uint8 crop gives a float
    tensor (3, H, W) with H, W = cfg.cnn.input_size; train transform is random, eval is not."""


@pytest.mark.skip(reason="TODO M3")
def test_chessred_loader():
    """load_chessred on a tiny fixture returns manifest columns with labels in CLASSES."""


@pytest.mark.skip(reason="TODO M3")
def test_chesscog_synthetic_loader():
    """load_chesscog_synthetic: 64 rows per render, labels match board_to_labels(FEN)."""
