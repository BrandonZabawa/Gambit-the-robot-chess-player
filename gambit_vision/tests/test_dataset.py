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


def transform_cfg():
    return SimpleNamespace(cnn=SimpleNamespace(
        input_size=[200, 100], augment=SimpleNamespace(color_jitter=0.3, rotation_deg=5, translate=0.05)))


def test_build_transforms_output_shape():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    cfg = transform_cfg()
    crop = np.random.default_rng(0).integers(0, 256, (200, 100, 3), dtype=np.uint8)
    ev, tr = dataset.build_transforms(cfg, train=False), dataset.build_transforms(cfg, train=True)
    a, b = ev(crop), ev(crop)
    assert a.shape == (3, 200, 100) and a.dtype == torch.float32 and torch.equal(a, b)
    torch.manual_seed(0)
    x, y = tr(crop), tr(crop)
    assert x.shape == (3, 200, 100) and not torch.equal(x, y)
    # a different input size is resized to cfg.cnn.input_size
    assert ev(np.zeros((50, 30, 3), np.uint8)).shape == (3, 200, 100)


# ----------------------------------------------------------------- geometry fixtures
GEOM = dataset.CropGeometry(square_px=20, margin_px=20, widen=0.5, extend_up=1.0, size_wh=(20, 40))


def code(sq):
    """Colour that identifies a square: red channel encodes the index."""
    return np.array([4 * sq + 2, 128, 0], np.uint8)


def topdown_truth(geom=GEOM):
    """Ideal top-down board, each square filled with code(square)."""
    img = np.zeros((geom.board_px, geom.board_px, 3), np.uint8)
    for sq in range(64):
        x0, y0, x1, y1 = dataset.square_crop_box(sq, dataset.CropGeometry(
            geom.square_px, geom.margin_px, 0.0, 0.0, geom.size_wh))
        img[y0:y1, x0:x1] = code(sq)
    return img


def camera_view(rot_k=0, geom=GEOM):
    """Render the truth board through a perspective "camera", board rotated by rot_k*90 deg.

    Returns (camera image, 3x3 board->camera homography, camera corners in a1,h1,h8,a8 order).
    """
    cv2 = pytest.importorskip("cv2")
    n = geom.board_px
    c = (n - 1) / 2
    ang = np.deg2rad(90 * rot_k)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    T = np.array([[1, 0, c], [0, 1, c], [0, 0, 1]]) @ R @ np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    dst = np.float32([[30, 20], [n + 10, 35], [n + 40, n + 50], [5, n + 30]])  # mild perspective
    M = cv2.getPerspectiveTransform(src, dst) @ T
    cam = cv2.warpPerspective(topdown_truth(geom), M, (n + 60, n + 60), flags=cv2.INTER_NEAREST)
    corners = cv2.perspectiveTransform(dataset.board_corner_targets(geom).reshape(-1, 1, 2), M).reshape(4, 2)
    return cam, M, corners


def center_of(crop, geom=GEOM):
    """Pixel at the centre of the square inside a crop (square is in the lower part)."""
    w, h = geom.size_wh
    sq_top = geom.extend_up / (1 + geom.extend_up) * h
    return crop[int(round((sq_top + h) / 2)), w // 2]


def test_load_crop_geometry_defaults(tmp_path):
    g = dataset.load_crop_geometry()  # repo configs/camera.yaml
    assert g.size_wh == (100, 200) and g.board_px == 8 * g.square_px + 2 * g.margin_px
    assert dataset.load_crop_geometry(tmp_path / "missing.yaml") == dataset.CropGeometry()


def test_square_crop_box():
    g = dataset.CropGeometry(square_px=100, margin_px=100, widen=0.5, extend_up=1.0)
    assert dataset.square_crop_box(0, g) == (50, 700, 250, 900)        # a1, bottom-left
    assert dataset.square_crop_box(63, g) == (750, 0, 950, 200)        # h8, top-right


def test_warp_and_crop_squares_roundtrip():
    cam, _, corners = camera_view(rot_k=0)
    warped, crops = dataset.crops_from_image(cam, corners, GEOM)
    assert warped.shape == (GEOM.board_px, GEOM.board_px, 3)
    assert len(crops) == 64 and all(c.shape == (40, 20, 3) for c in crops)
    for sq in range(64):
        assert abs(int(center_of(crops[sq])[0]) - int(code(sq)[0])) <= 2, chess.SQUARE_NAMES[sq]


def test_corner_orientations_handles_any_order():
    cam, _, corners = camera_view(rot_k=1)
    shuffled = corners[[2, 0, 3, 1]]
    opts = dataset.corner_orientations(shuffled)
    assert len(opts) == 4
    assert any(np.allclose(o, corners, atol=1e-3) for o in opts)


def chessred_fixture(root, rot_k=1):
    """Tiny ChessReD-format dataset: one image of the start position, board rotated."""
    cv2 = pytest.importorskip("cv2")
    import json

    cam, M, corners = camera_view(rot_k)
    (root / "images" / "0").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "0" / "G000_IMG000.jpg"), cv2.cvtColor(cam, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 100])
    names = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    cats = [{"id": i, "name": f"white-{n}"} for i, n in enumerate(names)]
    cats += [{"id": 6 + i, "name": f"black-{n}"} for i, n in enumerate(names)]
    cats.append({"id": 12, "name": "empty"})
    name_to_id = {c["name"]: c["id"] for c in cats}
    letter = {"p": "pawn", "r": "rook", "n": "knight", "b": "bishop", "q": "queen", "k": "king"}
    pieces = []
    s = GEOM.square_px
    for sq, piece in chess.Board().piece_map().items():
        cx = GEOM.margin_px + (sq % 8 + 0.5) * s
        cy = GEOM.margin_px + (7 - sq // 8 + 0.5) * s
        fx, fy = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), M).ravel()
        color = "white" if piece.color else "black"
        pieces.append({"image_id": 0, "category_id": name_to_id[f"{color}-{letter[piece.symbol().lower()]}"],
                       "chessboard_position": chess.SQUARE_NAMES[sq], "id": len(pieces),
                       "bbox": [float(fx - 4), float(fy - 27), 8.0, 30.0]})
    tl, tr, br, bl = dataset.sort_corner_points(corners)
    ann = {
        "images": [{"id": 0, "file_name": "G000_IMG000.jpg", "path": "images/0/G000_IMG000.jpg",
                    "game_id": 0, "move_id": 0, "height": cam.shape[0], "width": cam.shape[1]}],
        "categories": cats,
        "annotations": {"pieces": pieces, "corners": [{"image_id": 0, "corners": {
            "top_left": tl.tolist(), "top_right": tr.tolist(),
            "bottom_right": br.tolist(), "bottom_left": bl.tolist()}}]},
        "splits": {"train": {"image_ids": [0], "n_samples": 1}, "val": {"image_ids": [], "n_samples": 0}},
    }
    (root / "annotations.json").write_text(json.dumps(ann))


def test_chessred_category_to_label():
    assert dataset.chessred_category_to_label("white-pawn") == "P"
    assert dataset.chessred_category_to_label("black-knight") == "n"
    assert dataset.chessred_category_to_label("empty") == "empty"
    with pytest.raises(ValueError):
        dataset.chessred_category_to_label("red-dragon")


@pytest.mark.parametrize("rot_k", [0, 1, 3])
def test_chessred_loader(tmp_path, rot_k):
    chessred_fixture(tmp_path / "chessred", rot_k)
    df = dataset.load_chessred(tmp_path / "chessred", "train", geom=GEOM)
    assert set(dataset.MANIFEST_COLUMNS) <= set(df.columns) and len(df) == 64
    assert set(df.session_id) == {"chessred_0"}
    by_sq = df.set_index("square")
    assert list(by_sq.loc[chess.SQUARE_NAMES, "label"]) == board_to_labels(chess.Board())
    # orientation was recovered from the piece boxes: each crop shows its own square
    for name in ("a1", "h1", "e4", "h8"):
        crop = dataset.read_crop(by_sq.loc[name, "abs_path"])
        assert abs(int(center_of(crop)[0]) - int(code(chess.parse_square(name))[0])) <= 3, name
    # second call is served from the cache
    assert len(dataset.load_chessred(tmp_path / "chessred", "train", geom=GEOM)) == 64


def test_chesscog_synthetic_loader(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import json

    board = chess.Board()
    for san in ("e4", "c5", "Nf3"):
        board.push_san(san)
    split = tmp_path / "synth" / "train"
    split.mkdir(parents=True)
    for i, (white_turn, rot_k) in enumerate([(True, 0), (False, 2)]):
        cam, _, corners = camera_view(rot_k)
        cv2.imwrite(str(split / f"{i:04d}.png"), cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
        meta = {"fen": board.fen(), "corners": corners[[3, 1, 0, 2]].tolist(), "white_turn": white_turn}
        (split / f"{i:04d}.json").write_text(json.dumps(meta))
    df = dataset.load_chesscog_synthetic(tmp_path / "synth", "train", geom=GEOM)
    assert len(df) == 128 and set(df.session_id) == {"synth_train"}
    for fid, g in df.groupby("frame_id"):
        g = g.set_index("square")
        assert list(g.loc[chess.SQUARE_NAMES, "label"]) == board_to_labels(board)
        for name in ("a1", "c5", "h8"):
            crop = dataset.read_crop(g.loc[name, "abs_path"])
            assert abs(int(center_of(crop)[0]) - int(code(chess.parse_square(name))[0])) <= 2, (fid, name)
