"""M3: load square crops written by capture.py (see docs/contracts.md).

Also holds the shared board geometry (warp_board, crop_squares) used by capture.py and by
the external-dataset loaders (ChessReD, chesscog synthetic).

Layout: <crops_root>/<session_id>/manifest.csv plus squares/*.png.
Splits are always by session_id, never by frame.
torch is imported lazily so this module works without the `train` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import numpy as np
import pandas as pd

from gambit_vision.labels import (
    CLASS_TO_INDEX,
    CLASSES,
    board_to_labels,
    label_to_color3,
    label_to_occupancy,
    labels_to_board,
)

MANIFEST_COLUMNS = ("session_id", "game_id", "move_idx", "frame_id", "square", "label", "path", "fen")


# --------------------------------------------------------------------------- manifests
def list_sessions(crops_root: str | Path) -> list[str]:
    """Session ids under crops_root (directories that contain a manifest.csv)."""
    root = Path(crops_root)
    return sorted(p.parent.name for p in root.glob("*/manifest.csv"))


def load_manifest(crops_root: str | Path, sessions: Iterable[str] | None = None) -> pd.DataFrame:
    """Concatenate manifest.csv of the given sessions (all sessions if None).

    Adds columns: `abs_path` (crop path resolved against the session dir), `y` (13-class
    index), `y_occ` (0/1), `y_color3` (0 empty / 1 white / 2 black).
    """
    root = Path(crops_root)
    sessions = list_sessions(root) if sessions is None else list(sessions)
    if not sessions:
        raise ValueError(f"no sessions selected under {root}")

    frames = []
    for s in sessions:
        mpath = root / s / "manifest.csv"
        if not mpath.exists():
            raise FileNotFoundError(mpath)
        df = pd.read_csv(mpath, dtype={"frame_id": str, "game_id": str, "session_id": str})
        missing = set(MANIFEST_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{mpath} missing columns {sorted(missing)}")
        df["abs_path"] = [str(root / s / p) for p in df["path"]]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    bad = sorted(set(df["label"]) - set(CLASSES))
    if bad:
        raise ValueError(f"unknown labels in manifest: {bad}")
    df["y"] = df["label"].map(CLASS_TO_INDEX).astype(int)
    df["y_occ"] = df["label"].map(label_to_occupancy).astype(int)
    df["y_color3"] = df["label"].map(label_to_color3).astype(int)
    return df


def split_by_session(cfg) -> dict[str, pd.DataFrame]:
    """Return {"train", "val", "test"} DataFrames from cfg.data.*_sessions.

    Empty session lists give empty DataFrames. Raises if a session is in two splits.
    """
    d = cfg.data
    splits = {"train": d.train_sessions or [], "val": d.val_sessions or [], "test": d.test_sessions or []}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise ValueError(f"sessions {sorted(overlap)} appear in both {a} and {b}")
    return {
        k: load_manifest(d.crops_root, v) if v else pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        for k, v in splits.items()
    }


def read_crop(path: str | Path) -> np.ndarray:
    """Read a crop as an RGB uint8 array (H, W, 3)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- torch
def build_transforms(cfg, train: bool):
    """torchvision transform for crops, from cfg.cnn.input_size and cfg.cnn.augment.

    Eval: resize -> tensor -> ImageNet normalize. Train adds augmentation.
    """
    from torchvision import transforms as T

    h, w = cfg.cnn.input_size
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    steps = [T.ToPILImage(), T.Resize((h, w))]
    if train:
        aug = cfg.cnn.augment
        # Augmentation set (all label-safe for a single square crop):
        #  - ColorJitter: lighting differs between sessions (daylight / lamp / sim).
        #  - RandomAffine: small rotation/shift, covers imperfect warps and marker jitter.
        #  - RandomHorizontalFlip: a mirrored piece is still the same piece, and the crop is
        #    symmetric left/right around the square (widen is applied on both sides).
        #    No vertical flip: the crop is NOT symmetric up/down (margin is above the square).
        #  - Occasional Gaussian blur: focus / motion blur differences between cameras.
        steps += [
            T.ColorJitter(brightness=aug.color_jitter, contrast=aug.color_jitter,
                          saturation=aug.color_jitter, hue=min(0.05, aug.color_jitter / 4)),
            T.RandomAffine(degrees=aug.rotation_deg, translate=(aug.translate, aug.translate)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        ]
    steps += [T.ToTensor(), norm]
    return T.Compose(steps)


def _torch_dataset_base():
    try:
        from torch.utils.data import Dataset
    except ImportError:  # module must import without torch
        return object
    return Dataset


class SquareCropDataset(_torch_dataset_base()):
    """torch Dataset over a manifest DataFrame. Item: (image tensor, class index).

    `target` picks the label column: "y" (13 classes), "y_occ" or "y_color3".
    """

    def __init__(self, df: pd.DataFrame, transform=None, target: str = "y"):
        self.paths = df["abs_path"].tolist()
        self.targets = df[target].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = read_crop(self.paths[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


def class_weights(df: pd.DataFrame) -> np.ndarray:
    """Inverse-frequency weights (13,) for the loss; empty squares dominate real data.

    Classes absent from df get weight 0.
    """
    counts = np.bincount(df["y"].to_numpy(dtype=int), minlength=len(CLASSES)).astype(float)
    w = np.zeros_like(counts)
    nz = counts > 0
    w[nz] = counts.sum() / (nz.sum() * counts[nz])
    return w


# --------------------------------------------------------------------------- board geometry
# Shared by M2 capture.py (own camera) and the external-dataset loaders below, so every
# crop the models see has the same geometry (docs/contracts.md, configs/camera.yaml).
#
# Top-down image layout (white at the bottom, like a diagram):
#
#     (0,0) +-----------------------------+
#           |  margin                     |
#           |   a8 ............... h8     |
#           |   .                   .     |   each square is square_px wide;
#           |   .                   .     |   the 8x8 board starts at (margin, margin)
#           |   a1 ............... h1     |
#           |                     margin  |
#           +-----------------------------+  size = 8 * square_px + 2 * margin
#
# Board corners are always given in the order a1, h1, h8, a8 (the outer corners of the
# playing area, same order as the marker ids in camera.yaml).

DEFAULT_CAMERA_YAML = Path(__file__).resolve().parents[2] / "configs" / "camera.yaml"


@dataclass(frozen=True)
class CropGeometry:
    """Warp + crop parameters (configs/camera.yaml `warp:` and `crop:`)."""

    square_px: int = 100      # side of one square in the warped image
    margin_px: int = 100      # border around the board in the warped image
    widen: float = 0.5        # crop extends widen * square_px left AND right of the square
    extend_up: float = 1.0    # crop extends extend_up * square_px above the square
    size_wh: tuple[int, int] = (100, 200)  # final crop size (w, h)

    @property
    def board_px(self) -> int:
        """Side length of the warped image."""
        return 8 * self.square_px + 2 * self.margin_px


def load_crop_geometry(camera_yaml: str | Path | None = None) -> CropGeometry:
    """Read CropGeometry from configs/camera.yaml (defaults if the file/keys are missing)."""
    import yaml

    path = Path(camera_yaml) if camera_yaml is not None else DEFAULT_CAMERA_YAML
    if not path.exists():
        return CropGeometry()
    with open(path) as f:
        cam = yaml.safe_load(f) or {}
    warp, crop = cam.get("warp", {}) or {}, cam.get("crop", {}) or {}
    d = CropGeometry()
    return CropGeometry(
        square_px=int(warp.get("square_px", d.square_px)),
        margin_px=int(warp.get("margin_px", d.margin_px)),
        widen=float(crop.get("widen", d.widen)),
        extend_up=float(crop.get("extend_up", d.extend_up)),
        size_wh=tuple(int(v) for v in crop.get("size_wh", d.size_wh)),
    )


def board_corner_targets(geom: CropGeometry) -> np.ndarray:
    """Where a1, h1, h8, a8 land in the warped image, (4, 2) float32 (x, y)."""
    m, b = geom.margin_px, 8 * geom.square_px
    return np.float32([[m, m + b], [m + b, m + b], [m + b, m], [m, m]])


def board_homography(corners_a1_h1_h8_a8, geom: CropGeometry) -> np.ndarray:
    """3x3 homography from the camera image to the warped top-down image."""
    import cv2

    src = np.asarray(corners_a1_h1_h8_a8, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, board_corner_targets(geom))


def warp_board(image: np.ndarray, corners_a1_h1_h8_a8, geom: CropGeometry | None = None) -> np.ndarray:
    """Warp a camera image to the top-down board view (board_px x board_px).

    corners_a1_h1_h8_a8: pixel coordinates of the outer corners of the a1, h1, h8, a8
    squares, in that order. Pixels outside the camera image become black.
    """
    import cv2

    geom = geom or CropGeometry()
    H = board_homography(corners_a1_h1_h8_a8, geom)
    return cv2.warpPerspective(image, H, (geom.board_px, geom.board_px), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def square_crop_box(square: int, geom: CropGeometry) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the crop of `square` (python-chess index) in the warped image.

    The square itself spans [sx, sx + s] x [sy, sy + s]; the crop adds widen*s on each side
    and extend_up*s above it (pieces are tall). May exceed the image (crop_squares pads).
    """
    s = geom.square_px
    file, rank = square % 8, square // 8
    sx = geom.margin_px + file * s
    sy = geom.margin_px + (7 - rank) * s          # rank 8 is at the top
    x0 = int(round(sx - geom.widen * s))
    x1 = int(round(sx + s + geom.widen * s))
    y0 = int(round(sy - geom.extend_up * s))
    y1 = sy + s
    return x0, y0, x1, y1


def crop_squares(warped: np.ndarray, geom: CropGeometry | None = None) -> list[np.ndarray]:
    """Cut the 64 square crops (square-index order a1..h8) out of a warped board image.

    Each crop is resized to geom.size_wh. Regions outside the warped image are zero-padded,
    so geometries with a margin smaller than the crop overhang still work.
    """
    import cv2

    geom = geom or CropGeometry()
    pad = int(np.ceil(max(geom.widen, geom.extend_up) * geom.square_px)) + 1
    padded = cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    out = []
    for sq in range(64):
        x0, y0, x1, y1 = square_crop_box(sq, geom)
        crop = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
        out.append(cv2.resize(crop, tuple(geom.size_wh), interpolation=cv2.INTER_AREA))
    return out


def sort_corner_points(points) -> np.ndarray:
    """Order 4 image points as top-left, top-right, bottom-right, bottom-left (in the image).

    Same rule as chesscog: two smallest y are the top pair, each pair sorted by x.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    pts = pts[np.argsort(pts[:, 1], kind="stable")]
    top = pts[:2][np.argsort(pts[:2, 0], kind="stable")]
    bottom = pts[2:][np.argsort(pts[2:, 0], kind="stable")[::-1]]
    return np.concatenate([top, bottom])  # TL, TR, BR, BL


def _ccw_from_bottom_left(tl_tr_br_bl: np.ndarray) -> np.ndarray:
    """TL, TR, BR, BL -> BL, BR, TR, TL: the a1, h1, h8, a8 order when white is at the bottom."""
    tl, tr, br, bl = tl_tr_br_bl
    return np.stack([bl, br, tr, tl])


def corner_orientations(points) -> list[np.ndarray]:
    """The 4 possible a1, h1, h8, a8 assignments of 4 unordered board corners.

    A photo is never mirrored, so only the 4 rotations of the board are possible.
    Index k means the board is rotated by k * 90 degrees; k = 0 is white at the bottom.
    """
    base = _ccw_from_bottom_left(sort_corner_points(points))
    return [np.roll(base, -k, axis=0) for k in range(4)]


def crops_from_image(image: np.ndarray, corners_a1_h1_h8_a8, geom: CropGeometry | None = None):
    """Convenience: warp + crop. Returns (warped, list of 64 crops)."""
    geom = geom or CropGeometry()
    warped = warp_board(image, corners_a1_h1_h8_a8, geom)
    return warped, crop_squares(warped, geom)


# --------------------------------------------------------------------------- external data
# Both loaders cut the external images into crops ONCE and cache them on disk in the same
# layout as our own capture sessions (<cache>/<session>/squares + manifest.csv). After that
# they are ordinary sessions and go through load_manifest / SquareCropDataset unchanged.

def _write_session(cache_root: Path, session: str, frames: list[dict], geom: CropGeometry) -> None:
    """Write crops + manifest.csv for one session.

    frames: dicts with keys image (RGB), corners (a1,h1,h8,a8), labels (64 str), frame_id,
    game_id, move_idx, fen.
    """
    import cv2

    sdir = cache_root / session
    (sdir / "squares").mkdir(parents=True, exist_ok=True)
    (sdir / "frames").mkdir(exist_ok=True)
    rows = []
    for fr in frames:
        warped, crops = crops_from_image(fr["image"], fr["corners"], geom)
        fid = fr["frame_id"]
        cv2.imwrite(str(sdir / "frames" / f"{fid}.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
        for sq, crop in enumerate(crops):
            name = chess.SQUARE_NAMES[sq]
            rel = f"squares/{fid}_{name}.png"
            cv2.imwrite(str(sdir / rel), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            rows.append({"session_id": session, "game_id": fr["game_id"], "move_idx": fr["move_idx"],
                         "frame_id": fid, "square": name, "label": fr["labels"][sq],
                         "path": rel, "fen": fr["fen"]})
    pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)).to_csv(sdir / "manifest.csv", index=False)


def _cached_sessions(cache_root: Path) -> list[str] | None:
    """Session list if a complete cache exists (marked by a DONE file), else None."""
    if (cache_root / "DONE").exists():
        return list_sessions(cache_root)
    return None


# ChessReD category names look like "white-pawn" / "black-king" / "empty".
_PIECE_LETTER = {"pawn": "p", "knight": "n", "bishop": "b", "rook": "r", "queen": "q", "king": "k"}


def chessred_category_to_label(name: str) -> str:
    """'white-pawn' -> 'P', 'black-knight' -> 'n', 'empty' -> 'empty'."""
    name = name.lower().replace("_", "-").replace(" ", "-")
    if name == "empty":
        return "empty"
    color, _, piece = name.partition("-")
    if color not in ("white", "black") or piece not in _PIECE_LETTER:
        raise ValueError(f"unknown ChessReD category {name!r}")
    letter = _PIECE_LETTER[piece]
    return letter.upper() if color == "white" else letter


def _chessred_corners(entry) -> np.ndarray:
    """4 corner points from a ChessReD corner annotation (dict of named corners or a list)."""
    # verify against annotations.json: we expect {"top_left": [x, y], "top_right": ...,
    # "bottom_right": ..., "bottom_left": ...}. The names are not trusted for orientation
    # (see _pick_orientation); only the 4 points are used.
    c = entry.get("corners", entry) if isinstance(entry, dict) else entry
    if isinstance(c, dict):
        pts = [c[k] for k in ("top_left", "top_right", "bottom_right", "bottom_left") if k in c]
        if len(pts) != 4:
            pts = list(c.values())[:4]
    else:
        pts = c
    return np.asarray(pts, dtype=np.float32).reshape(4, 2)


def _pick_orientation(points, pieces: list[tuple[str, list[float]]], geom: CropGeometry) -> np.ndarray:
    """Choose which of the 4 board rotations matches the piece annotations best.

    pieces: (square name, COCO bbox [x, y, w, h]) per piece. The bottom centre of a piece's
    box is near where it stands, so under the right homography it falls in its annotated
    square. We count hits for each rotation and keep the best one. Without boxes we fall
    back to rotation 0 (white at the bottom of the image).
    """
    import cv2

    options = corner_orientations(points)
    if not pieces:
        return options[0]
    feet = np.float32([[x + w / 2, y + 0.9 * h] for _, (x, y, w, h) in pieces]).reshape(-1, 1, 2)
    want = np.array([chess.parse_square(sq) for sq, _ in pieces])
    best, best_hits = options[0], -1
    for corners in options:
        H = board_homography(corners, geom)
        p = cv2.perspectiveTransform(feet, H).reshape(-1, 2)
        file = np.floor((p[:, 0] - geom.margin_px) / geom.square_px).astype(int)
        rank = 7 - np.floor((p[:, 1] - geom.margin_px) / geom.square_px).astype(int)
        ok = (file >= 0) & (file < 8) & (rank >= 0) & (rank < 8)
        hits = int(np.sum(ok & (rank * 8 + file == want)))
        if hits > best_hits:
            best, best_hits = corners, hits
    return best


def load_chessred(root: str | Path, split: str = "train", cache_dir: str | Path | None = None,
                  geom: CropGeometry | None = None, max_images: int | None = None) -> pd.DataFrame:
    """ChessReD (Masouris & van Gemert 2023): real photos of full boards -> crop manifest.

    Expected source layout (verify against annotations.json, field names from memory):

        <root>/annotations.json
        <root>/images/<game_id>/G000_IMG000.jpg ...

        annotations.json = {
          "images": [{"id", "file_name", "path" (relative to root), "game_id", "move_id"}, ...],
          "categories": [{"id", "name": "white-pawn" ... "black-king", "empty"}, ...],
          "annotations": {
            "pieces":  [{"image_id", "category_id", "chessboard_position": "e4", "bbox"}, ...],
            "corners": [{"image_id", "corners": {"top_left": [x, y], ...}}, ...]  # ChessReD2K subset
          },
          "splits": {"train": {"image_ids": [...]}, "val": {...}, "test": {...},
                     "chessred2k": {"train": {...}, ...}}
        }

    Only images with corner annotations can be warped (about 2k images). The board
    rotation in each photo is resolved from the piece bounding boxes (_pick_orientation),
    so we do not depend on what "top_left" means. Squares without a piece annotation are
    "empty". Crops are cached under cache_dir (default <root>/gambit_crops/<split>), one
    session per game: session_id = "chessred_<game_id>".
    Returns a load_manifest-style DataFrame (manifest columns + abs_path, y, y_occ, y_color3).
    """
    import json

    import cv2

    root = Path(root)
    geom = geom or load_crop_geometry()
    cache = Path(cache_dir) if cache_dir is not None else root / "gambit_crops" / split
    sessions = _cached_sessions(cache)
    if sessions is not None:
        return load_manifest(cache, sessions)

    with open(root / "annotations.json") as f:
        ann = json.load(f)
    cat_to_label = {c["id"]: chessred_category_to_label(c["name"]) for c in ann["categories"]}
    a = ann["annotations"]
    corners_by_img = {c["image_id"]: _chessred_corners(c) for c in a.get("corners", [])}
    pieces_by_img: dict = {}
    for p in a.get("pieces", []):
        pieces_by_img.setdefault(p["image_id"], []).append(p)

    split_ids = None
    splits = ann.get("splits", {})
    if split in splits and "image_ids" in splits[split]:
        split_ids = set(splits[split]["image_ids"])

    frames_by_session: dict[str, list[dict]] = {}
    n = 0
    for img in ann["images"]:
        iid = img["id"]
        if iid not in corners_by_img or (split_ids is not None and iid not in split_ids):
            continue
        rel = img.get("path") or img["file_name"]  # verify against annotations.json
        bgr = cv2.imread(str(root / rel), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(root / rel)
        labels = ["empty"] * 64
        boxed = []
        for p in pieces_by_img.get(iid, []):
            lab = cat_to_label[p["category_id"]]
            if lab == "empty":
                continue
            sq = p["chessboard_position"]
            labels[chess.parse_square(sq)] = lab
            if p.get("bbox"):
                boxed.append((sq, p["bbox"]))
        corners = _pick_orientation(corners_by_img[iid], boxed, geom)
        game = str(img.get("game_id", "0"))
        session = f"chessred_{game}"
        frames_by_session.setdefault(session, []).append({
            "image": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "corners": corners, "labels": labels,
            "frame_id": f"{iid:06d}" if isinstance(iid, int) else str(iid), "game_id": game,
            "move_idx": int(img.get("move_id", 0)),
            "fen": labels_to_board(labels).board_fen(),
        })
        n += 1
        if max_images is not None and n >= max_images:
            break
    if not frames_by_session:
        raise ValueError(f"no ChessReD images with corner annotations in split {split!r}")
    for session, frames in frames_by_session.items():
        _write_session(cache, session, frames, geom)
    (cache / "DONE").write_text(f"{n} images\n")
    return load_manifest(cache, sorted(frames_by_session))


def load_chesscog_synthetic(root: str | Path, split: str = "train", cache_dir: str | Path | None = None,
                            geom: CropGeometry | None = None, max_images: int | None = None) -> pd.DataFrame:
    """chesscog synthetic renders (Wölflein & Arandjelović 2021) -> crop manifest.

    Source layout: <root>/<split>/NNNN.png with NNNN.json = {"fen": ..., "corners": [[x, y] x4],
    "white_turn": bool, ...}. The corners are unordered image points; as in chesscog, we sort
    them TL, TR, BR, BL and the camera looks from the side of the player to move: white_turn
    -> white at the bottom of the image, else black at the bottom (board rotated 180 deg).
    Labels come from the FEN. Crops cached under cache_dir (default
    <root>/gambit_crops/<split>), one session: session_id = "synth_<split>".
    """
    import json

    import cv2

    root = Path(root)
    geom = geom or load_crop_geometry()
    cache = Path(cache_dir) if cache_dir is not None else root / "gambit_crops" / split
    sessions = _cached_sessions(cache)
    if sessions is not None:
        return load_manifest(cache, sessions)

    jsons = sorted((root / split).glob("*.json"))
    if max_images is not None:
        jsons = jsons[:max_images]
    if not jsons:
        raise FileNotFoundError(f"no renders in {root / split}")
    frames = []
    for j in jsons:
        with open(j) as f:
            meta = json.load(f)
        bgr = cv2.imread(str(j.with_suffix(".png")), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(j.with_suffix(".png"))
        board = chess.Board(meta["fen"]) if " " in meta["fen"] else chess.Board(meta["fen"] + " w - - 0 1")
        options = corner_orientations(meta["corners"])
        corners = options[0] if meta.get("white_turn", True) else options[2]
        frames.append({"image": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "corners": corners,
                       "labels": board_to_labels(board), "frame_id": j.stem, "game_id": "synth",
                       "move_idx": 0, "fen": board.board_fen()})
    session = f"synth_{split}"
    _write_session(cache, session, frames, geom)
    (cache / "DONE").write_text(f"{len(frames)} images\n")
    return load_manifest(cache, [session])


def load_external(cfg, which: str, split: str = "train", **kw) -> pd.DataFrame:
    """Dispatch used by train.py for cfg.cnn.pretrain_on: "chessred" or "synthetic"."""
    ext = cfg.data.external
    if which == "chessred":
        return load_chessred(ext.chessred_root, split, **kw)
    if which == "synthetic":
        return load_chesscog_synthetic(ext.synthetic_root, split, **kw)
    raise ValueError(f"unknown pretrain dataset {which!r}; use chessred | synthetic")
