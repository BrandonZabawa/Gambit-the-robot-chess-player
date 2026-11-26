"""M3: load square crops written by capture.py (see docs/contracts.md).

Layout: <crops_root>/<session_id>/manifest.csv plus squares/*.png.
Splits are always by session_id, never by frame.
torch is imported lazily so this module works without the `train` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gambit_vision.labels import CLASS_TO_INDEX, CLASSES, label_to_color3, label_to_occupancy

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
        # TODO M3: decide on the final augmentation set. Horizontal flips are label-safe for a
        # single piece; consider also RandomPerspective / blur to mimic session differences.
        steps += [
            T.ColorJitter(brightness=aug.color_jitter, contrast=aug.color_jitter,
                          saturation=aug.color_jitter),
            T.RandomAffine(degrees=aug.rotation_deg, translate=(aug.translate, aug.translate)),
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


# --------------------------------------------------------------------------- external data
def load_chessred(root: str | Path, split: str = "train") -> pd.DataFrame:
    """ChessReD (Masouris & van Gemert 2023): real photos of full boards.

    Source format: `images/<game>/*.jpg` and one `annotations.json` (COCO-like) with per-image
    piece annotations (category + chessboard_position like "e4") and, for a subset, the four
    board corners. Splits are given in the JSON.

    TODO M3: for images with corner annotations, warp with homography.warp_board and cut crops
    with the same geometry as capture.py, then return a DataFrame with the manifest columns
    (session_id = "chessred_<game>") so it plugs into SquareCropDataset. Map ChessReD category
    names to labels.CLASSES. Probably cache crops to disk once rather than cropping on the fly.
    """
    raise NotImplementedError("TODO M3: ChessReD loader")


def load_chesscog_synthetic(root: str | Path, split: str = "train") -> pd.DataFrame:
    """chesscog synthetic dataset (Wölflein & Arandjelović 2021): Blender renders.

    Source format: `<split>/NNNN.png` with a matching `NNNN.json` holding `fen`, `corners`
    (4 board corners in pixels), `white_turn` and lighting metadata.

    TODO M3: warp each render with the corners, crop the 64 squares like capture.py, derive
    labels from the FEN via labels.board_to_labels, and return a manifest-style DataFrame
    (session_id = "synth"). Used when cfg.cnn.pretrain_on == "synthetic".
    """
    raise NotImplementedError("TODO M3: chesscog synthetic loader")
