"""Smoke tests for the CNN training loop (CPU, tiny fake data, no weight download)."""

import json
from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
cv2 = pytest.importorskip("cv2")

from gambit_vision import train  # noqa: E402
from gambit_vision.labels import CLASSES, NUM_CLASSES  # noqa: E402


def write_fake_session(root, session, n=26, seed=0):
    """Random crops with every class present twice, written in the manifest layout."""
    rng = np.random.default_rng(seed)
    sdir = root / session
    (sdir / "squares").mkdir(parents=True)
    lines = ["session_id,game_id,move_idx,frame_id,square,label,path,fen"]
    for i in range(n):
        label = CLASSES[i % NUM_CLASSES]
        rel = f"squares/{i:06d}_a1.png"
        cv2.imwrite(str(sdir / rel), rng.integers(0, 256, (64, 32, 3), dtype=np.uint8))
        lines.append(f"{session},g1,0,{i:06d},a1,{label},{rel},{chess.STARTING_BOARD_FEN}")
    (sdir / "manifest.csv").write_text("\n".join(lines) + "\n")


def tiny_cfg(tmp_path, epochs=1):
    return SimpleNamespace(
        data=SimpleNamespace(crops_root=str(tmp_path / "crops"), train_sessions=["s01"],
                             val_sessions=["s02"], test_sessions=[],
                             external=SimpleNamespace(chessred_root="x", synthetic_root="y")),
        cnn=SimpleNamespace(arch="resnet18", pretrained=False, num_classes=13, input_size=[64, 32],
                            epochs=epochs, batch_size=8, lr=1e-3, freeze_backbone_epochs=1,
                            augment=SimpleNamespace(color_jitter=0.3, rotation_deg=5, translate=0.05),
                            pretrain_on=None, out_dir=str(tmp_path / "models"), num_workers=0))


def test_make_optimizer_backbone_lr():
    cfg = tiny_cfg(Path("."))
    model = train.build_model(cfg)
    opt = train.make_optimizer(model, cfg)
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs["head"] == pytest.approx(1e-3) and lrs["backbone"] == pytest.approx(1e-4)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_train_cnn_smoke(tmp_path):
    torch.manual_seed(0)
    write_fake_session(tmp_path / "crops", "s01", seed=0)
    write_fake_session(tmp_path / "crops", "s02", n=13, seed=1)
    cfg = tiny_cfg(tmp_path, epochs=2)  # epoch 0 head only, epoch 1 full fine-tune
    best = train.train_cnn(cfg, run_name="smoke")
    out = best.parent
    assert best.exists() and (out / "last.pt").exists() and (out / "config.yaml").exists()
    hist = json.loads((out / "history.json").read_text())["finetune"]
    assert len(hist["epochs"]) == 2 and 0.0 <= hist["epochs"][-1]["val_acc"] <= 1.0
    assert np.load(out / "val_confusion.npy").sum() == 13

    clf = train.CNNClassifier.from_checkpoint(best, device="cpu")
    crops = [np.random.default_rng(i).integers(0, 256, (200, 100, 3), dtype=np.uint8) for i in range(5)]
    p = clf.predict_proba(crops)
    assert p.shape == (5, 13) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert clf.predict(crops).shape == (5,)


def test_train_cnn_with_pretraining(tmp_path, monkeypatch):
    """pretrain_on: external crops first (here faked), then fine-tune on our own sessions."""
    from gambit_vision import dataset

    write_fake_session(tmp_path / "crops", "s01", seed=0)
    write_fake_session(tmp_path / "crops", "s02", n=13, seed=1)
    write_fake_session(tmp_path / "ext", "synth_train", seed=2)
    calls = []

    def fake_external(cfg, which, split="train", **kw):
        calls.append((which, split))
        if split != "train":
            raise FileNotFoundError(split)  # no val split -> pretraining picks best by train acc
        return dataset.load_manifest(tmp_path / "ext", ["synth_train"])

    monkeypatch.setattr(dataset, "load_external", fake_external)
    cfg = tiny_cfg(tmp_path)
    cfg.cnn.pretrain_on = "synthetic"
    cfg.cnn.pretrain_epochs = 1
    best = train.train_cnn(cfg, run_name="pre")
    assert ("synthetic", "train") in calls
    assert (best.parent / "pretrain_best.pt").exists() and best.exists()
    hist = json.loads((best.parent / "history.json").read_text())
    assert "pretrain" in hist and "finetune" in hist
