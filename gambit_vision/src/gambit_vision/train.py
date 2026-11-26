"""M3: ResNet-18 transfer learning on square crops, 13 classes (labels.CLASSES order).

torch / torchvision are imported inside functions so the package imports without them
(install with `pip install -e .[train]`, or use Colab).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gambit_vision.labels import NUM_CLASSES


def build_model(cfg):
    """torchvision ResNet-18 (ImageNet weights if cfg.cnn.pretrained), fc -> NUM_CLASSES."""
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    if cfg.cnn.arch != "resnet18":
        raise ValueError(f"only resnet18 is supported, got {cfg.cnn.arch!r}")
    if cfg.cnn.num_classes != NUM_CLASSES:
        raise ValueError(f"cnn.num_classes must be {NUM_CLASSES}")
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.cnn.pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def set_backbone_trainable(model, trainable: bool) -> None:
    """Freeze / unfreeze everything except the final fc layer."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable


def train_cnn(cfg, run_name: str | None = None) -> Path:
    """Fine-tune ResNet-18 on the train sessions, validate on val sessions.

    Writes <cfg.cnn.out_dir>/<run_name>/{best.pt, last.pt, config.yaml} and TensorBoard
    logs under <out_dir>/<run_name>/tb. Returns the path to best.pt.
    Run three times for the experiments: sim only, real only, mixed (different
    train_sessions lists in the config).
    """
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from gambit_vision.config import to_dict
    from gambit_vision.dataset import SquareCropDataset, build_transforms, split_by_session

    c = cfg.cnn
    out = Path(c.out_dir) / (run_name or "run")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = split_by_session(cfg)
    train_ds = SquareCropDataset(splits["train"], build_transforms(cfg, train=True))
    val_ds = SquareCropDataset(splits["val"], build_transforms(cfg, train=False))
    train_dl = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=c.batch_size, shuffle=False, num_workers=2)

    model = build_model(cfg).to(device)
    # TODO M3: if c.pretrain_on is set, first train on dataset.load_chessred /
    # load_chesscog_synthetic (or load weights from that run) before fine-tuning here.

    # TODO M3: class-weighted loss? dataset.class_weights(splits["train"]) -> CrossEntropyLoss(weight=...)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr)  # TODO M3: lower lr for backbone?
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
    writer = SummaryWriter(out / "tb")

    best_acc = -1.0
    for epoch in range(c.epochs):
        set_backbone_trainable(model, epoch >= c.freeze_backbone_epochs)

        model.train()
        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            # TODO M3: forward, loss, optimizer.zero_grad(), backward, step;
            # log loss to writer with global step epoch * len(train_dl) + step.
            raise NotImplementedError("TODO M3: training step")

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                # TODO M3: predictions, count correct per square; also keep a per-class
                # confusion matrix for the report.
                raise NotImplementedError("TODO M3: validation step")
        val_acc = correct / max(total, 1)
        writer.add_scalar("val/acc", val_acc, epoch)
        scheduler.step()

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                "config": to_dict(cfg)}
        torch.save(ckpt, out / "last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, out / "best.pt")

    writer.close()
    return out / "best.pt"


class CNNClassifier:
    """Inference wrapper: crops -> (N, 13) probabilities (docs/contracts.md)."""

    def __init__(self, model, cfg, device: str | None = None):
        import torch

        from gambit_vision.dataset import build_transforms

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.transform = build_transforms(cfg, train=False)

    @classmethod
    def from_checkpoint(cls, path: str | Path, cfg=None, device: str | None = None) -> "CNNClassifier":
        """Load a checkpoint written by train_cnn. Uses the config saved in it if cfg is None."""
        import torch

        from gambit_vision.config import _to_ns

        ckpt = torch.load(path, map_location="cpu")
        cfg = cfg or _to_ns(ckpt["config"])
        cfg.cnn.pretrained = False  # weights come from the checkpoint
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"])
        return cls(model, cfg, device)

    def predict_proba(self, crops, batch_size: int = 64) -> np.ndarray:
        """crops: list or array of RGB uint8 (H, W, 3). Returns (N, 13), rows sum to 1."""
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                x = torch.stack([self.transform(c) for c in crops[i:i + batch_size]]).to(self.device)
                out.append(torch.softmax(self.model(x), dim=1).cpu().numpy())
        if not out:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def predict(self, crops) -> np.ndarray:
        return self.predict_proba(crops).argmax(axis=1)
