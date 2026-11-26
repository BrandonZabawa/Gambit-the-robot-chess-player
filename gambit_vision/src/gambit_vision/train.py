"""M3: ResNet-18 transfer learning on square crops, 13 classes (labels.CLASSES order).

torch / torchvision are imported inside functions so the package imports without them
(install with `pip install -e .[train]`, or use Colab).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gambit_vision.config import cfg_get, to_dict
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


class _NullWriter:
    """Stand-in for SummaryWriter when tensorboard is not installed."""

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def _make_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # tensorboard is optional; training still works
        print("tensorboard not installed; no TensorBoard logs")
        return _NullWriter()
    return SummaryWriter(str(log_dir))


def make_optimizer(model, cfg):
    """AdamW with two parameter groups: the new fc head at cfg.cnn.lr and the pretrained
    backbone at lr * cnn.backbone_lr_mult (default 0.1). The backbone already has good
    ImageNet features; a smaller step size changes them gently instead of destroying them.
    """
    import torch

    lr = cfg.cnn.lr
    mult = cfg_get(cfg, "cnn.backbone_lr_mult", 0.1)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    return torch.optim.AdamW([
        {"params": backbone, "lr": lr * mult, "name": "backbone"},
        {"params": head, "lr": lr, "name": "head"},
    ], weight_decay=cfg_get(cfg, "cnn.weight_decay", 1e-4))


def train_one_epoch(model, loader, criterion, optimizer, device, writer=None, step0: int = 0,
                    tag: str = "train") -> tuple[float, float]:
    """One pass over `loader`. Returns (mean loss, accuracy). Logs loss per batch."""
    import torch

    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), torch.as_tensor(y, device=device)
        logits = model(x)                     # forward
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()                       # backprop
        optimizer.step()                      # update weights
        n = y.numel()
        loss_sum += loss.item() * n
        correct += (logits.argmax(1) == y).sum().item()
        total += n
        if writer is not None:
            writer.add_scalar(f"{tag}/loss", loss.item(), step0 + step)
    return loss_sum / max(total, 1), correct / max(total, 1)


def evaluate_loader(model, loader, device, criterion=None) -> dict:
    """Accuracy, mean loss and a (13, 13) confusion matrix (rows true, cols predicted)."""
    import torch

    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loss_sum, total = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), torch.as_tensor(y, device=device)
            logits = model(x)
            if criterion is not None:
                loss_sum += criterion(logits, y).item() * y.numel()
            pred = logits.argmax(1)
            np.add.at(cm, (y.cpu().numpy(), pred.cpu().numpy()), 1)
            total += y.numel()
    acc = float(np.trace(cm) / total) if total else float("nan")
    return {"acc": acc, "loss": loss_sum / total if total else float("nan"), "n": total,
            "confusion_matrix": cm}


def _fit(model, cfg, train_df, val_df, out: Path, writer, device, epochs: int, prefix: str = "") -> dict:
    """Train `model` on train_df for `epochs`, validate on val_df after each epoch.

    Saves <out>/<prefix>last.pt every epoch and <prefix>best.pt when val accuracy improves
    (if val_df is empty, train accuracy is used to pick best). Returns a history dict.
    """
    import torch
    from torch.utils.data import DataLoader

    from gambit_vision.dataset import SquareCropDataset, build_transforms, class_weights

    c = cfg.cnn
    workers = cfg_get(cfg, "cnn.num_workers", 2)
    pin = device.type == "cuda"
    train_dl = DataLoader(SquareCropDataset(train_df, build_transforms(cfg, train=True)),
                          batch_size=c.batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_dl = None
    if len(val_df):
        val_dl = DataLoader(SquareCropDataset(val_df, build_transforms(cfg, train=False)),
                            batch_size=c.batch_size, shuffle=False, num_workers=workers, pin_memory=pin)

    # Class-weighted loss: empty squares are ~half of all crops and kings/queens are rare,
    # so plain CE would under-train the rare classes. Inverse-frequency weights balance it.
    # Classes absent from train get weight 0 (they contribute nothing either way).
    if cfg_get(cfg, "cnn.class_weighted_loss", True):
        weight = torch.tensor(class_weights(train_df), dtype=torch.float32, device=device)
    else:
        weight = None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    eval_criterion = torch.nn.CrossEntropyLoss()  # unweighted val loss, comparable across runs
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history = {"epochs": [], "best_epoch": None, "best_val_acc": None}
    best_acc = -1.0
    for epoch in range(epochs):
        # Warm-up: train only the new head first so random fc gradients don't wreck the
        # pretrained backbone, then fine-tune everything.
        set_backbone_trainable(model, epoch >= c.freeze_backbone_epochs)
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device, writer,
                                          step0=epoch * len(train_dl), tag=f"{prefix}train")
        writer.add_scalar(f"{prefix}train/epoch_loss", tr_loss, epoch)
        writer.add_scalar(f"{prefix}train/acc", tr_acc, epoch)
        for g in optimizer.param_groups:
            writer.add_scalar(f"{prefix}lr/{g.get('name', 'group')}", g["lr"], epoch)

        if val_dl is not None:
            val = evaluate_loader(model, val_dl, device, eval_criterion)
            val_acc = val["acc"]
            writer.add_scalar(f"{prefix}val/acc", val_acc, epoch)
            writer.add_scalar(f"{prefix}val/loss", val["loss"], epoch)
            cm = val["confusion_matrix"]
        else:
            val_acc, cm = float("nan"), None
        scheduler.step()

        score = tr_acc if val_dl is None else val_acc
        print(f"{prefix}epoch {epoch + 1}/{epochs}: train loss {tr_loss:.4f} acc {tr_acc:.4f}"
              f" | val acc {val_acc:.4f}")
        history["epochs"].append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                                  "val_acc": val_acc})
        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                "config": to_dict(cfg),
                "val_confusion_matrix": None if cm is None else cm.tolist()}
        torch.save(ckpt, out / f"{prefix}last.pt")
        if score > best_acc:
            best_acc = score
            torch.save(ckpt, out / f"{prefix}best.pt")
            history["best_epoch"], history["best_val_acc"] = epoch, val_acc
            if cm is not None:
                np.save(out / f"{prefix}val_confusion.npy", cm)
    return history


def train_cnn(cfg, run_name: str | None = None) -> Path:
    """Fine-tune ResNet-18 on the train sessions, validate on val sessions.

    Writes <cfg.cnn.out_dir>/<run_name>/{best.pt, last.pt, config.yaml, history.json,
    val_confusion.npy} and TensorBoard logs under <out_dir>/<run_name>/tb. Returns the
    path to best.pt (highest val accuracy). Run three times for the experiments: sim only,
    real only, mixed (different train_sessions lists in the config).

    Optional pretraining (cfg.cnn.pretrain_on = "chessred" | "synthetic"): first train on
    the external dataset's crops for cnn.pretrain_epochs (default 5), validating on its
    "val" split, then reload the best pretraining weights and fine-tune on our own sessions.
    Pretraining files get a "pretrain_" prefix.

    Optional keys (defaults in brackets): cnn.backbone_lr_mult [0.1], cnn.weight_decay [1e-4],
    cnn.num_workers [2], cnn.class_weighted_loss [true], cnn.pretrain_epochs [5],
    cnn.pretrain_max_images [all].
    """
    import json

    import pandas as pd
    import torch
    import yaml

    from gambit_vision.dataset import load_external, split_by_session

    c = cfg.cnn
    out = Path(c.out_dir) / (run_name or "run")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = split_by_session(cfg)
    if len(splits["train"]) == 0:
        raise ValueError("cfg.data.train_sessions is empty")

    model = build_model(cfg).to(device)
    writer = _make_writer(out / "tb")
    history = {}

    if c.pretrain_on:
        kw = {"max_images": cfg_get(cfg, "cnn.pretrain_max_images", None)}
        pre_train = load_external(cfg, c.pretrain_on, "train", **kw)
        try:
            pre_val = load_external(cfg, c.pretrain_on, "val", **kw)
        except (FileNotFoundError, ValueError):
            pre_val = pd.DataFrame()
        print(f"pretraining on {c.pretrain_on}: {len(pre_train)} train / {len(pre_val)} val crops")
        history["pretrain"] = _fit(model, cfg, pre_train, pre_val, out, writer, device,
                                   epochs=cfg_get(cfg, "cnn.pretrain_epochs", 5), prefix="pretrain_")
        best_pre = torch.load(out / "pretrain_best.pt", map_location=device)
        model.load_state_dict(best_pre["model"])

    history["finetune"] = _fit(model, cfg, splits["train"], splits["val"], out, writer, device,
                               epochs=c.epochs)
    writer.close()
    (out / "history.json").write_text(json.dumps(history, indent=2))
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
