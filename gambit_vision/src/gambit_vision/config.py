"""Tiny YAML config loader. `cfg.cnn.lr` style access, `to_dict(cfg)` to go back."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def to_dict(obj: Any) -> Any:
    """Inverse of load_config's conversion (for saving the config next to a checkpoint)."""
    if isinstance(obj, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a YAML file into nested SimpleNamespaces.

    Paths inside the config are relative to gambit_vision/; run commands from there.
    """
    with open(path) as f:
        return _to_ns(yaml.safe_load(f) or {})


def cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read an optional setting, e.g. cfg_get(cfg, "cnn.backbone_lr_mult", 0.1).

    Lets new optional keys have a default without editing every YAML file.
    Works on SimpleNamespace configs and on plain dicts.
    """
    obj = cfg
    for key in dotted.split("."):
        if isinstance(obj, dict):
            if key not in obj:
                return default
            obj = obj[key]
        elif hasattr(obj, key):
            obj = getattr(obj, key)
        else:
            return default
    return default if obj is None else obj
