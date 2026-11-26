"""M3 baseline: hand-crafted color + edge features -> scikit-learn classifier.

Predicts the coarse targets from labels.py: occupancy (empty/occupied) and color3
(empty/white/black). It does not tell piece types apart, so it is compared with the CNN at
the occupancy / color level, not on all 13 classes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from gambit_vision.labels import COLOR3, OCCUPANCY

TARGETS = {"occupancy": ("y_occ", OCCUPANCY), "color3": ("y_color3", COLOR3)}


# --------------------------------------------------------------------------- features
def _gray(crop: np.ndarray) -> np.ndarray:
    import cv2

    return crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)


def hsv_histogram(crop: np.ndarray, bins: tuple[int, int, int] = (8, 4, 4)) -> np.ndarray:
    """Normalized 3D HSV histogram of an RGB crop, flattened (prod(bins),).

    Hue gets the most bins: it separates piece colors best; saturation / value bins help
    with white vs black pieces and with lighting. OpenCV hue range is [0, 180).
    Possible extension: separate histograms for the lower part of the crop (the square,
    occupancy) and the upper part (the piece body, color).
    """
    import cv2

    hsv = cv2.cvtColor(np.ascontiguousarray(crop, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = hist.ravel().astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def edge_density(crop: np.ndarray, low: int = 50, high: int = 150, grid: int = 2) -> np.ndarray:
    """Fraction of Canny edge pixels, overall and per cell of a grid x grid split
    (1 + grid*grid,). Empty squares have few edges; pieces have many.
    """
    import cv2

    edges = cv2.Canny(np.ascontiguousarray(_gray(crop), dtype=np.uint8), low, high) > 0
    feats = [edges.mean()]
    h, w = edges.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            cell = edges[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            feats.append(cell.mean() if cell.size else 0.0)
    return np.asarray(feats, dtype=np.float32)


def lbp_histogram(crop: np.ndarray, points: int = 8, radius: int = 1) -> np.ndarray:
    """Histogram of uniform local binary patterns on the gray crop (points + 2,).

    Small numpy implementation (no scikit-image dependency), same definition as
    skimage's method="uniform":
      1. For every pixel, sample `points` neighbours on a circle of `radius`
         (bilinear interpolation for off-grid positions).
      2. Bit p = 1 if neighbour p >= centre pixel.
      3. A pattern is "uniform" if the circular bit string has <= 2 0/1 transitions
         (edges, corners, flat spots). Uniform patterns get code = number of 1 bits
         (0..points); all others share code points + 1.
    The normalized histogram of codes describes texture: flat squares vs carved pieces.
    """
    g = _gray(crop).astype(np.float32)
    h, w = g.shape
    pad = radius + 1
    gp = np.pad(g, pad, mode="edge")
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy += pad
    xx += pad

    bits = []
    for p in range(points):
        angle = 2 * np.pi * p / points
        sy = yy - radius * np.sin(angle)
        sx = xx + radius * np.cos(angle)
        # Snap tiny float errors so on-grid neighbours are sampled exactly.
        sy, sx = np.round(sy, 6), np.round(sx, 6)
        y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
        fy, fx = sy - y0, sx - x0
        val = (gp[y0, x0] * (1 - fy) * (1 - fx) + gp[y0, x0 + 1] * (1 - fy) * fx
               + gp[y0 + 1, x0] * fy * (1 - fx) + gp[y0 + 1, x0 + 1] * fy * fx)
        bits.append(val >= g - 1e-4)
    bits = np.stack(bits).astype(np.int32)                 # (points, h, w)
    transitions = np.abs(bits - np.roll(bits, -1, axis=0)).sum(axis=0)
    codes = np.where(transitions <= 2, bits.sum(axis=0), points + 1)
    hist = np.bincount(codes.ravel(), minlength=points + 2).astype(np.float32)
    return hist / max(hist.sum(), 1.0)


FEATURES = {"hsv_hist": hsv_histogram, "edge_density": edge_density, "lbp": lbp_histogram}


def extract_features(crops, features: list[str]) -> np.ndarray:
    """Stack the named features for each crop -> (N, D) float32."""
    unknown = set(features) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}; choose from {sorted(FEATURES)}")
    rows = [np.concatenate([np.ravel(FEATURES[f](c)) for f in features]) for c in crops]
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- classifier
def make_estimator(kind: str, **params):
    """StandardScaler + the chosen sklearn classifier, with predict_proba enabled.

    Extra keyword args override the classifier defaults, e.g. make_estimator("svm", C=10)
    or make_estimator("random_forest", n_estimators=500) when tuning on the val sessions.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Defaults are reasonable starting points; tune on the val sessions with real data.
    models = {
        # SVC(probability=True) is deprecated in recent sklearn; calibrate explicitly instead.
        "svm": lambda: CalibratedClassifierCV(
            SVC(**{"kernel": "rbf", "class_weight": "balanced", **params}), ensemble=False),
        "random_forest": lambda: RandomForestClassifier(
            **{"n_estimators": 200, "class_weight": "balanced", **params}),
        "logistic": lambda: LogisticRegression(
            **{"max_iter": 1000, "class_weight": "balanced", **params}),
    }
    if kind not in models:
        raise ValueError(f"unknown classifier {kind!r}; choose from {sorted(models)}")
    return make_pipeline(StandardScaler(), models[kind]())


class BaselineClassifier:
    """One sklearn model per target ("occupancy", "color3") on shared features."""

    def __init__(self, features: list[str], classifier: str = "svm",
                 targets: tuple[str, ...] = ("occupancy", "color3")):
        for t in targets:
            if t not in TARGETS:
                raise ValueError(f"unknown target {t!r}; choose from {sorted(TARGETS)}")
        self.features = list(features)
        self.classifier = classifier
        self.targets = tuple(targets)
        self.models: dict = {}

    def featurize(self, crops) -> np.ndarray:
        return extract_features(crops, self.features)

    def fit(self, crops, y: dict[str, np.ndarray]) -> "BaselineClassifier":
        """y maps target name -> int labels (e.g. df["y_occ"]). Features are computed once."""
        X = self.featurize(crops)
        for t in self.targets:
            self.models[t] = make_estimator(self.classifier).fit(X, np.asarray(y[t]))
        return self

    def predict_proba(self, crops, target: str = "color3") -> np.ndarray:
        """(N, len(OCCUPANCY)) or (N, len(COLOR3)) probabilities for one target.

        Columns follow the class order in labels.OCCUPANCY / COLOR3. Note: this is not the
        (N, 13) model contract; the baseline cannot see piece types.
        If a class was missing from the training data, sklearn has no column for it; we
        re-insert a zero column so the shape is always fixed.
        """
        model = self.models[target]
        p = model.predict_proba(self.featurize(crops))
        out = np.zeros((p.shape[0], len(TARGETS[target][1])), dtype=np.float64)
        out[:, np.asarray(model.classes_, dtype=int)] = p
        return out

    def predict(self, crops, target: str = "color3") -> np.ndarray:
        return self.models[target].predict(self.featurize(crops))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BaselineClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BaselineClassifier):
            raise TypeError(f"{path} does not contain a BaselineClassifier")
        return obj


def evaluate_baseline(clf: BaselineClassifier, crops, y: dict[str, np.ndarray]) -> dict:
    """Accuracy + confusion matrix (rows = true, cols = predicted) per target."""
    from sklearn.metrics import confusion_matrix

    X = clf.featurize(crops)  # features once, shared by all targets
    out = {}
    for t in clf.targets:
        names = TARGETS[t][1]
        y_true = np.asarray(y[t], dtype=int)
        y_pred = np.asarray(clf.models[t].predict(X), dtype=int)
        out[t] = {
            "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
            "n": int(len(y_true)),
            "classes": list(names),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(names)))).tolist(),
        }
    return out


def train_baseline(cfg, out_path: str | Path = "models/baseline/baseline.pkl") -> BaselineClassifier:
    """Fit the baseline on cfg.data.train_sessions, report val accuracy, save to out_path.

    Also writes <out_path without suffix>.json (e.g. models/baseline/baseline.json) with
    train and val accuracy + confusion matrices per target, the sessions used and the
    settings, so M4 / the report can pick the numbers up.
    """
    import json

    from gambit_vision.dataset import read_crop, split_by_session

    splits = split_by_session(cfg)
    train = splits["train"]
    if len(train) == 0:
        raise ValueError("cfg.data.train_sessions is empty")
    b = cfg.baseline
    clf = BaselineClassifier(b.features, b.classifier, tuple(b.targets))
    crops = [read_crop(p) for p in train["abs_path"]]
    y_train = {t: train[TARGETS[t][0]].to_numpy() for t in clf.targets}
    clf.fit(crops, y_train)

    report = {
        "features": clf.features, "classifier": clf.classifier, "targets": list(clf.targets),
        "train_sessions": list(cfg.data.train_sessions or []),
        "val_sessions": list(cfg.data.val_sessions or []),
        "train": evaluate_baseline(clf, crops, y_train),
    }
    val = splits["val"]
    if len(val):
        val_crops = [read_crop(p) for p in val["abs_path"]]
        report["val"] = evaluate_baseline(clf, val_crops, {t: val[TARGETS[t][0]].to_numpy() for t in clf.targets})
        for t, r in report["val"].items():
            print(f"baseline val {t}: accuracy {r['accuracy']:.4f} (n={r['n']})")
    else:
        report["val"] = None
        print("baseline: no val sessions configured, only train accuracy reported")

    out_path = Path(out_path)
    clf.save(out_path)
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return clf
