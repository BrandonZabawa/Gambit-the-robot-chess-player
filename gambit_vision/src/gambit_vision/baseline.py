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
def hsv_histogram(crop: np.ndarray, bins: tuple[int, int, int] = (8, 4, 4)) -> np.ndarray:
    """Normalized 3D HSV histogram of an RGB crop, flattened (prod(bins),).

    Hint: cv2.cvtColor(..., COLOR_RGB2HSV), cv2.calcHist, divide by the pixel count.
    TODO M3: maybe only use the lower part of the crop (the square itself) for occupancy and
    the upper part (the piece body) for color.
    """
    raise NotImplementedError("TODO M3")


def edge_density(crop: np.ndarray, low: int = 50, high: int = 150, grid: int = 2) -> np.ndarray:
    """Fraction of Canny edge pixels, overall and per cell of a grid x grid split
    (1 + grid*grid,). Empty squares have few edges; pieces have many.
    """
    raise NotImplementedError("TODO M3")


def lbp_histogram(crop: np.ndarray, points: int = 8, radius: int = 1) -> np.ndarray:
    """Histogram of uniform local binary patterns on the gray crop (points + 2,).

    Either skimage.feature.local_binary_pattern (adds scikit-image dependency) or a small
    numpy implementation. TODO M3: pick one.
    """
    raise NotImplementedError("TODO M3")


FEATURES = {"hsv_hist": hsv_histogram, "edge_density": edge_density, "lbp": lbp_histogram}


def extract_features(crops, features: list[str]) -> np.ndarray:
    """Stack the named features for each crop -> (N, D) float32."""
    unknown = set(features) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}; choose from {sorted(FEATURES)}")
    rows = [np.concatenate([np.ravel(FEATURES[f](c)) for f in features]) for c in crops]
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------- classifier
def make_estimator(kind: str):
    """StandardScaler + the chosen sklearn classifier, with predict_proba enabled."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # TODO M3: tune hyperparameters on the val sessions (C / gamma, n_estimators, ...).
    models = {
        # SVC(probability=True) is deprecated in recent sklearn; calibrate explicitly instead.
        "svm": lambda: CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced"), ensemble=False),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced"),
        "logistic": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
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
        TODO M3 (optional): if a class is missing from training data, sklearn drops its column;
        re-insert zeros using self.models[target].classes_.
        """
        return self.models[target].predict_proba(self.featurize(crops))

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


def train_baseline(cfg, out_path: str | Path = "models/baseline/baseline.pkl") -> BaselineClassifier:
    """Fit the baseline on cfg.data.train_sessions, report val accuracy, save to out_path."""
    from gambit_vision.dataset import read_crop, split_by_session

    splits = split_by_session(cfg)
    train = splits["train"]
    if len(train) == 0:
        raise ValueError("cfg.data.train_sessions is empty")
    b = cfg.baseline
    clf = BaselineClassifier(b.features, b.classifier, tuple(b.targets))
    crops = [read_crop(p) for p in train["abs_path"]]
    clf.fit(crops, {t: train[TARGETS[t][0]].to_numpy() for t in clf.targets})

    # TODO M3: evaluate on splits["val"] per target (accuracy, confusion matrix) and write
    # the numbers somewhere evaluate.py can pick them up (results/<run_name>/baseline.json).
    clf.save(out_path)
    return clf
