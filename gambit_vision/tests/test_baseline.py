import numpy as np
import pytest

from gambit_vision import baseline
from gambit_vision.baseline import BaselineClassifier

pytest.importorskip("sklearn")


def mean_rgb(crop):
    return crop.reshape(-1, 3).mean(axis=0) / 255.0


def fake_data(n=60, seed=0):
    """Dark crops = empty, bright = white piece, red = black piece (stand-in for real data)."""
    rng = np.random.default_rng(seed)
    crops, color3 = [], []
    for i in range(n):
        c = i % 3
        base = {0: (20, 20, 20), 1: (230, 230, 230), 2: (200, 30, 30)}[c]
        img = np.clip(np.array(base) + rng.normal(0, 10, (20, 10, 3)), 0, 255).astype(np.uint8)
        crops.append(img)
        color3.append(c)
    color3 = np.array(color3)
    return crops, {"color3": color3, "occupancy": (color3 > 0).astype(int)}


@pytest.fixture
def fake_feature(monkeypatch):
    monkeypatch.setitem(baseline.FEATURES, "mean_rgb", mean_rgb)


def test_unknown_feature_and_target():
    with pytest.raises(ValueError):
        baseline.extract_features([np.zeros((4, 4, 3), np.uint8)], ["nope"])
    with pytest.raises(ValueError):
        BaselineClassifier(["hsv_hist"], targets=("pieces",))
    with pytest.raises(ValueError):
        baseline.make_estimator("xgboost")


@pytest.mark.parametrize("kind", ["svm", "random_forest", "logistic"])
def test_fit_predict_save_load(fake_feature, tmp_path, kind):
    crops, y = fake_data()
    clf = BaselineClassifier(["mean_rgb"], classifier=kind).fit(crops, y)
    p = clf.predict_proba(crops[:6], target="color3")
    assert p.shape == (6, 3) and np.allclose(p.sum(axis=1), 1)
    assert clf.predict_proba(crops[:6], target="occupancy").shape == (6, 2)
    assert (clf.predict(crops, target="color3") == y["color3"]).mean() > 0.9

    clf.save(tmp_path / "b.pkl")
    clf2 = BaselineClassifier.load(tmp_path / "b.pkl")
    assert np.array_equal(clf2.predict(crops, "occupancy"), clf.predict(crops, "occupancy"))


@pytest.mark.skip(reason="TODO M3")
def test_hsv_histogram():
    """Shape == prod(bins), sums to 1; a pure-red crop puts all mass in one hue bin."""


@pytest.mark.skip(reason="TODO M3")
def test_edge_density():
    """Flat crop -> all zeros; checkerboard pattern -> high density; shape 1 + grid*grid."""


@pytest.mark.skip(reason="TODO M3")
def test_lbp_histogram():
    """Shape points + 2, sums to 1; constant image puts all mass in one bin."""


@pytest.mark.skip(reason="TODO M3")
def test_train_baseline_end_to_end():
    """Tiny fake session with real crop PNGs -> train_baseline(cfg) writes the pickle and
    val accuracy is reported."""
