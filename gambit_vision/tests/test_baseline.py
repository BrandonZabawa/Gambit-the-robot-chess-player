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


def test_hsv_histogram():
    red = np.zeros((20, 10, 3), np.uint8)
    red[..., 0] = 255
    h = baseline.hsv_histogram(red, bins=(8, 4, 4))
    assert h.shape == (128,) and np.isclose(h.sum(), 1)
    assert np.isclose(h.max(), 1)  # all pixels in one (hue, sat, val) bin
    noisy = np.random.default_rng(0).integers(0, 256, (20, 10, 3), dtype=np.uint8)
    assert np.isclose(baseline.hsv_histogram(noisy).sum(), 1)


def test_edge_density():
    flat = np.full((40, 40, 3), 128, np.uint8)
    e = baseline.edge_density(flat, grid=2)
    assert e.shape == (5,) and np.allclose(e, 0)
    yy, xx = np.mgrid[0:40, 0:40]
    checker = (((yy // 4 + xx // 4) % 2) * 255).astype(np.uint8)
    checker = np.stack([checker] * 3, axis=-1)
    e = baseline.edge_density(checker, grid=2)
    assert e[0] > 0.15 and (e[1:] > 0.15).all()
    assert baseline.edge_density(checker, grid=3).shape == (10,)


def test_lbp_histogram():
    const = np.full((20, 10, 3), 77, np.uint8)
    h = baseline.lbp_histogram(const, points=8, radius=1)
    assert h.shape == (10,) and np.isclose(h.sum(), 1)
    assert np.isclose(h[8], 1)  # every neighbour >= centre -> all ones -> code P
    noisy = np.random.default_rng(1).integers(0, 256, (30, 30, 3), dtype=np.uint8)
    hn = baseline.lbp_histogram(noisy)
    assert np.isclose(hn.sum(), 1) and hn[9] > 0  # random texture has non-uniform patterns
    # A vertical step edge: pixels next to it see a half-circle of brighter neighbours.
    step = np.zeros((10, 10), np.uint8)
    step[:, 5:] = 200
    hs = baseline.lbp_histogram(step)
    assert hs[8] < 1 and hs[9] == 0  # straight edges are uniform patterns


def test_extract_features_real_features():
    crops = [np.random.default_rng(i).integers(0, 256, (200, 100, 3), dtype=np.uint8) for i in range(3)]
    X = baseline.extract_features(crops, ["hsv_hist", "edge_density", "lbp"])
    assert X.shape == (3, 128 + 5 + 10) and X.dtype == np.float32


def test_predict_proba_missing_class(fake_feature):
    crops, y = fake_data()
    keep = y["color3"] != 2  # no black pieces in training data
    crops = [c for c, k in zip(crops, keep) if k]
    y = {k: v[keep] for k, v in y.items()}
    clf = BaselineClassifier(["mean_rgb"], classifier="logistic").fit(crops, y)
    p = clf.predict_proba(crops[:4], target="color3")
    assert p.shape == (4, 3) and np.allclose(p[:, 2], 0) and np.allclose(p.sum(axis=1), 1)


def test_train_baseline_end_to_end(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import json
    from types import SimpleNamespace

    crops, y = fake_data(n=48)
    for sess, part in (("s01", slice(0, 36)), ("s02", slice(36, 48))):
        sdir = tmp_path / "crops" / sess
        (sdir / "squares").mkdir(parents=True)
        lines = ["session_id,game_id,move_idx,frame_id,square,label,path,fen"]
        for i, (c, lab) in enumerate(zip(crops[part], y["color3"][part])):
            label = {0: "empty", 1: "P", 2: "p"}[int(lab)]
            rel = f"squares/{i:06d}_a1.png"
            cv2.imwrite(str(sdir / rel), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
            lines.append(f"{sess},g1,0,{i:06d},a1,{label},{rel},x")
        (sdir / "manifest.csv").write_text("\n".join(lines) + "\n")
    cfg = SimpleNamespace(
        data=SimpleNamespace(crops_root=str(tmp_path / "crops"), train_sessions=["s01"],
                             val_sessions=["s02"], test_sessions=[]),
        baseline=SimpleNamespace(features=["hsv_hist", "edge_density", "lbp"],
                                 classifier="logistic", targets=["occupancy", "color3"]))
    out = tmp_path / "models" / "baseline.pkl"
    clf = baseline.train_baseline(cfg, out_path=out)
    assert out.exists() and isinstance(BaselineClassifier.load(out), BaselineClassifier)
    report = json.loads(out.with_suffix(".json").read_text())
    for t in ("occupancy", "color3"):
        assert report["val"][t]["n"] == 12
        assert report["val"][t]["accuracy"] > 0.9
    assert np.array(report["val"]["color3"]["confusion_matrix"]).shape == (3, 3)
    assert clf.predict_proba(crops[:2], "occupancy").shape == (2, 2)
