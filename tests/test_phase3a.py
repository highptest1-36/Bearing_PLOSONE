"""Phase 3A: nested folds, training-only calibration, temperature scaling, B0 rule, selection, curve/refit consistency."""
import numpy as np
import pandas as pd
import pytest
import torch

from pave.calibration.temperature import apply_temperature, fit_temperature
from pave.experiments import phase3a as P3
from pave.features import physics as PH


def test_outer_inner_folds_nested_and_disjoint():
    outs = P3.outer_folds()
    assert len(outs) == 5
    tested = []
    for o in outs:
        P3.check_nested(o)
        assert len(o["test"]) == 3 and len(o["train"]) == 12
        inn = P3.inner_folds(o)
        assert len(inn) == 4
        for i in inn:
            assert len(i["train"]) == 9 and len(i["val"]) == 3
            assert not ((i["train"] | i["val"]) & o["test"])
        tested += sorted(o["test"])
    assert len(set(tested)) == 15


def test_check_nested_detects_leak():
    o = dict(P3.outer_folds()[0])
    o["train"] = o["train"] | {"K001"}  # K001 is an outer-test bearing of O1
    with pytest.raises(AssertionError):
        P3.check_nested(o)


class _FakeDS:
    """Two bearings: 'A' (OR, line +2 %) and 'B' (OR, line -2 %); envelope spectra are synthetic."""

    def __init__(self):
        n = 40
        f = np.arange(1001, dtype=float)
        env = np.ones((2 * n, 1, 1001), dtype=np.float32)
        fr = 50.0
        rows = []
        for i in range(2 * n):
            b = "A" if i < n else "B"
            shift = 1.02 if b == "A" else 0.98
            env[i, 0, int(round(3.0 * fr * shift))] = 50.0  # 153 Hz / 147 Hz
            rows.append({"bearing": b, "label3": "OR", "fr_hz": fr})
        self.meta = pd.DataFrame(rows)
        self.env = env
        self.f = f


def _scan(ds, kin):
    rows = []
    for i in range(len(ds.meta)):
        s = PH.window_line_scan(ds.f, ds.env[i, 0].astype(float), 50.0, {"BPFO": kin[i, 0], "BPFI": kin[i, 1]})
        rows.append(s)
    return pd.DataFrame(rows)


def test_physics_cache_calibrates_only_on_given_training_bearings():
    ds = _FakeDS()
    kin = np.tile([3.0, 5.0], (len(ds.meta), 1))
    cache = P3.PhysicsCache(ds, kin, _scan(ds, kin), np.arange(len(ds.meta)))
    _, regA = cache.features(frozenset({"A"}))
    _, regB = cache.features(frozenset({"B"}))
    assert regA.centre["BPFO"] == pytest.approx(0.02, abs=0.005)
    assert regB.centre["BPFO"] == pytest.approx(-0.02, abs=0.005)


def test_temperature_scaling_reduces_nll_on_overconfident_probs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3))
    logits[np.arange(2000), y] += 1.0
    P = np.exp(4 * logits)
    P /= P.sum(1, keepdims=True)
    T = fit_temperature(P, y)
    assert T > 1.5
    Q = apply_temperature(P, T)
    nll = lambda A: -np.mean(np.log(A[np.arange(len(y)), y]))  # noqa: E731
    assert nll(Q) < nll(P)
    assert (Q.argmax(1) == P.argmax(1)).all()


def test_b0_rule():
    F = np.zeros((6, len(PH.FEATURE_NAMES)))
    io, ii, ic = PH.FEATURE_NAMES.index("BPFO_harm"), PH.FEATURE_NAMES.index("BPFI_harm"), PH.FEATURE_NAMES.index("contrast")
    F[:, io] = [1, 2, 20, 3, 18, 2]
    F[:, ii] = [1, 1, 2, 20, 3, 25]
    F[:, ic] = F[:, io] - F[:, ii]
    y = np.array([0, 0, 2, 1, 2, 1])
    tau = P3.b0_fit(F, y)
    assert 2 < tau <= 18
    assert list(P3.b0_predict(F, tau).argmax(1)) == [0, 0, 2, 1, 2, 1]


def test_selection_prefers_simpler_on_ties():
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rp3a", Path(__file__).resolve().parents[1] / "scripts" / "run_phase3a.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rp3a"] = mod
    spec.loader.exec_module(mod)
    k, _ = mod.select({("simple",): [0.80, 0.80], ("complex",): [0.803, 0.803]})
    assert k == ("simple",)
    k, _ = mod.select({("simple",): [0.80, 0.80], ("complex",): [0.85, 0.85]})
    assert k == ("complex",)


def test_fit_fixed_reproduces_curve_epoch():
    from pave.models.mlp import ConceptMLP
    from pave.training.curves import fit_curve, fit_fixed
    from pave.training.engine import predict
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 9)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    Xv = rng.normal(size=(50, 9)).astype(np.float32)
    r = fit_curve(lambda: ConceptMLP(9, 3, 16), X, y, Xv, 3, 1e-3, 1e-2, 64, 10, 7, "cpu", amp=False)
    m = fit_fixed(lambda: ConceptMLP(9, 3, 16), X, y, 3, 1e-3, 1e-2, 64, 6, 7, "cpu", amp=False, sched_epochs=10)["model"]
    np.testing.assert_allclose(predict(m, Xv, "cpu"), r["val_probs"][6], atol=1e-5)
    # a different seed must change the initialisation (seed controls weights)
    m2 = fit_fixed(lambda: ConceptMLP(9, 3, 16), X, y, 3, 1e-3, 1e-2, 64, 6, 8, "cpu", amp=False, sched_epochs=10)["model"]
    assert not np.allclose(predict(m2, Xv, "cpu"), r["val_probs"][6])


def test_bearing_metrics_and_bootstrap():
    P = np.array([[0.9, 0.05, 0.05]] * 4 + [[0.1, 0.8, 0.1]] * 4 + [[0.1, 0.2, 0.7]] * 4)
    y = np.array([0] * 4 + [1] * 4 + [2] * 4)
    b = np.repeat(["a", "b", "c", "d", "e", "f"], 2)
    bt = P3.bearing_table(P, y, b)
    m = P3.bearing_metrics(bt)
    assert m["n_bearings"] == 6 and m["macro_f1"] == 1.0
    lo, hi = P3.stratified_bootstrap(bt, lambda d: float((d.pred == d.y).mean()), n=200)
    assert lo == hi == 1.0
    r = P3.mcnemar_exact(np.array([True] * 6), np.array([False] * 6))
    assert r["a_only_correct"] == 6 and r["exact_p"] < 0.05
