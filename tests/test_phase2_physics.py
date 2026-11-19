"""Synthetic-signal tests for Phase 2 physics features and order resampling (SYNTHETIC, not results)."""
import numpy as np
import pytest

from pave.features import physics as P
from pave.signal.order import order_resample
from pave.signal.physics import BearingGeometry, orders

FS = 25600.0
G6203 = BearingGeometry("6203", 8, 6.75, 28.55, 0.0)
KIN = {k: v for k, v in orders(G6203).items() if k in ("BPFO", "BPFI")}


def impulses(f_fault, fr, dur=1.0, fs=FS, fres=4000.0, noise=0.3, am_shaft=False, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    x = np.zeros(n)
    ring_t = np.arange(int(0.003 * fs)) / fs
    ring = np.exp(-ring_t * 2500) * np.sin(2 * np.pi * fres * ring_t)
    k = 0
    while True:
        t = k / f_fault
        i0 = int(t * fs)
        if i0 >= n:
            break
        amp = 1 + (0.8 * np.cos(2 * np.pi * fr * t) if am_shaft else 0)
        seg = ring[: n - i0]
        x[i0:i0 + len(seg)] += amp * seg
        k += 1
    return x + noise * rng.standard_normal(n)


def feats(x, fr, reg=None, fs=FS):
    f, E = P.envelope_spectrum_window(x, fs, (1000, 10000))
    return f, E, P.window_features(f, E, fr, KIN, reg or P.Regions())


def test_detects_shifted_or_line_and_harmonics():
    fr = 25.0
    f_true = KIN["BPFO"] * fr * 1.02  # +2 % shift
    f, E, ft = feats(impulses(f_true, fr, dur=2.0), fr)
    assert ft["BPFO_snr1"] > 15
    assert ft["BPFO_nharm"] >= 2
    assert ft["contrast"] > 6
    scan = P.window_line_scan(f, E, fr, KIN)
    assert scan["BPFO_scan_r"] == pytest.approx(0.02, abs=0.006)


def test_detects_ir_with_sidebands():
    fr = 25.0
    f, E, ft = feats(impulses(KIN["BPFI"] * fr * 0.988, fr, dur=2.0, am_shaft=True), fr)
    assert ft["BPFI_snr1"] > 15
    assert ft["BPFI_sb"] > 6
    assert ft["contrast"] < -6


def test_noise_has_no_family_evidence():
    x = np.random.default_rng(3).standard_normal(int(2 * FS))
    _, _, ft = feats(x, 25.0)
    assert ft["BPFO_nharm"] <= 1 and ft["BPFI_nharm"] <= 1
    assert abs(ft["contrast"]) < 6


def test_calibration_uses_only_matching_class_and_min_count():
    rng = np.random.default_rng(0)
    n = 100
    labels = np.array(["OR"] * 50 + ["IR"] * 30 + ["H"] * 20)
    r_o = np.where(labels == "OR", 0.02 + 0.001 * rng.standard_normal(n), rng.uniform(-0.035, 0.035, n))
    r_i = np.where(labels == "IR", -0.012 + 0.001 * rng.standard_normal(n), rng.uniform(-0.035, 0.035, n))
    snr = np.full(n, 20.0)
    reg = P.calibrate_regions({"BPFO": r_o, "BPFI": r_i}, {"BPFO": snr, "BPFI": snr}, labels)
    assert reg.centre["BPFO"] == pytest.approx(0.02, abs=0.002) and not reg.fallback["BPFO"]
    assert reg.centre["BPFI"] == pytest.approx(-0.012, abs=0.002)
    assert reg.half["BPFO"] >= P.CAL_MIN_HALF
    # too few qualifying windows -> fallback to +-3.5 %
    reg2 = P.calibrate_regions({"BPFO": r_o, "BPFI": r_i}, {"BPFO": snr, "BPFI": np.full(n, 5.0)}, labels)
    assert reg2.fallback["BPFI"] and reg2.half["BPFI"] == P.SEARCH_REL


def test_calibrated_region_excludes_wrong_line():
    """A narrow calibrated region at -1.2 % must not pick a stronger line at +3 % inside the wide search."""
    fr = 25.0
    fk = KIN["BPFI"] * fr
    x = impulses(fk * 0.988, fr, dur=2.0, noise=0.3) + 2 * impulses(fk * 1.03, fr, dur=2.0, noise=0.0, fres=6000, seed=1)
    f, E = P.envelope_spectrum_window(x, FS, (1000, 10000))
    wide = P.window_line_scan(f, E, fr, KIN)
    assert wide["BPFI_scan_r"] == pytest.approx(0.03, abs=0.006)
    reg = P.Regions(centre={"BPFO": 0.0, "BPFI": -0.012}, half={"BPFO": 0.035, "BPFI": 0.005})
    ft = P.window_features(f, E, fr, KIN, reg)
    assert ft["BPFI_snr1"] > 10


def test_order_resampling_speed_invariance():
    """The same order content at two speeds maps to the same order spectrum."""
    fs = 25600.0
    spr, nrev = 512, 48
    specs = []
    for fr in (49.9, 50.2):
        t = np.arange(int(1.2 * fs)) / fs
        x = np.sin(2 * np.pi * 3.6 * fr * t) + 0.5 * np.sin(2 * np.pi * 7.2 * fr * t)
        y = order_resample(x, fs, fr, spr, nrev)
        assert y.shape == (spr * nrev,)
        S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        o = np.fft.rfftfreq(len(y), 1.0 / spr)
        specs.append(o[np.argmax(S)])
    assert specs[0] == pytest.approx(3.6, abs=0.03) and specs[1] == pytest.approx(3.6, abs=0.03)


def test_order_resampling_errors():
    with pytest.raises(ValueError):
        order_resample(np.zeros(1000), 25600, 50.0, 512, 48)
    with pytest.raises(ValueError):
        order_resample(np.zeros(100000), 25600, 0.0, 512, 48)


def test_handcrafted_shapes():
    from pave.features.handcrafted import handcrafted, names
    v = handcrafted(np.random.default_rng(0).standard_normal(25600), 25600)
    assert v.shape == (len(names()),) and np.isfinite(v).all()


def test_cnn_forward_shapes():
    import torch
    from pave.models.cnn1d import CNN1D, n_params
    m = CNN1D(2, 3)
    assert m(torch.zeros(4, 2, 25600)).shape == (4, 3)
    assert m(torch.zeros(4, 2, 24576)).shape == (4, 3)
    assert n_params(m) < 1_000_000


def test_metrics_group_level():
    from pave.evaluation.metrics import evaluate
    probs = np.array([[0.9, 0.1, 0], [0.6, 0.4, 0], [0.2, 0.8, 0], [0.1, 0.1, 0.8]])
    y = np.array([0, 1, 1, 2])
    r = evaluate(probs, y, {"bearing": np.array(["a", "b", "b", "c"])}, 3)
    assert r["window"]["acc"] == 0.75 and r["bearing"]["acc"] == 1.0
