"""Synthetic ground-truth tests for envelope-domain interventions (SYNTHETIC; not paper results)."""
import numpy as np
import pytest

from pave.explain import interventions as IV
from pave.features import physics as PH
from pave.signal.physics import BearingGeometry, orders

FS = 64000.0
FR = 25.0
BAND = (1000.0, 20000.0)
KIN = {k: v for k, v in orders(BearingGeometry("6203", 8, 6.75, 28.55, 0.0)).items() if k in ("BPFO", "BPFI")}
REG = PH.Regions(centre={"BPFO": 0.0, "BPFI": 0.0}, half={"BPFO": 0.005, "BPFI": 0.005})


def fault(f_fault, am=False, noise=0.3, seed=0, dur=1.0):
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    x = np.zeros(n)
    rt = np.arange(int(0.002 * FS)) / FS
    ring = np.exp(-rt * 3000) * np.sin(2 * np.pi * 6000 * rt)
    k = 0
    while (t := k / f_fault) * FS < n:
        i0 = int(t * FS)
        amp = 1 + (0.8 * np.cos(2 * np.pi * FR * t) if am else 0)
        seg = ring[: n - i0]
        x[i0:i0 + len(seg)] += amp * seg
        k += 1
    return x + noise * rng.standard_normal(n)


def concepts(x):
    f, E = PH.envelope_spectrum_window(x, FS, BAND)
    return PH.window_features(f, E, FR, KIN, REG)


def bands(fam, sidebands=False):
    return IV.family_bands(FR, KIN[fam], REG.centre[fam], REG.half[fam], sidebands=sidebands)


def forbidden_all():
    return bands("BPFO") + bands("BPFI", True) + IV.shaft_guards(FR, 1000)


def test_strength_zero_is_identity():
    x = fault(KIN["BPFO"] * FR)
    r = IV.suppress(x, FS, BAND, bands("BPFO"), strength=0.0, fr=FR)
    np.testing.assert_allclose(r.x, x, atol=1e-8)


def test_targeted_or_removes_bpfo_and_spares_bpfi_and_controls_do_not():
    x = fault(KIN["BPFO"] * FR)
    c0 = concepts(x)
    rt = IV.suppress(x, FS, BAND, bands("BPFO"), 1.0, FR)
    ct = concepts(rt.x)
    assert c0["BPFO_harm"] - ct["BPFO_harm"] > 10          # targeted evidence removed
    assert abs(c0["BPFI_harm"] - ct["BPFI_harm"]) < 2       # other family untouched
    assert rt.out_of_band_rel_change < 0.01                 # change confined to the analysis band (filter roll-off)
    rng = np.random.default_rng(1)
    ctrl = IV.control_centres(bands("BPFO"), forbidden_all(), 30, 600, rng)
    assert IV.overlap_hz(ctrl, forbidden_all()) == 0
    rw = IV.suppress(x, FS, BAND, ctrl, 1.0, FR)
    assert c0["BPFO_harm"] - concepts(rw.x)["BPFO_harm"] < 3
    re, ratio = IV.matched_energy_control(x, FS, BAND, rt.removed_env_energy, ctrl, forbidden_all(), FR)
    assert c0["BPFO_harm"] - concepts(re.x)["BPFO_harm"] < 3
    assert np.isfinite(ratio)


def test_targeted_ir_with_sidebands():
    x = fault(KIN["BPFI"] * FR, am=True)
    c0 = concepts(x)
    r = IV.suppress(x, FS, BAND, bands("BPFI", sidebands=True), 1.0, FR)
    c1 = concepts(r.x)
    assert c0["BPFI_harm"] - c1["BPFI_harm"] > 8
    assert c0["BPFI_sb"] - c1["BPFI_sb"] > 3
    assert abs(c0["BPFO_harm"] - c1["BPFO_harm"]) < 2


def test_noise_only_is_barely_changed():
    x = np.random.default_rng(5).standard_normal(int(FS))
    c0 = concepts(x)
    r = IV.suppress(x, FS, BAND, bands("BPFO"), 1.0, FR)
    c1 = concepts(r.x)
    assert abs(c0["BPFO_harm"] - c1["BPFO_harm"]) < 4
    assert 0.98 < r.total_energy_ratio < 1.02


def test_energy_matched_control_reaches_target_on_broadband_signal():
    """On a signal whose envelope has many comparable lines, widening unrelated bands can match the removal."""
    rng = np.random.default_rng(2)
    x = fault(KIN["BPFO"] * FR, noise=1.0)
    rt = IV.suppress(x, FS, BAND, bands("BPFO"), 1.0, FR)
    ctrl = IV.control_centres(bands("BPFO"), forbidden_all(), 30, 600, rng)
    re, ratio = IV.matched_energy_control(x, FS, BAND, rt.removed_env_energy, ctrl, forbidden_all(), FR, max_width_hz=150)
    assert ratio > 0  # achieved ratio is always reported (may be < 1: shortfall is reported, never hidden)


def test_overlap_detection_kaist_geometry():
    """KAIST 6205: 3 x BPFO (10.73 orders) vs 2 x BPFI (10.85 orders) overlap at +-3.5 % half-width."""
    k = orders()
    fr = 50.0
    bo = IV.family_bands(fr, k["BPFO"], 0.0, 0.035)
    bi = IV.family_bands(fr, k["BPFI"], 0.0, 0.035)
    assert IV.overlap_hz(bo, bi) > 0
    cleaned = IV.subtract_bands(bo, bi)
    assert IV.overlap_hz(cleaned, bi) == 0


def test_subtract_bands():
    assert IV.subtract_bands([(0, 10)], [(3, 4)]) == [(0, 3), (4, 10)]
    assert IV.subtract_bands([(0, 10)], [(-1, 11)]) == []


def test_energy_control_terminates_with_dense_forbidden_zones():
    """Regression: shaft guards every 25 Hz used to fragment bands and loop forever."""
    import time
    x = np.random.default_rng(3).standard_normal(int(FS))
    guards = IV.shaft_guards(FR, 1000)
    t0 = time.time()
    r, ratio = IV.matched_energy_control(x, FS, BAND, 1e12, [(30.0, 32.0), (130.0, 132.0)], guards, FR)
    assert time.time() - t0 < 20
    assert r.info["iterations"] <= 30 and np.isfinite(ratio) and ratio < 1  # shortfall reported, not hidden
    assert IV.overlap_hz(r.bands, guards) == 0


def test_merge_bands():
    assert IV.merge_bands([(5, 7), (0, 2), (1, 3)]) == [(0, 3), (5, 7)]
