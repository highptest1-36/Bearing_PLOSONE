"""Concept V3 synthetic tests (SYNTHETIC ground truth; not paper results)."""
import numpy as np
import pytest

from pave.features import physics_v3 as V3
from pave.features.physics import Regions, envelope_spectrum_window
from pave.signal.envelope import envelope
from pave.signal.physics import BearingGeometry, orders

FS = 64000.0
FAG = {k: v for k, v in orders(BearingGeometry("FAG", 8, 6.75, 28.55, 0)).items() if k in ("BPFO", "BPFI")}
IBU = {k: v for k, v in orders(BearingGeometry("IBU", 8, 6.75, 29.05, 0)).items() if k in ("BPFO", "BPFI")}
REG = Regions(centre={"BPFO": 0.0, "BPFI": 0.0}, half={"BPFO": 0.005, "BPFI": 0.005})


def impulses(f_fault, fr, dur, noise=0.3, am=False, seed=0, shaft_amp=0.0):
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    x = np.zeros(n)
    rt = np.arange(int(0.002 * FS)) / FS
    ring = np.exp(-rt * 3000) * np.sin(2 * np.pi * 6000 * rt)
    k = 0
    while (t := k / f_fault) * FS < n:
        i0 = int(t * FS)
        a = 1 + (0.8 * np.cos(2 * np.pi * fr * t) if am else 0)
        seg = ring[: n - i0]
        x[i0:i0 + len(seg)] += a * seg
        k += 1
    tt = np.arange(n) / FS
    # shaft-rate modulation of the carrier band creates envelope lines at k*fr
    x *= 1 + shaft_amp * np.cos(2 * np.pi * fr * tt)
    return x + noise * rng.standard_normal(n)


def spec(x, dur):
    e = envelope(x, FS, (1000, 20000))
    n = len(e)
    w = np.hanning(n)
    E = np.abs(np.fft.rfft(e * w)) * 2 / w.sum()
    f = np.fft.rfftfreq(n, 1 / FS)
    keep = f <= 1000
    return f[keep], E[keep]


def test_or_detection_valid_and_no_nan():
    fr = 25.0
    f, E = spec(impulses(FAG["BPFO"] * fr, fr, 4.0), 4.0)
    r = V3.features_v3(f, E, fr, FAG, REG)
    assert r.info["valid_BPFO"]
    assert r.c[V3.CONCEPTS_V3.index("BPFO_harm")] > 15
    assert r.c[V3.CONCEPTS_V3.index("contrast")] > 6
    assert np.isfinite(r.c).all() and np.isfinite(r.q).all()
    assert len(r.c) == 8 and len(r.m) == 8 and len(r.q) == 7
    assert "BPFO_sb" not in V3.CONCEPTS_V3


def test_ir_sidebands():
    fr = 25.0
    f, E = spec(impulses(FAG["BPFI"] * fr, fr, 4.0, am=True), 4.0)
    r = V3.features_v3(f, E, fr, FAG, REG)
    assert r.c[V3.CONCEPTS_V3.index("BPFI_sb")] > 6
    assert r.c[V3.CONCEPTS_V3.index("contrast")] < -6


def test_collision_fag_900rpm_invalid_at_1s_and_4s():
    """FAG BPFI at 900 rpm sits 0.81 Hz from 5 fr: not identifiable at 1 s, and STILL not at 4 s
    (region half-width 0.37 Hz + guard 0.25 Hz + 1 bin > 0.81 Hz) - longer records do not automatically resolve it."""
    fr = 15.0
    for dur in (1.0, 4.0):
        df = 1.0 / dur
        regs = V3.family_regions(fr, df, 0.0, FAG["BPFI"], 0.0, 0.005)
        assert V3.identifiability(fr, df, 0.0, regs[1]) < 1


def test_collision_ibu_900rpm_resolved_only_at_4s():
    fr = 15.0
    d1 = V3.identifiability(fr, 1.0, 0.0, V3.family_regions(fr, 1.0, 0.0, IBU["BPFI"], 0.0, 0.005)[1])
    d4 = V3.identifiability(fr, 0.25, 0.0, V3.family_regions(fr, 0.25, 0.0, IBU["BPFI"], 0.0, 0.005)[1])
    assert d1 < 1 <= d4


def test_speed_uncertainty_can_invalidate():
    fr = 25.0
    regs0 = V3.family_regions(fr, 0.25, 0.0, IBU["BPFI"], 0.0, 0.005)
    regs1 = V3.family_regions(fr, 0.25, 0.1, IBU["BPFI"], 0.0, 0.005)
    assert V3.identifiability(fr, 0.25, 0.0, regs0[1]) >= 1
    assert V3.identifiability(fr, 0.25, 0.1, regs1[1]) < 1


def test_invalid_family_is_masked_with_zero_fill():
    fr = 15.0
    f, E = spec(impulses(FAG["BPFI"] * fr, fr, 1.0, shaft_amp=0.5), 1.0)
    r = V3.features_v3(f, E, fr, FAG, REG)
    for k in ("BPFI_snr1", "BPFI_harm", "BPFI_nharm", "BPFI_sb", "contrast"):
        j = V3.CONCEPTS_V3.index(k)
        assert r.m[j] == 0 and r.c[j] == 0.0
    assert r.q[V3.RELIAB_V3.index("ident_BPFI")] < 1


def test_4s_resolves_nearby_shaft_line_on_synthetic_signal():
    """IBU BPFI at 900 rpm with strong shaft modulation: masked at 1 s, measured (and strong) at 4 s."""
    fr = 15.0
    x = impulses(IBU["BPFI"] * fr, fr, 4.0, shaft_amp=0.5)
    f1, E1 = spec(x[: int(FS)], 1.0)
    f4, E4 = spec(x, 4.0)
    r1 = V3.features_v3(f1, E1, fr, IBU, REG)
    r4 = V3.features_v3(f4, E4, fr, IBU, REG)
    assert not r1.info["valid_BPFI"]
    assert r4.info["valid_BPFI"] and r4.c[V3.CONCEPTS_V3.index("BPFI_harm")] > 10


def test_overlapping_harmonics_excluded_kaist_geometry():
    kin = {k: v for k, v in orders().items() if k in ("BPFO", "BPFI")}  # 6205: 3 BPFO ~ 2 BPFI
    fr = 50.0
    f = np.arange(0, 1000.25, 0.25)
    E = np.ones_like(f)
    wide = Regions(centre={"BPFO": 0.0, "BPFI": 0.0}, half={"BPFO": 0.02, "BPFI": 0.02})
    r = V3.features_v3(f, E, fr, kin, wide)
    assert r.q[V3.RELIAB_V3.index("n_excluded")] >= 2


def test_noise_no_evidence():
    x = np.random.default_rng(4).standard_normal(int(FS))
    f, E = spec(x, 1.0)
    r = V3.features_v3(f, E, 25.0, FAG, REG)
    assert r.c[V3.CONCEPTS_V3.index("BPFO_nharm")] <= 1 and r.c[V3.CONCEPTS_V3.index("BPFI_nharm")] <= 1


def test_1s_regions_not_identifiable_for_6203_at_1500rpm():
    """Protocol V2.3 finding: at df = 1 Hz no 6203 family is identifiable (lines 1.4-1.8 Hz from 3fr / 5fr)."""
    for kin in (FAG, IBU):
        for F in ("BPFO", "BPFI"):
            reg = V3.family_regions(25.0, 1.0, 0.0, kin[F], 0.0, 0.005)[1]
            assert V3.identifiability(25.0, 1.0, 0.0, reg) < 1


def test_scan_v3_finds_shift():
    fr = 25.0
    f, E = spec(impulses(FAG["BPFO"] * fr * 1.02, fr, 1.0), 1.0)
    s = V3.scan_v3(f, E, fr, FAG)
    assert s["BPFO_scan_r"] == pytest.approx(0.02, abs=0.006)


def test_standardise_uses_train_valid_entries_only():
    C = np.array([[1.0], [3.0], [100.0], [0.0]])
    M = np.array([[1.0], [1.0], [1.0], [0.0]])
    Z, mu, sd = V3.standardise(C, M, np.array([0, 1, 3]))
    assert mu[0] == 2.0 and sd[0] == 1.0
    assert Z[3, 0] == 0.0 and np.isfinite(Z).all()


def test_line_identifiability_v24():
    """V2.4: a strong line 1.36 Hz from 3fr is identifiable at 4 s (df 0.25) but not at 1 s (df 1)."""
    fr = 25.0
    x = impulses(FAG["BPFO"] * fr, fr, 4.0)
    f4, E4 = spec(x, 4.0)
    wide = Regions(centre={"BPFO": 0.0, "BPFI": 0.0}, half={"BPFO": 0.017, "BPFI": 0.013})
    r4 = V3.features_v3(f4, E4, fr, FAG, wide)
    assert r4.info["valid_BPFO"] and r4.c[V3.CONCEPTS_V3.index("BPFO_harm")] > 10
    assert r4.info["delta_region_BPFO"] < 1 <= r4.info["delta_BPFO"]
    f1, E1 = spec(x[: int(FS)], 1.0)
    r1 = V3.features_v3(f1, E1, fr, FAG, wide)
    assert not r1.info["valid_BPFO"]


def test_shaft_leakage_tail_not_identifiable():
    """Pure shaft-harmonic envelope lines: the region's maximum is a leakage tail next to a guard -> not identifiable."""
    fr = 25.0
    f = np.arange(0, 1000.25, 0.25)
    E = np.ones_like(f) * 0.01
    for k in range(1, 40):
        E += 1.0 * np.exp(-0.5 * ((f - k * fr) / 0.3) ** 2)
    wide = Regions(centre={"BPFO": 0.0, "BPFI": 0.0}, half={"BPFO": 0.017, "BPFI": 0.013})
    r = V3.features_v3(f, E, fr, FAG, wide)
    assert not r.info["valid_BPFO"] and not r.info["valid_BPFI"]
