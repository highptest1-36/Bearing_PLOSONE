"""Envelope/evidence tests on SYNTHETIC signals (sanity tests, not paper results)."""
import numpy as np
import pytest

from pave.signal import envelope as env
from pave.signal.physics import bpfi, bpfo, orders
from pave.signal.spectral import estimate_shaft_speed

FS = 25600.0


def impulsive_fault(f_fault, fr=50.0, dur=8.0, fres=4000.0, noise=0.5, seed=0, am_shaft=False):
    """Resonance ringing excited at f_fault + shaft tone + white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    x = np.zeros_like(t)
    period = 1.0 / f_fault
    ring_t = np.arange(int(0.004 * FS)) / FS
    ring = np.exp(-ring_t * 1500) * np.sin(2 * np.pi * fres * ring_t)
    for k in range(int(dur / period)):
        i0 = int(k * period * FS)
        amp = 1.0 + (0.8 * np.cos(2 * np.pi * fr * k * period) if am_shaft else 0.0)
        seg = ring[: len(x) - i0]
        x[i0:i0 + len(seg)] += amp * seg
    x += 0.2 * np.sin(2 * np.pi * fr * t)
    x += noise * rng.standard_normal(len(t))
    return x


def _spec(x, band=(2500, 6000)):
    e = env.envelope(x, FS, band)
    return env.envelope_spectrum(e, FS, 1.0)


def test_envelope_detects_bpfo_and_not_bpfi():
    fr = 50.0
    fe, E = _spec(impulsive_fault(bpfo(fr), fr))
    r = env.evidence_against_controls(fe, E, fr, bpfo(fr))
    assert r["peak_ratio"] == pytest.approx(1.0, abs=0.01)
    assert r["excess_db"] > 10
    assert r["ctrl_exceed_frac"] == 0.0
    assert env.evidence_against_controls(fe, E, fr, bpfi(fr))["excess_db"] < 6


def test_envelope_detects_bpfi_with_shaft_sidebands():
    fr = 50.0
    fe, E = _spec(impulsive_fault(bpfi(fr), fr, am_shaft=True))
    assert env.evidence_against_controls(fe, E, fr, bpfi(fr))["excess_db"] > 10
    assert env.evidence_against_controls(fe, E, fr, bpfi(fr) + fr)["excess_db"] > 3


def test_noise_only_has_no_evidence():
    x = np.random.default_rng(1).standard_normal(int(8 * FS))
    fe, E = _spec(x, (1000, 10000))
    for f in (bpfo(50), bpfi(50)):
        r = env.evidence_against_controls(fe, E, 50.0, f)
        assert r["excess_db"] < 6
        assert r["ctrl_exceed_frac"] > 0.05


def test_sk_band_finds_resonance():
    # sparse impulses (30 Hz) so the resonance band is genuinely impulsive
    x = impulsive_fault(30.0, fres=7000.0, noise=0.3)
    band, _, _ = env.spectral_kurtosis_band(x, FS, bw_hz=2000)
    assert band[0] <= 7000 <= band[1]


def test_bandpass_rejects_invalid_band():
    with pytest.raises(ValueError):
        env.bandpass_sos(1000, 13000, FS)
    with pytest.raises(ValueError):
        env.bandpass_sos(5000, 4000, FS)


def test_control_orders_avoid_harmonics_and_fault_orders():
    c = env.CONTROL_ORDERS
    assert len(c) == 24
    assert np.all(np.abs(c - np.round(c)) > 0.1)
    o = orders()
    for key in ("BPFO", "BPFI"):
        for h in (1, 2, 3):
            assert np.all(np.abs(c - h * o[key]) > 0.1)


def test_shaft_speed_estimate():
    t = np.arange(int(20 * FS)) / FS
    fr = 49.89
    x = 0.3 * np.sin(2 * np.pi * fr * t) + 0.5 * np.sin(2 * np.pi * 2 * fr * t + 0.3)
    x += 0.05 * np.random.default_rng(0).standard_normal(len(t))
    r = estimate_shaft_speed(x, FS, nominal_hz=50.17)
    assert r["fr_hz"] == pytest.approx(fr, abs=0.005)
    assert r["f1x_hz"] == pytest.approx(fr, abs=0.005)


def test_family_search_finds_shifted_fault_line():
    """A fault line 2 % above the kinematic BPFO is missed at +-1 % but found by the +-3.5 % search."""
    fr = 50.0
    f_true = bpfo(fr) * 1.02
    fe, E = _spec(impulsive_fault(f_true, fr, dur=16.0), (2500, 6000))
    strict = env.evidence_against_controls(fe, E, fr, bpfo(fr), tol=0.01)
    fam = env.family_search(fe, E, fr, orders()["BPFO"])
    assert abs(fam["f0_hz"] - f_true) <= 1.0
    assert fam["snr_db"] > strict["snr_db"] + 6
    assert fam["h2_snr_db"] > 10


def test_family_search_skips_shaft_harmonics():
    fr = 50.0
    f = np.arange(0, 1200, 0.25)
    E = np.ones_like(f)
    E[np.argmin(np.abs(f - 4 * fr * 0.0 - 5 * fr))] = 100.0  # strong 5x shaft line at 250 Hz
    r = env.family_search(f, E, fr, 5.0, rel_width=0.035)
    assert abs(r["f0_hz"] - 250.0) > 0.9
