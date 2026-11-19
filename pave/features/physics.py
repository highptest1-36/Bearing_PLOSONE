"""Frozen physics-feature definitions (docs/RESEARCH_PROTOCOL_V2.md §6) with source calibration.

Workflow per fold:
  1. ``window_line_scan``     - label-free: strongest line in +-3.5 % around each kinematic order (post-hoc
                                 width from KAIST), with SNR -> relative offset r = f0/f_kin - 1.
  2. ``calibrate_regions``    - SOURCE ONLY: robust centre/width of r from TRAIN windows of the matching class.
  3. ``window_features``      - per window: family SNR, harmonic mean SNR, harmonic count, sideband SNR,
                                 family contrast, inside the calibrated regions.
No test-fold information is used in step 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pave.signal.envelope import envelope

SEARCH_REL = 0.035          # +-3.5 % search width (POST-HOC on KAIST; frozen before PU)
CAL_MIN_SNR_DB = 10.0
CAL_MIN_WINDOWS = 20
CAL_MIN_HALF = 0.005
CAL_MAD_K = 3.0
HARM_TOL_REL = 0.005
HARM_DET_DB = 6.0
RING_HZ = 20.0
FAMILIES = ("BPFO", "BPFI")
FEATURE_NAMES = [f"{f}_{k}" for f in FAMILIES for k in ("snr1", "harm", "nharm", "sb")] + ["contrast"]


def envelope_spectrum_window(x: np.ndarray, fs: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Single-window Hann envelope amplitude spectrum (resolution 1/len)."""
    e = envelope(x, fs, band)
    n = len(e)
    w = np.hanning(n)
    E = np.abs(np.fft.rfft(e * w)) * 2.0 / w.sum()
    return np.fft.rfftfreq(n, 1.0 / fs), E


def _shaft_mask(f: np.ndarray, fr: float, guard_hz: float = 1.0) -> np.ndarray:
    h = np.round(f / fr)
    return (np.abs(f - h * fr) <= guard_hz) & (h >= 1)


def _snr_at(f: np.ndarray, E: np.ndarray, fc: float, half: float, fr: float, exclude_peak: bool = True) -> tuple[float, float]:
    """Max in [fc-half, fc+half] (no shaft bins) over median of a +-RING_HZ ring. Returns (snr_db, f_peak)."""
    df = f[1] - f[0]
    half = max(half, df)
    shaft = _shaft_mask(f, fr)
    band = (f >= fc - half) & (f <= fc + half) & ~shaft
    if not band.any():
        return np.nan, np.nan
    i = int(np.flatnonzero(band)[np.argmax(E[band])])
    ring = (f >= fc - half - RING_HZ) & (f <= fc + half + RING_HZ) & ~shaft
    if exclude_peak:
        ring[max(i - 2, 0): i + 3] = False
    ring &= ~band
    if ring.sum() < 3:
        return np.nan, float(f[i])
    floor = float(np.median(E[ring]))
    return (20 * np.log10(E[i] / floor) if floor > 0 and E[i] > 0 else np.nan), float(f[i])


def window_line_scan(f: np.ndarray, E: np.ndarray, fr: float, kin_orders: dict[str, float]) -> dict[str, float]:
    """Label-free line search around kinematic orders. Returns r (relative offset) and SNR per family."""
    out = {}
    for fam in FAMILIES:
        fk = kin_orders[fam] * fr
        snr, fp = _snr_at(f, E, fk, SEARCH_REL * fk, fr)
        out[f"{fam}_scan_snr"] = snr
        out[f"{fam}_scan_r"] = fp / fk - 1.0 if np.isfinite(fp) else np.nan
    return out


@dataclass
class Regions:
    centre: dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FAMILIES})
    half: dict[str, float] = field(default_factory=lambda: {f: SEARCH_REL for f in FAMILIES})
    n_used: dict[str, int] = field(default_factory=lambda: {f: 0 for f in FAMILIES})
    fallback: dict[str, bool] = field(default_factory=lambda: {f: True for f in FAMILIES})

    def to_dict(self) -> dict:
        return {"centre": self.centre, "half": self.half, "n_used": self.n_used, "fallback": self.fallback}


def calibrate_regions(scan_r: dict[str, np.ndarray], scan_snr: dict[str, np.ndarray], labels3: np.ndarray) -> Regions:
    """SOURCE-ONLY calibration. ``labels3`` must be TRAIN labels; OR windows calibrate BPFO, IR windows BPFI."""
    reg = Regions()
    for fam, lab in (("BPFO", "OR"), ("BPFI", "IR")):
        r = np.asarray(scan_r[fam], float)
        s = np.asarray(scan_snr[fam], float)
        ok = (np.asarray(labels3) == lab) & np.isfinite(r) & np.isfinite(s) & (s >= CAL_MIN_SNR_DB)
        reg.n_used[fam] = int(ok.sum())
        if ok.sum() < CAL_MIN_WINDOWS:
            continue
        c = float(np.median(r[ok]))
        mad = float(np.median(np.abs(r[ok] - c)) * 1.4826)
        lo, hi = max(c - max(CAL_MAD_K * mad, CAL_MIN_HALF), -SEARCH_REL), min(c + max(CAL_MAD_K * mad, CAL_MIN_HALF), SEARCH_REL)
        reg.centre[fam] = (lo + hi) / 2
        reg.half[fam] = (hi - lo) / 2
        reg.fallback[fam] = False
    return reg


def window_features(f: np.ndarray, E: np.ndarray, fr: float, kin_orders: dict[str, float], reg: Regions) -> dict[str, float]:
    out = {}
    harm_means = {}
    for fam in FAMILIES:
        fk = kin_orders[fam] * fr
        fc = fk * (1 + reg.centre[fam])
        snr1, f0 = _snr_at(f, E, fc, reg.half[fam] * fk, fr)
        snrs = [snr1]
        if np.isfinite(f0):
            for h in (2, 3):
                s, _ = _snr_at(f, E, h * f0, max(HARM_TOL_REL * h * f0, f[1] - f[0]), fr)
                snrs.append(s)
            sb = [_snr_at(f, E, f0 + sgn * fr, f[1] - f[0], fr)[0] for sgn in (-1, 1)]
        else:
            snrs += [np.nan, np.nan]
            sb = [np.nan, np.nan]
        snrs = np.array(snrs, float)
        out[f"{fam}_snr1"] = snr1
        out[f"{fam}_harm"] = float(np.nanmean(snrs)) if np.isfinite(snrs).any() else np.nan
        out[f"{fam}_nharm"] = float(np.sum(snrs >= HARM_DET_DB))
        out[f"{fam}_sb"] = float(np.nanmean(sb)) if np.isfinite(sb).any() else np.nan
        harm_means[fam] = out[f"{fam}_harm"]
    out["contrast"] = harm_means["BPFO"] - harm_means["BPFI"]
    return out
