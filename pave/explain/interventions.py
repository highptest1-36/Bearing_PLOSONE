"""Controlled envelope-domain spectral interventions for explanation verification (Phase 3A-E).

Bearing-fault evidence lives in the ENVELOPE of a resonance band, not as raw spectral lines. An intervention
therefore works in the envelope domain:

    x_b = bandpass(x, band);  z = hilbert(x_b) = a(t) exp(i phi(t))
    A(f) = rfft(a - mean a);  in each suppressed band: |A| -> |A| - s (|A| - floor)  (phase kept), floor = ring median
    a' = max(irfft(A') + mean a, 0);  x' = x + bandpass(a' cos(phi) - x_b)   (change confined to the band)

Variants
  targeted        : bands around h * f0_family (h = 1..3, calibrated region width) [+ +-fr sidebands of h=1 for BPFI]
  control_width   : same number and widths, centred at unrelated frequencies (no overlap with either family's
                    bands/sidebands or shaft-harmonic guards)
  control_energy  : unrelated bands widened step-wise until the removed envelope energy matches the targeted removal
                    (achieved ratio is reported; a shortfall is never hidden)

Nothing here uses class labels. Physical overlaps between the targeted family and the other family / shaft
harmonics are reported; overlapping parts can be excluded from the targeted bands.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from pave.signal.envelope import bandpass_sos

RING_HZ = 20.0
SHAFT_GUARD_HZ = 1.0


@dataclass
class InterventionResult:
    x: np.ndarray
    bands: list[tuple[float, float]]
    removed_env_energy: float
    env_energy: float
    clip_fraction: float
    out_of_band_rel_change: float
    in_band_snr_db: float
    total_energy_ratio: float
    info: dict = field(default_factory=dict)


def family_bands(fr: float, kin_order: float, centre_rel: float, half_rel: float, harmonics=(1, 2, 3),
                 sidebands: bool = False, min_half_hz: float = 1.0) -> list[tuple[float, float]]:
    f0 = kin_order * fr * (1 + centre_rel)
    out = []
    for h in harmonics:
        hw = max(half_rel * kin_order * fr * h, min_half_hz)
        out.append((h * f0 - hw, h * f0 + hw))
    if sidebands:
        hw = max(half_rel * kin_order * fr, min_half_hz)
        out += [(f0 - fr - hw, f0 - fr + hw), (f0 + fr - hw, f0 + fr + hw)]
    return out


def shaft_guards(fr: float, f_max: float) -> list[tuple[float, float]]:
    return [(k * fr - SHAFT_GUARD_HZ, k * fr + SHAFT_GUARD_HZ) for k in range(1, int(f_max / fr) + 2)]


def overlap_hz(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    tot = 0.0
    for lo, hi in a:
        for lo2, hi2 in b:
            tot += max(0.0, min(hi, hi2) - max(lo, lo2))
    return tot


def subtract_bands(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove intervals b from intervals a (used to exclude overlaps from targeted bands)."""
    out = []
    for lo, hi in a:
        segs = [(lo, hi)]
        for lo2, hi2 in b:
            nxt = []
            for s, e in segs:
                if hi2 <= s or lo2 >= e:
                    nxt.append((s, e))
                else:
                    if lo2 > s:
                        nxt.append((s, lo2))
                    if hi2 < e:
                        nxt.append((hi2, e))
            segs = nxt
        out += [(s, e) for s, e in segs if e - s > 1e-9]
    return out


def _decompose(x: np.ndarray, fs: float, band: tuple[float, float]):
    xb = signal.sosfiltfilt(bandpass_sos(band[0], band[1], fs), np.asarray(x, np.float64))
    z = signal.hilbert(xb)
    return xb, np.abs(z), np.angle(z)


def _mask(f: np.ndarray, bands) -> np.ndarray:
    m = np.zeros(len(f), bool)
    for lo, hi in bands:
        m |= (f >= lo) & (f <= hi)
    return m


def decompose(x: np.ndarray, fs: float, band: tuple[float, float]):
    """Pre-compute (x_b, a, phi) once per window; pass as ``dec`` to ``suppress`` to avoid repeated filtering."""
    return _decompose(x, fs, band)


def suppress(x: np.ndarray, fs: float, band: tuple[float, float], bands: list[tuple[float, float]], strength: float = 1.0,
             fr: float | None = None, dec=None) -> InterventionResult:
    xb, a, phi = dec if dec is not None else _decompose(x, fs, band)
    n = len(a)
    am = a.mean()
    A = np.fft.rfft(a - am)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(A)
    target = _mask(f, bands)
    guard = _mask(f, shaft_guards(fr, f.max())) if fr else np.zeros(len(f), bool)
    A2 = A.copy()
    for lo, hi in bands:
        sel = (f >= lo) & (f <= hi)
        if not sel.any():
            continue
        ring = (f >= lo - RING_HZ) & (f <= hi + RING_HZ) & ~target & ~guard & (f > 0)
        floor = float(np.median(mag[ring])) if ring.sum() >= 3 else 0.0
        newmag = mag[sel] - strength * np.maximum(mag[sel] - floor, 0.0)
        A2[sel] = A[sel] * np.where(mag[sel] > 0, newmag / np.maximum(mag[sel], 1e-30), 0.0)
    a2 = np.fft.irfft(A2, n) + am
    clip = float(np.mean(a2 < 0))
    a2 = np.maximum(a2, 0.0)
    xb2 = a2 * np.cos(phi)
    # confine the modification to the analysis band (re-modulation spreads energy past the band edges)
    delta = signal.sosfiltfilt(bandpass_sos(band[0], band[1], fs), xb2 - xb)
    x2 = np.asarray(x, np.float64) + delta
    xb2 = xb + delta
    removed = float(np.sum(np.abs(A) ** 2 - np.abs(A2) ** 2))
    # out-of-band check: the change must be confined to the analysis band
    d = x2 - np.asarray(x, np.float64)
    D = np.abs(np.fft.rfft(d)) ** 2
    fx = np.fft.rfftfreq(len(d), 1.0 / fs)
    inb = (fx >= band[0]) & (fx <= band[1])
    oob = float(D[~inb].sum() / max(D.sum(), 1e-30))
    snr = float(10 * np.log10(np.sum(xb ** 2) / max(np.sum((xb2 - xb) ** 2), 1e-30)))
    return InterventionResult(x2, list(bands), removed, float(np.sum(np.abs(A) ** 2)), clip, oob, snr,
                              float(np.sum(x2 ** 2) / np.sum(np.asarray(x, np.float64) ** 2)))


def control_centres(target: list[tuple[float, float]], forbidden: list[tuple[float, float]], f_lo: float, f_hi: float,
                    rng: np.random.Generator, n_try: int = 5000) -> list[tuple[float, float]]:
    """Same widths as ``target``, placed at random unrelated centres without overlapping ``forbidden`` or each other."""
    out: list[tuple[float, float]] = []
    for lo, hi in target:
        w = hi - lo
        for _ in range(n_try):
            c = rng.uniform(f_lo + w, f_hi - w)
            cand = (c - w / 2, c + w / 2)
            if overlap_hz([cand], forbidden + out) == 0:
                out.append(cand)
                break
        else:
            raise RuntimeError("could not place a control band")
    return out


def merge_bands(b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(b):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def matched_energy_control(x: np.ndarray, fs: float, band, target_removed: float, ctrl: list[tuple[float, float]],
                           forbidden: list[tuple[float, float]], fr: float, strength: float = 1.0, grow: float = 1.25,
                           max_width_hz: float = 60.0, tol: float = 0.1, max_iter: int = 30) -> tuple[InterventionResult, float]:
    """Widen control bands around their NOMINAL centres until removed envelope energy >= (1 - tol) * target.

    Widening is tracked on nominal (centre, width) pairs; the actual bands are the nominal intervals minus the
    forbidden zones, merged. Stops when the nominal width exceeds ``max_width_hz`` or after ``max_iter`` steps; the
    achieved energy ratio is returned (shortfalls are reported, not hidden).
    """
    nominal = [((lo + hi) / 2, hi - lo) for lo, hi in ctrl]
    dec = _decompose(x, fs, band)
    bands = merge_bands(list(ctrl))
    res = suppress(x, fs, band, bands, strength, fr, dec)
    it = 0
    while res.removed_env_energy < (1 - tol) * target_removed and it < max_iter:
        nominal = [(c, w * grow) for c, w in nominal]
        if max(w for _, w in nominal) > max_width_hz:
            break
        bands = merge_bands(subtract_bands([(c - w / 2, c + w / 2) for c, w in nominal], forbidden))
        res = suppress(x, fs, band, bands, strength, fr, dec)
        it += 1
    ratio = res.removed_env_energy / target_removed if target_removed > 0 else np.nan
    res.info.update({"energy_ratio_vs_target": ratio, "iterations": it, "final_nominal_width_hz": max(w for _, w in nominal)})
    return res, ratio
