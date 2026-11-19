"""Envelope analysis and physics-evidence measurement.

Pipeline: band-pass (zero-phase Butterworth SOS) -> analytic signal (Hilbert)
-> |.| envelope -> remove mean -> averaged envelope amplitude spectrum.

Evidence statistic for a target frequency f (e.g. BPFI) with relative tolerance
``tol`` (default 1 %):

    peak  = max |E(f')| for f' in [f(1-tol), f(1+tol)]
    floor = median |E(f')| over a reference ring [f-ref_hz, f+ref_hz] excluding
            the tolerance band and excluding +-guard around shaft harmonics
    SNR_dB = 20 log10(peak / floor)

Because SNR_dB is the max over several bins divided by a median, it is
positively biased even for pure noise. ``evidence_against_controls`` therefore
compares the target statistic with the same statistic evaluated at a set of
CONTROL frequencies that are not shaft harmonics and not bearing frequencies
(or their low-order harmonics) - the empirical null for that recording/window.
Band selection for the envelope is either a fixed preregistered band or a
label-free spectral-kurtosis choice computed from the signal itself; neither
uses class labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .physics import orders


def bandpass_sos(lo: float, hi: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if not (0 < lo < hi < nyq):
        raise ValueError(f"invalid band [{lo}, {hi}] for fs={fs} (Nyquist {nyq})")
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def envelope(x: np.ndarray, fs: float, band: tuple[float, float] | None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if band is not None:
        x = signal.sosfiltfilt(bandpass_sos(band[0], band[1], fs), x)
    env = np.abs(signal.hilbert(x))
    return env - env.mean()


def envelope_spectrum(env: np.ndarray, fs: float, seg_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Averaged amplitude spectrum of the envelope over non-overlapping segments.

    Frequency resolution = 1/seg_s Hz. Averaging magnitudes over segments
    stabilises the noise floor without assuming phase coherence across segments.
    """
    n = int(round(seg_s * fs))
    k = len(env) // n
    if k < 1:
        raise ValueError("signal shorter than one segment")
    segs = env[: k * n].reshape(k, n)
    segs = segs - segs.mean(axis=1, keepdims=True)
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(segs * w, axis=1)) * 2.0 / w.sum()
    return np.fft.rfftfreq(n, 1.0 / fs), X.mean(axis=0)


def spectral_kurtosis_band(x: np.ndarray, fs: float, nperseg: int = 256, bw_hz: float = 2000.0,
                           fmin: float = 1000.0, fmax: float | None = None) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    """Label-free band choice: centre at the STFT spectral-kurtosis maximum.

    SK(f) = <|S|^4> / <|S|^2>^2 - 2  (Antoni 2006). The band [fc-bw/2, fc+bw/2]
    is clipped to [fmin, 0.95*Nyquist]. SK is smoothed over 5 bins.
    """
    fmax = fmax if fmax is not None else 0.95 * fs / 2
    f, _, S = signal.stft(np.asarray(x, dtype=np.float64), fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                          window="hann", boundary=None, padded=False)
    P = np.abs(S) ** 2
    sk = (P ** 2).mean(axis=1) / np.maximum(P.mean(axis=1) ** 2, 1e-300) - 2.0
    sk_s = np.convolve(sk, np.ones(5) / 5, mode="same")
    valid = (f >= fmin + bw_hz / 2) & (f <= fmax - bw_hz / 2)
    fc = float(f[valid][np.argmax(sk_s[valid])])
    return (fc - bw_hz / 2, fc + bw_hz / 2), f, sk


@dataclass
class PeakEvidence:
    target_hz: float
    peak_hz: float
    peak_amp: float
    floor_amp: float
    snr_db: float


def peak_snr(freqs: np.ndarray, spec: np.ndarray, f_target: float, tol: float = 0.01, ref_hz: float = 20.0,
             fr: float | None = None, guard_hz: float = 1.0) -> PeakEvidence:
    df = freqs[1] - freqs[0]
    half = max(tol * f_target, df)  # at least one bin
    inband = (freqs >= f_target - half) & (freqs <= f_target + half)
    ring = (freqs >= f_target - ref_hz) & (freqs <= f_target + ref_hz) & ~inband
    if fr is not None and fr > 0:
        h = np.round(freqs / fr)
        ring &= ~((np.abs(freqs - h * fr) <= guard_hz) & (h >= 1))
    if not inband.any() or ring.sum() < 3:
        return PeakEvidence(f_target, np.nan, np.nan, np.nan, np.nan)
    i = np.argmax(np.where(inband, spec, -np.inf))
    floor = float(np.median(spec[ring]))
    peak = float(spec[i])
    snr = 20 * np.log10(peak / floor) if floor > 0 and peak > 0 else np.nan
    return PeakEvidence(f_target, float(freqs[i]), peak, floor, float(snr))


def control_orders(n: int = 24, lo: float = 2.3, hi: float = 19.0, seed: int = 0,
                   min_sep_order: float = 0.12) -> np.ndarray:
    """Deterministic set of control frequency orders (multiples of fr).

    Excludes integers (shaft harmonics), half-integers, and +-1 sidebands of the
    first three harmonics of BPFO, BPFI, FTF-related orders, each by at least
    ``min_sep_order`` orders.
    """
    o = orders()
    forbidden = []
    for k in range(0, int(hi) + 2):
        forbidden += [k, k + 0.5]
    for key in ("BPFO", "BPFI", "BSF", "2xBSF"):
        for h in (1, 2, 3):
            for sb in (-2, -1, 0, 1, 2):
                forbidden.append(h * o[key] + sb)
    forbidden = np.asarray(forbidden)
    rng = np.random.default_rng(seed)
    out: list[float] = []
    tries = 0
    while len(out) < n and tries < 100000:
        c = float(rng.uniform(lo, hi))
        tries += 1
        if np.min(np.abs(forbidden - c)) < min_sep_order:
            continue
        if out and np.min(np.abs(np.asarray(out) - c)) < 2 * min_sep_order:
            continue
        out.append(c)
    return np.sort(np.asarray(out))


CONTROL_ORDERS = control_orders()


def evidence_against_controls(freqs: np.ndarray, spec: np.ndarray, fr: float, f_target: float,
                              tol: float = 0.01, ref_hz: float = 20.0,
                              ctrl_orders: np.ndarray = CONTROL_ORDERS) -> dict[str, float]:
    """Target SNR and its standing relative to control-frequency SNRs in the same spectrum."""
    t = peak_snr(freqs, spec, f_target, tol, ref_hz, fr)
    ctrl = np.array([peak_snr(freqs, spec, c * fr, tol, ref_hz, fr).snr_db for c in ctrl_orders])
    ctrl = ctrl[np.isfinite(ctrl)]
    med = float(np.median(ctrl)) if ctrl.size else np.nan
    mad = float(np.median(np.abs(ctrl - med)) * 1.4826) if ctrl.size else np.nan
    exceed = float(np.mean(ctrl >= t.snr_db)) if ctrl.size and np.isfinite(t.snr_db) else np.nan
    return {
        "snr_db": t.snr_db,
        "peak_hz": t.peak_hz,
        "peak_ratio": t.peak_hz / f_target if np.isfinite(t.peak_hz) else np.nan,
        "ctrl_median_db": med,
        "ctrl_q95_db": float(np.quantile(ctrl, 0.95)) if ctrl.size else np.nan,
        "ctrl_max_db": float(ctrl.max()) if ctrl.size else np.nan,
        "excess_db": t.snr_db - med if np.isfinite(med) else np.nan,
        "robust_z": (t.snr_db - med) / mad if mad and np.isfinite(mad) and mad > 0 else np.nan,
        "ctrl_exceed_frac": exceed,  # empirical p-value-like (fraction of controls >= target)
    }


# ----------------------------------------------------------------------------- family search
FAMILY_REL_WIDTH = 0.035  # +-3.5 % search around kinematic order (POST-HOC; see audit report)


def family_search(freqs: np.ndarray, spec: np.ndarray, fr: float, center_order: float,
                  rel_width: float = FAMILY_REL_WIDTH, ref_hz: float = 20.0, guard_hz: float = 1.0,
                  harm_tol: float = 0.005, harmonics: tuple[int, ...] = (2, 3)) -> dict[str, float]:
    """Strongest line within +-rel_width of ``center_order * fr`` (label-free).

    Bins within +-guard_hz of shaft harmonics k*fr are excluded from the search
    and from the noise floor. Floor = median spectrum over the search band
    extended by +-ref_hz, excluding the peak +-2 bins and shaft-harmonic bins.
    Harmonic support = SNR at h*f0 (+-harm_tol) against a +-ref_hz ring.
    The search maximum is positively biased under noise; thresholds must come
    from fault-free recordings analysed with the identical statistic.
    """
    df = freqs[1] - freqs[0]
    c = center_order * fr
    lo, hi = c * (1 - rel_width), c * (1 + rel_width)
    h = np.round(freqs / fr)
    shaft = (np.abs(freqs - h * fr) <= guard_hz) & (h >= 1)
    band = (freqs >= lo) & (freqs <= hi) & ~shaft
    if not band.any():
        return {"f0_hz": np.nan, "order": np.nan, "snr_db": np.nan}
    i = int(np.flatnonzero(band)[np.argmax(spec[band])])
    ring = (freqs >= lo - ref_hz) & (freqs <= hi + ref_hz) & ~shaft
    ring[max(i - 2, 0): i + 3] = False
    floor = float(np.median(spec[ring]))
    f0 = float(freqs[i])
    out = {"f0_hz": f0, "order": f0 / fr, "snr_db": float(20 * np.log10(spec[i] / floor)) if floor > 0 else np.nan,
           "rel_offset": f0 / c - 1.0, "freq_res_hz": float(df)}
    for k in harmonics:
        r = peak_snr(freqs, spec, k * f0, tol=harm_tol, ref_hz=ref_hz, fr=fr, guard_hz=guard_hz)
        out[f"h{k}_snr_db"] = r.snr_db
    return out
