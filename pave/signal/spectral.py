"""Spectral utilities: interpolated peak picking, speed estimation, Welch PSD."""
from __future__ import annotations

import numpy as np
from scipy import signal


def interpolated_peak(freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Largest peak in [lo, hi] refined by parabolic interpolation on log magnitude.

    Returns (frequency_hz, magnitude_at_bin).
    """
    m = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    if m.size == 0:
        raise ValueError("empty search band")
    i = int(np.argmax(mag[m]))
    k = m[i]
    df = freqs[1] - freqs[0]
    if 0 < k < len(mag) - 1 and mag[k - 1] > 0 and mag[k + 1] > 0 and mag[k] > 0:
        a, b, c = np.log(mag[k - 1:k + 2])
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    return float(freqs[k] + delta * df), float(mag[k])


def amplitude_spectrum(x: np.ndarray, fs: float, pad: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided Hann-windowed amplitude spectrum (amplitude-corrected)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x * w, n * pad)) * 2.0 / w.sum()
    f = np.fft.rfftfreq(n * pad, 1.0 / fs)
    return f, X


def estimate_shaft_speed(x: np.ndarray, fs: float, nominal_hz: float = 50.17, search_rel: float = 0.04,
                         use_harmonic: int = 2, decim: int = 16) -> dict[str, float]:
    """Estimate shaft frequency from 1X and 2X spectral peaks of a vibration channel.

    The signal is decimated (FIR anti-alias) to speed up the long FFT. The result
    from harmonic ``use_harmonic`` is divided by that harmonic number. Both 1X and
    2X estimates are returned so their agreement can be checked.
    """
    xd = signal.decimate(np.asarray(x, dtype=np.float64), decim, ftype="fir", zero_phase=True)
    fsd = fs / decim
    f, X = amplitude_spectrum(xd, fsd, pad=2)
    out = {}
    for h in (1, 2):
        c = h * nominal_hz
        pk, amp = interpolated_peak(f, X, c * (1 - search_rel), c * (1 + search_rel))
        out[f"f{h}x_hz"] = pk / h
        out[f"a{h}x"] = amp
    out["fr_hz"] = out[f"f{use_harmonic}x_hz"]
    out["fr_1x_2x_diff_hz"] = abs(out["f1x_hz"] - out["f2x_hz"])
    out["freq_resolution_hz"] = fsd / (len(xd) * 2)
    return out


def track_peak(x: np.ndarray, fs: float, center: float, half_width: float, win_s: float = 2.0,
               hop_s: float = 0.25, pad: int = 8) -> np.ndarray:
    """Per-block interpolated peak frequency near ``center`` (for speed stability)."""
    nwin = int(round(win_s * fs))
    hop = int(round(hop_s * fs))
    w = np.hanning(nwin)
    nfft = nwin * pad
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    out = []
    for s in range(0, len(x) - nwin + 1, hop):
        seg = x[s:s + nwin]
        X = np.abs(np.fft.rfft((seg - seg.mean()) * w, nfft))
        out.append(interpolated_peak(f, X, center - half_width, center + half_width)[0])
    return np.asarray(out)


def welch_psd(x: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    return signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann",
                        detrend="constant", scaling="density", axis=0)
