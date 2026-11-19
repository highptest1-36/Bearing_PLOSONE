"""Signal-quality and descriptive statistics for raw sensor channels."""
from __future__ import annotations

import numpy as np
from scipy import stats


def longest_constant_run(x: np.ndarray) -> int:
    """Length of the longest run of exactly identical consecutive samples."""
    x = np.asarray(x)
    if x.size == 0:
        return 0
    change = np.flatnonzero(np.diff(x) != 0)
    edges = np.concatenate(([-1], change, [x.size - 1]))
    return int(np.max(np.diff(edges)))


def clipping_fraction(x: np.ndarray, rel_tol: float = 1e-6, min_run: int = 3) -> dict[str, float]:
    """Detect saturation: samples sitting at the extreme value in runs >= ``min_run``.

    A genuinely clipped channel shows many samples at (nearly) exactly the
    max/min value, often in consecutive runs. Returns the fraction of samples at
    the extremes and the longest run at an extreme.
    """
    x = np.asarray(x, dtype=np.float64)
    mx, mn = x.max(), x.min()
    span = max(mx - mn, 1e-300)
    at_hi = x >= mx - rel_tol * span
    at_lo = x <= mn + rel_tol * span
    at = at_hi | at_lo

    def longest(mask: np.ndarray) -> int:
        if not mask.any():
            return 0
        d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
        return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())

    lr = max(longest(at_hi), longest(at_lo))
    return {"extreme_frac": float(at.mean()), "extreme_count": int(at.sum()),
            "extreme_longest_run": lr, "clipping_suspect": bool(lr >= min_run and at.sum() >= 10)}


def quantization_step(x: np.ndarray, sample: int = 2_000_000) -> float:
    """Smallest positive difference between sorted unique values (on a subsample)."""
    v = np.unique(np.asarray(x[:sample], dtype=np.float64))
    d = np.diff(v)
    d = d[d > 0]
    return float(d.min()) if d.size else float("nan")


def channel_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    out = {"n": int(x.size), "n_nan": int(np.isnan(x).sum()), "n_inf": int(np.isinf(x).sum())}
    xf = x[finite]
    mean = float(xf.mean())
    std = float(xf.std())
    rms = float(np.sqrt(np.mean(xf ** 2)))
    peak = float(np.max(np.abs(xf)))
    out.update({
        "mean": mean, "std": std, "rms": rms, "min": float(xf.min()), "max": float(xf.max()),
        "peak_abs": peak, "p2p": float(xf.max() - xf.min()),
        "crest_factor": peak / rms if rms > 0 else np.nan,
        "kurtosis": float(stats.kurtosis(xf, fisher=False)),  # Gaussian = 3
        "skewness": float(stats.skew(xf)),
        "longest_constant_run": longest_constant_run(xf),
        "n_unique_first2M": int(np.unique(xf[:2_000_000]).size),
        "quant_step": quantization_step(xf),
        "is_constant": bool(std == 0.0),
    })
    out.update(clipping_fraction(xf))
    return out


def segment_stats(x: np.ndarray, fs: float, seg_s: float = 1.0) -> dict[str, np.ndarray]:
    """Per-segment RMS / kurtosis / mean for stationarity checks."""
    n = int(round(seg_s * fs))
    k = len(x) // n
    s = np.asarray(x[: k * n], dtype=np.float64).reshape(k, n)
    return {
        "rms": np.sqrt((s ** 2).mean(axis=1)),
        "mean": s.mean(axis=1),
        "kurtosis": stats.kurtosis(s, axis=1, fisher=False),
        "peak": np.abs(s).max(axis=1),
    }
