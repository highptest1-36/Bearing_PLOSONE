"""Classical hand-crafted vibration features (time domain + 16 log band energies)."""
from __future__ import annotations

import numpy as np
from scipy import stats

N_BANDS = 16
TIME_NAMES = ["mean", "std", "rms", "peak", "p2p", "crest", "kurtosis", "skewness", "shape", "impulse",
              "clearance", "abs_mean"]


def time_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    a = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2))
    am = a.mean()
    pk = a.max()
    sr = np.mean(np.sqrt(a)) ** 2
    return np.array([x.mean(), x.std(), rms, pk, x.max() - x.min(), pk / rms if rms else 0.0,
                     stats.kurtosis(x, fisher=False), stats.skew(x), rms / am if am else 0.0,
                     pk / am if am else 0.0, pk / sr if sr else 0.0, am])


def band_energies(x: np.ndarray, fs: float, f_max: float | None = None, n_bands: int = N_BANDS) -> np.ndarray:
    """log10 of mean power in ``n_bands`` equal-width bands from 0 to f_max (default Nyquist)."""
    x = np.asarray(x, np.float64) - np.mean(x)
    w = np.hanning(len(x))
    P = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    f_max = f_max or fs / 2
    edges = np.linspace(0, f_max, n_bands + 1)
    out = np.empty(n_bands)
    for i in range(n_bands):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out[i] = np.log10(P[m].mean() + 1e-20)
    return out


def handcrafted(x: np.ndarray, fs: float) -> np.ndarray:
    return np.concatenate([time_features(x), band_energies(x, fs)])


def names(prefix: str = "") -> list[str]:
    return [prefix + n for n in TIME_NAMES] + [f"{prefix}band{i:02d}" for i in range(N_BANDS)]
