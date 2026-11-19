"""Angle-domain (order) resampling for constant-speed windows.

Within a short window the shaft speed is treated as constant (verified: KAIST sigma <= 0.01 Hz over 2-s
blocks; PU speed channel std ~0.1 rpm). A window starting at sample ``start`` is resampled so that one
shaft revolution spans exactly ``samples_per_rev`` samples:

    t_k = start/fs + k / (samples_per_rev * fr),  k = 0..n_out-1

An anti-alias low-pass (zero-phase, 8th-order Butterworth) at 0.45 * (samples_per_rev * fr) is applied
first whenever the new rate is below the original rate. Spectral lines at order o (frequency o*fr) then
appear at the same position (o cycles per revolution) for any speed, which removes speed-dependent line
positions as a shortcut.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def order_resample(x: np.ndarray, fs: float, fr: float, samples_per_rev: int, n_revs: float,
                   start: int = 0) -> np.ndarray:
    """Resample ``x[start:]`` (1-D or (n, C)) to ``n_revs`` revolutions at ``samples_per_rev``."""
    if fr <= 0:
        raise ValueError("shaft frequency must be positive")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    fs_new = samples_per_rev * fr
    n_out = int(round(samples_per_rev * n_revs))
    dur = n_out / fs_new
    need = int(np.ceil(dur * fs)) + 2
    if start + need > x.shape[0]:
        raise ValueError(f"window needs {need} samples from {start}, only {x.shape[0] - start} available")
    pad = min(start, 512)
    seg = x[start - pad: start + need + 512]
    if fs_new < fs:
        sos = signal.butter(8, 0.45 * fs_new, btype="low", fs=fs, output="sos")
        seg = signal.sosfiltfilt(sos, seg, axis=0)
    t_src = (np.arange(seg.shape[0]) - pad) / fs
    t_new = np.arange(n_out) / fs_new
    out = np.stack([np.interp(t_new, t_src, seg[:, j]) for j in range(seg.shape[1])], axis=1)
    return out[:, 0] if out.shape[1] == 1 else out


def samples_needed(fs: float, fr: float, samples_per_rev: int, n_revs: float) -> int:
    return int(np.ceil(samples_per_rev * n_revs / (samples_per_rev * fr) * fs)) + 2
