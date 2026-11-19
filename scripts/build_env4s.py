"""Protocol V2.3: 4-s envelope spectra (df = 0.25 Hz, 0-1000 Hz) for every readable PU measurement.

Output results/phase3b/env4s/: env4.npy (M, 4001) float32, meas.csv (bearing, setting, measurement, fr_hz, sigma_fr_hz,
label3, group), and the window -> measurement index win2meas.npy aligned with results/phase2/windows/pu/meta.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.signal.envelope import envelope  # noqa: E402

FS = 64000.0
N4 = 256000
BAND = (1000.0, 20000.0)
OUT = ROOT / "results" / "phase3b" / "env4s"


def env_spec(x: np.ndarray, fs: float, band, n_fft: int = N4, f_keep: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """Hann over the available samples; zero-padded to n_fft so the bin grid is fixed (0.25 Hz)."""
    e = envelope(x, fs, band)
    w = np.hanning(len(e))
    E = np.abs(np.fft.rfft(e * w, n_fft)) * 2.0 / w.sum()
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    k = f <= f_keep + 1e-9
    return f[k], E[k].astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_files.csv")
    files = files[files.read_error.isna()].sort_values(["bearing", "setting", "measurement"]).reset_index(drop=True)
    rows, specs = [], []
    t0 = time.time()
    for i, r in enumerate(files.itertuples()):
        x = np.load(ROOT / "results/phase2/cache_pu/vib" / r.bearing / f"{r.setting}_{r.bearing}_{r.measurement}.npy")
        x = x[:N4].astype(np.float64)  # 55 files are 1-6060 samples short: zero-padded FFT (documented)
        f, E = env_spec(x, FS, BAND)
        specs.append(E)
        b = BEARINGS[r.bearing]
        rows.append({"bearing": r.bearing, "setting": r.setting, "measurement": r.measurement,
                     "fr_hz": r.speed_rpm_mean / 60.0, "sigma_fr_hz": r.speed_rpm_std / 60.0,
                     "label3": b["label3"] or "", "group": b["group"]})
        if i % 400 == 0:
            print(f"{i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    assert abs(f[1] - f[0] - 0.25) < 1e-9 and len(f) == 4001
    np.save(OUT / "env4.npy", np.stack(specs))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "meas.csv", index=False)
    meta = pd.read_csv(ROOT / "results/phase2/windows/pu/meta.csv")
    key = {(a, b, c): j for j, (a, b, c) in enumerate(zip(M.bearing, M.setting, M.measurement))}
    w2m = np.array([key[(a, b, c)] for a, b, c in zip(meta.bearing, meta.setting, meta.measurement)])
    np.save(OUT / "win2meas.npy", w2m)
    print(f"done: {len(M)} measurements in {time.time()-t0:.0f}s; sigma_fr median {M.sigma_fr_hz.median():.4f} Hz, "
          f"max {M.sigma_fr_hz.max():.4f} Hz", flush=True)


if __name__ == "__main__":
    main()
