"""Window caches for Phase 2 (protocol V2 §5). Windows are defined per recording / measurement file BEFORE any
split is applied; split assignment happens later by recording/bearing identity and time intervals.

Outputs per dataset in ``results/phase2/windows/<ds>/``:
  meta.csv        one row per raw window (recording/bearing, labels, t_start, t_end, shaft fr, ...)
  raw.npy         memmap (N, C, L_raw)   KAIST: 2 x 25600 @25.6 kHz;  PU: 1 x 32000 (decimated x2 -> 32 kHz)
  order_meta.csv  one row per order window
  order.npy       memmap (M, C, 24576)   angle-domain resampled windows
  hand.npy        (N, F) hand-crafted features (per channel concatenated)
  short.npy       (N, 1 + C) [shaft fr (Hz), log10 RMS per channel]
  env.npy         (N, C, 1001) envelope amplitude spectrum 0..1000 Hz at 1 Hz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from pave.features.handcrafted import handcrafted
from pave.features.physics import envelope_spectrum_window
from pave.signal.order import order_resample, samples_needed
from pave.signal.spectral import amplitude_spectrum, interpolated_peak

ENV_MAX_HZ = 1000
KAIST_BAND = (1000.0, 10000.0)
PU_BAND = (1000.0, 20000.0)
ORDER_SPR = {"kaist": 512, "pu": 1024}
ORDER_NREV = {"kaist": 48, "pu": 24}


def window_speed_2x(x: np.ndarray, fs: float, nominal: float = 50.17, rel: float = 0.04) -> float:
    f, X = amplitude_spectrum(x, fs, pad=4)
    return interpolated_peak(f, X, 2 * nominal * (1 - rel), 2 * nominal * (1 + rel))[0] / 2


def _env(x: np.ndarray, fs: float, band) -> np.ndarray:
    f, E = envelope_spectrum_window(x, fs, band)
    df = f[1] - f[0]
    assert abs(df - 1.0) < 1e-9, df
    return E[: ENV_MAX_HZ + 1].astype(np.float32)


def build_kaist(rec_csv: Path, out: Path, read_fn) -> None:
    rec = pd.read_csv(rec_csv)
    rec = rec[rec.in_bearing_scope].sort_values("recording_id").reset_index(drop=True)
    fs = 25600.0
    L = 25600
    n_total = int(sum(int(d) for d in rec.vib_duration_s))
    spr, nrev = ORDER_SPR["kaist"], ORDER_NREV["kaist"]
    out.mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(out / "raw.npy", mode="w+", dtype=np.float16, shape=(n_total, 2, L))
    order = np.lib.format.open_memmap(out / "order.npy", mode="w+", dtype=np.float16, shape=(n_total, 2, spr * nrev))
    hand, short, env, meta = [], [], [], []
    i = 0
    for r in rec.itertuples():
        v = read_fn(r.vib_relative_path)
        X = v.values[:, :2]  # Point1, Point2 = housing A x, y (ASSUMED mapping)
        nw = X.shape[0] // L
        for k in range(nw):
            w = X[k * L:(k + 1) * L]
            fr = window_speed_2x(w[:, 0], fs)
            raw[i] = w.T.astype(np.float16)
            order[i] = order_resample(w, fs, fr, spr, nrev, start=0).T.astype(np.float16)
            hand.append(np.concatenate([handcrafted(w[:, c], fs) for c in range(2)]))
            short.append([fr] + [np.log10(np.sqrt(np.mean(w[:, c].astype(np.float64) ** 2))) for c in range(2)])
            env.append(np.stack([_env(w[:, c], fs, KAIST_BAND) for c in range(2)]))
            meta.append({"idx": i, "group": r.recording_id, "recording_id": r.recording_id, "label3": r.label3,
                         "label7": r.label7, "load_Nm": r.load_Nm, "session": r.session, "window": k,
                         "t_start": float(k), "t_end": float(k + 1), "fr_hz": fr})
            i += 1
        del v, X
    assert i == n_total, (i, n_total)
    raw.flush()
    order.flush()
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    pd.DataFrame(meta)[["idx", "group"]].to_csv(out / "order_meta.csv", index=False)  # 1:1 for KAIST
    np.save(out / "hand.npy", np.asarray(hand, np.float32))
    np.save(out / "short.npy", np.asarray(short, np.float64))
    np.save(out / "env.npy", np.asarray(env, np.float32))
    (out / "info.json").write_text(json.dumps({"fs_raw": fs, "fs_env": fs, "band": KAIST_BAND, "order_spr": spr,
                                               "order_nrev": nrev, "n": n_total, "channels": ["x_housing_A", "y_housing_A"]}))


def build_pu(files_csv: Path, bearings_csv: Path, vib_dir: Path, out: Path, log=print) -> None:
    files = pd.read_csv(files_csv).sort_values(["bearing", "setting", "measurement"]).reset_index(drop=True)
    if "read_error" in files:
        files = files[files.read_error.isna()].reset_index(drop=True)  # unreadable files excluded (documented)
    bear = pd.read_csv(bearings_csv).set_index("bearing")
    fs = 64000.0
    L = 64000
    spr, nrev = ORDER_SPR["pu"], ORDER_NREV["pu"]
    n_raw = int(sum(n // L for n in files.n_vib))
    n_ord = 0
    for f in files.itertuples():
        fr = f.speed_rpm_mean / 60.0
        need = samples_needed(fs, fr, spr, nrev) + 1024
        n_ord += f.n_vib // need
    out.mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(out / "raw.npy", mode="w+", dtype=np.float16, shape=(n_raw, 1, L // 2))
    order = np.lib.format.open_memmap(out / "order.npy", mode="w+", dtype=np.float16, shape=(n_ord, 1, spr * nrev))
    hand, short, env, meta, ometa = [], [], [], [], []
    i = j = 0
    for fi, f in enumerate(files.itertuples()):
        x = np.load(vib_dir / f.bearing / f"{f.setting}_{f.bearing}_{f.measurement}.npy").astype(np.float64)
        fr = f.speed_rpm_mean / 60.0
        xd = signal.decimate(x, 2, ftype="fir", zero_phase=True)
        b = bear.loc[f.bearing]
        for k in range(len(x) // L):
            w = x[k * L:(k + 1) * L]
            raw[i, 0] = xd[k * (L // 2):(k + 1) * (L // 2)].astype(np.float16)
            hand.append(handcrafted(w, fs))
            short.append([fr, np.log10(np.sqrt(np.mean(w ** 2)))])
            env.append(_env(w, fs, PU_BAND)[None])
            meta.append({"idx": i, "group": f.bearing, "bearing": f.bearing, "bearing_group": b.group,
                         "label3": b.label3 if isinstance(b.label3, str) else "", "setting": f.setting,
                         "measurement": f.measurement, "window": k, "t_start": float(k), "t_end": float(k + 1),
                         "fr_hz": fr, "datetime": f.datetime})
            i += 1
        need = samples_needed(fs, fr, spr, nrev) + 1024
        for k in range(len(x) // need):
            order[j, 0] = order_resample(x, fs, fr, spr, nrev, start=k * need + 512).astype(np.float16)
            ometa.append({"idx": j, "group": f.bearing, "bearing": f.bearing, "setting": f.setting,
                          "measurement": f.measurement, "segment": k,
                          "label3": b.label3 if isinstance(b.label3, str) else "", "fr_hz": fr})
            j += 1
        if fi % 200 == 0:
            log(f"PU windows: file {fi}/{len(files)}")
    assert i == n_raw and j == n_ord, (i, n_raw, j, n_ord)
    raw.flush()
    order.flush()
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    pd.DataFrame(ometa).to_csv(out / "order_meta.csv", index=False)
    np.save(out / "hand.npy", np.asarray(hand, np.float32))
    np.save(out / "short.npy", np.asarray(short, np.float64))
    np.save(out / "env.npy", np.asarray(env, np.float32))
    (out / "info.json").write_text(json.dumps({"fs_raw": fs / 2, "fs_env": fs, "band": PU_BAND, "order_spr": spr,
                                               "order_nrev": nrev, "n": n_raw, "n_order": n_ord, "channels": ["vibration_1"]}))
