"""Per-file temperature / motor-current audit for TDMS logs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from pave.data.read_tdms import read_tdms
from pave.signal.spectral import amplitude_spectrum, interpolated_peak
from .vibration_audit import _json_default


def held_values(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) where a sample-and-hold signal changes (incl. first sample)."""
    ch = np.flatnonzero(np.diff(x) != 0) + 1
    idx = np.concatenate(([0], ch))
    return idx, x[idx]


def temperature_stats(x: np.ndarray, dt_s: float) -> dict:
    idx, v = held_values(x)
    runs = np.diff(idx)
    t = idx * dt_s
    # use only update instants (true measurements) for trend/noise
    tt = t[1:] if len(t) > 2 else t  # first value may be a partial hold
    vv = v[1:] if len(v) > 2 else v
    p1 = np.polyfit(tt, vv, 1)
    p2 = np.polyfit(tt, vv, 2)
    resid = vv - np.polyval(p2, tt)
    dur = len(x) * dt_s
    first = x[: int(5 / dt_s)].mean()
    last = x[-int(5 / dt_s):].mean()
    return {
        "n_samples": int(len(x)), "n_nan": int(np.isnan(x).sum()), "n_inf": int(np.isinf(x).sum()),
        "n_updates": int(len(idx)), "update_period_samples_median": float(np.median(runs)) if len(runs) else np.nan,
        "update_period_samples_min": float(runs.min()) if len(runs) else np.nan,
        "update_period_samples_max": float(runs.max()) if len(runs) else np.nan,
        "effective_rate_hz": float(1.0 / (np.median(runs) * dt_s)) if len(runs) else np.nan,
        "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()),
        "first5s_mean": float(first), "last5s_mean": float(last), "rise_C": float(last - first),
        "slope_C_per_min": float(p1[0] * 60.0), "curvature_C_per_min2": float(p2[0] * 3600.0),
        "resid_noise_std_C": float(resid.std()),
        "step_median_C": float(np.median(np.abs(np.diff(v)))) if len(v) > 1 else np.nan,
        "duration_s": float(dur),
    }


def current_features(I: np.ndarray, fs: float) -> dict:
    """Supply frequency and motor-shaft speed from eccentricity sidebands (f_s +- f_rm)."""
    xd = signal.decimate(I, 16, ftype="fir")
    fsd = fs / 16
    f, X = amplitude_spectrum(xd, fsd, pad=2)
    fsup, asup = interpolated_peak(f, X, 40, 60)
    # motor shaft ~ 24 Hz for this rig (gear ratio ~2.07 to ~50 Hz output shaft)
    usb, ausb = interpolated_peak(f, X, fsup + 20, fsup + 28)
    lsb, alsb = interpolated_peak(f, X, fsup - 28, fsup - 20)
    return {"supply_hz": fsup, "supply_amp_A": asup, "usb_hz": usb, "lsb_hz": lsb,
            "motor_fr_usb_hz": usb - fsup, "motor_fr_lsb_hz": fsup - lsb,
            "usb_amp_A": ausb, "lsb_amp_A": alsb, "rms_A": float(np.sqrt(np.mean((I - I.mean()) ** 2)))}


def audit_tdms_file(path: Path, cache_dir: Path) -> dict:
    rec = read_tdms(path, load=("temperature", "current"))
    out: dict = {"file": path.name, "stem": path.stem, "file_name_prop": rec.file_name_prop,
                 "run_number": rec.run_number, "product": rec.product,
                 "date_created_utc": rec.date_created_utc.isoformat() if rec.date_created_utc else None,
                 "channels": []}
    for c in rec.channels:
        out["channels"].append({
            "raw_name": c.raw_name, "name": c.name, "kind": c.kind, "n_samples": c.n_samples, "unit": c.unit,
            "increment_s": c.increment_s, "start_time_utc": c.start_time_utc.isoformat() if c.start_time_utc else None,
            "device": c.device, "tc_type": c.properties.get("NI_ThermocoupleType"),
            "cjc_source": c.properties.get("NI_CjcSource"),
            "prop_min": c.properties.get("minimum"), "prop_max": c.properties.get("maximum"),
        })
    temps = {}
    arrays = {}
    tch = [c for c in rec.channels if c.kind == "temperature"]
    dt_s = tch[0].increment_s
    out["temp_start_utc"] = tch[0].start_time_utc.isoformat()
    out["temp_increment_s"] = dt_s
    out["temp_nominal_rate_hz"] = 1.0 / dt_s
    for c in tch:
        x = rec.data[c.name]
        temps[c.name] = temperature_stats(x, dt_s)
        idx, v = held_values(x)
        arrays[f"T|{c.name}|t"] = (idx * dt_s).astype(np.float64)
        arrays[f"T|{c.name}|v"] = v.astype(np.float64)
    out["temperature"] = temps
    a, b = rec.data.get("T_housing_A"), rec.data.get("T_housing_B")
    if a is not None and b is not None:
        ia, va = held_values(a)
        ib, vb = held_values(b)
        out["T_updates_simultaneous"] = bool(len(ia) == len(ib) and np.array_equal(ia, ib))
        d = a - b
        out["T_A_minus_B"] = {"mean": float(d.mean()), "std": float(d.std()), "min": float(d.min()), "max": float(d.max())}
        if len(va) == len(vb) and len(va) > 3:
            out["T_A_B_corr_detrended"] = float(np.corrcoef(signal.detrend(va), signal.detrend(vb))[0, 1])
        out["T_channels_identical"] = bool(np.array_equal(a, b))
    cur = {}
    for c in rec.channels:
        if c.kind == "current":
            if c.n_samples == 0:
                cur[c.name] = {"available": False}
                continue
            cur[c.name] = {"available": True, **current_features(rec.data[c.name], 1.0 / c.increment_s)}
    out["current"] = cur
    np.savez_compressed(cache_dir / f"tdms_{path.stem}.npz", **arrays)
    (cache_dir / f"tdms_{path.stem}.json").write_text(json.dumps(out, indent=1, default=_json_default))
    del rec
    return out
