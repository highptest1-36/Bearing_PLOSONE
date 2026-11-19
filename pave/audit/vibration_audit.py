"""Per-recording vibration audit: integrity, quality, speed, spectra, physics evidence.

Nothing here reads class labels; evidence is computed identically for every
recording (healthy, faulty and out-of-scope conditions alike).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pave.data.read_mat import read_vibration_mat
from pave.signal import envelope as env
from pave.signal.physics import all_frequencies, orders
from pave.signal.quality import channel_stats, segment_stats
from pave.signal.spectral import estimate_shaft_speed, track_peak, welch_psd

FIXED_BAND = (1000.0, 10000.0)  # preregistered broad band (Hz); label-free
SK_BW = 2000.0
ENV_SEG_RECORD_S = 4.0          # whole-record envelope spectrum resolution 0.25 Hz
ENV_SEG_WINDOW_S = 1.0          # window-level resolution 1 Hz
TOL = 0.01                      # +-1 % frequency tolerance (primary)
TOL_SENS = (0.005, 0.02)        # sensitivity tolerances
TARGETS = {  # name -> (base key, harmonic)
    "BPFO_1": ("BPFO", 1), "BPFO_2": ("BPFO", 2), "BPFO_3": ("BPFO", 3),
    "BPFI_1": ("BPFI", 1), "BPFI_2": ("BPFI", 2), "BPFI_3": ("BPFI", 3),
}


def audit_vibration_file(path: Path, cache_dir: Path) -> dict:
    rec = read_vibration_mat(path, dtype=np.float64)
    X = rec.values
    fs = rec.fs_hz
    out: dict = {
        "file": path.name, "stem": path.stem, "fs_hz": fs, "increment_s": rec.increment_s,
        "x_start_s": rec.x_start_s, "n_samples_header": rec.n_samples, "n_samples": int(X.shape[0]),
        "n_channels": int(X.shape[1]), "dtype": str(X.dtype), "duration_s": X.shape[0] / fs,
        "channel_labels": rec.channel_labels, "channel_names": rec.channel_names,
        "unit_label": rec.unit_label, "unit_factor": rec.unit_factor, "unit_offset": rec.unit_offset,
        "unit_info": rec.unit_info, "run_name": rec.run_name, "absolute_time_raw": rec.absolute_time_raw,
        "absolute_time": rec.absolute_time.isoformat() if rec.absolute_time else None,
        "creation_time_raw": rec.creation_time_raw, "orig_location": rec.orig_location,
        "mat_header": rec.mat_header, "extra": {k: v for k, v in rec.extra.items() if k != "values_shape"},
    }
    # --- integrity & quality per channel
    ch_stats = {}
    seg = {}
    for j, name in enumerate(rec.channel_names):
        x = X[:, j]
        s = channel_stats(x)
        ss = segment_stats(x, fs, 1.0)
        s["seg_rms_cv"] = float(ss["rms"].std() / ss["rms"].mean())
        s["seg_rms_drift_rel"] = float((ss["rms"][-5:].mean() - ss["rms"][:5].mean()) / ss["rms"].mean())
        s["seg_kurt_median"] = float(np.median(ss["kurtosis"]))
        s["seg_kurt_iqr"] = float(np.subtract(*np.percentile(ss["kurtosis"], [75, 25])))
        s["seg_mean_abs_max"] = float(np.abs(ss["mean"]).max())
        ch_stats[name] = s
        seg[name] = ss
    out["channels"] = ch_stats
    # duplicated channels?
    dup = []
    for a in range(X.shape[1]):
        for b in range(a + 1, X.shape[1]):
            if np.array_equal(X[:200000, a], X[:200000, b]):
                dup.append([rec.channel_names[a], rec.channel_names[b]])
    out["duplicate_channel_pairs"] = dup
    # --- speed (from housing-A x; cross-checked on all channels)
    sp = {name: estimate_shaft_speed(X[:, j], fs) for j, name in enumerate(rec.channel_names)}
    out["speed_by_channel"] = sp
    frs = np.array([v["fr_hz"] for v in sp.values()])
    out["fr_hz"] = float(np.median(frs))
    out["fr_spread_hz"] = float(frs.max() - frs.min())
    from scipy import signal as ss_
    xd = ss_.decimate(X[:, 0], 16, ftype="fir")
    trk = track_peak(xd, fs / 16, 2 * out["fr_hz"], 1.0, win_s=2.0, hop_s=1.0) / 2
    out["fr_track_std_hz"] = float(trk.std())
    out["fr_track_range_hz"] = float(trk.max() - trk.min())
    freqs = all_frequencies(out["fr_hz"])
    out["bearing_freqs_measured_speed"] = freqs
    # --- PSD (1 Hz resolution)
    f_psd, P = welch_psd(X, fs, nperseg=int(fs))
    # --- envelope analysis
    evidence = {}
    env_spectra = {}
    win_ev = {}
    sk_info = {}
    nyq = fs / 2
    for j, name in enumerate(rec.channel_names):
        x = X[:, j]
        band_sk, f_sk, sk = env.spectral_kurtosis_band(x, fs, nperseg=256, bw_hz=SK_BW)
        sk_info[name] = {"band": list(band_sk), "sk_max": float(np.max(sk[(f_sk > 1000) & (f_sk < 0.95 * nyq)]))}
        for band_name, band in (("FIXED", FIXED_BAND), ("SK", band_sk)):
            e = env.envelope(x, fs, band)
            fe, E = env.envelope_spectrum(e, fs, ENV_SEG_RECORD_S)
            keep = fe <= 1200
            fe, E = fe[keep], E[keep]
            env_spectra[f"{name}|{band_name}"] = E.astype(np.float32)
            env_f = fe
            for t, (key, h) in TARGETS.items():
                ft = h * freqs[key]
                r = env.evidence_against_controls(fe, E, out["fr_hz"], ft, tol=TOL)
                for tol in TOL_SENS:
                    r2 = env.evidence_against_controls(fe, E, out["fr_hz"], ft, tol=tol)
                    r[f"excess_db_tol{tol}"] = r2["excess_db"]
                evidence[f"{name}|{band_name}|{t}"] = r
            # shaft modulation (1x fr in envelope) - informative for IR
            r = env.peak_snr(fe, E, out["fr_hz"], tol=TOL, ref_hz=10.0)
            evidence[f"{name}|{band_name}|FR_env"] = {"snr_db": r.snr_db, "peak_hz": r.peak_hz}
            # BPFI +- fr sidebands
            for sgn, lab in ((-1, "BPFI_m_fr"), (1, "BPFI_p_fr")):
                r = env.evidence_against_controls(fe, E, out["fr_hz"], freqs["BPFI"] + sgn * out["fr_hz"], tol=TOL)
                evidence[f"{name}|{band_name}|{lab}"] = r
            # label-free family search (+-3.5 %, POST-HOC, see report)
            for fam in ("BPFO", "BPFI"):
                evidence[f"{name}|{band_name}|FAM_{fam}"] = env.family_search(fe, E, out["fr_hz"], orders()[fam])
            # window-level (1 s, non-overlapping) for BPFO_1 / BPFI_1
            n = int(fs * ENV_SEG_WINDOW_S)
            k = len(e) // n
            W = e[: k * n].reshape(k, n)
            W = W - W.mean(axis=1, keepdims=True)
            w = np.hanning(n)
            fw = np.fft.rfftfreq(n, 1.0 / fs)
            kw = fw <= 1200
            WE = (np.abs(np.fft.rfft(W * w, axis=1)) * 2.0 / w.sum())[:, kw]
            fw = fw[kw]
            for t in ("BPFO_1", "BPFI_1"):
                key = t.split("_")[0]
                vals = np.array([[env.evidence_against_controls(fw, WE[i], out["fr_hz"], freqs[key], tol=TOL)[m]
                                  for m in ("excess_db", "snr_db", "ctrl_exceed_frac")] for i in range(k)])
                win_ev[f"{name}|{band_name}|{t}"] = vals.astype(np.float32)
            for fam in ("BPFO", "BPFI"):
                vals = []
                for i in range(k):
                    r = env.family_search(fw, WE[i], out["fr_hz"], orders()[fam])
                    vals.append([r["snr_db"], r["order"], r.get("h2_snr_db", np.nan)])
                win_ev[f"{name}|{band_name}|FAM_{fam}"] = np.asarray(vals, dtype=np.float32)
            del e, W, WE
    out["sk_band"] = sk_info
    out["evidence"] = evidence
    out["envelope_freq_res_hz"] = 1.0 / ENV_SEG_RECORD_S
    out["window_evidence_summary"] = {
        k: ({"n_windows": int(v.shape[0]), "median_snr_db": float(np.nanmedian(v[:, 0])),
             "median_order": float(np.nanmedian(v[:, 1]))} if "|FAM_" in k else
            {"n_windows": int(v.shape[0]),
             "frac_excess_ge_6dB": float(np.mean(v[:, 0] >= 6.0)),
             "frac_ctrl_exceed_le_0.05": float(np.mean(v[:, 2] <= 0.05)),
             "median_excess_db": float(np.median(v[:, 0]))})
        for k, v in win_ev.items()}
    # --- snippets for plots (first 0.2 s of every channel; and 1 s at mid-recording)
    mid = X.shape[0] // 2
    np.savez_compressed(
        cache_dir / f"vib_{path.stem}.npz",
        psd_f=f_psd[f_psd <= 12800].astype(np.float32), psd=P[f_psd <= 12800].astype(np.float32),
        env_f=env_f.astype(np.float32), **{f"env|{k}": v for k, v in env_spectra.items()},
        **{f"win|{k}": v for k, v in win_ev.items()},
        **{f"seg|{c}|{m}": v.astype(np.float32) for c, d in seg.items() for m, v in d.items()},
        snippet=X[mid: mid + int(0.2 * fs)].astype(np.float32), speed_track=trk.astype(np.float32),
    )
    (cache_dir / f"vib_{path.stem}.json").write_text(json.dumps(out, indent=1, default=_json_default))
    del X, rec
    return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
