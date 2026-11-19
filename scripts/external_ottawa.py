"""External validation on Ottawa UORED-VAFCLS (protocol EV-V1). NEVER trains on or tunes with Ottawa.

  python -I scripts/external_ottawa.py --stage audit      # label-free integrity + speed-channel audit (before freeze)
  python -I scripts/external_ottawa.py --stage features   # PU-final physics artefacts + Ottawa features (no labels used)
  python -I scripts/external_ottawa.py --stage evaluate   # single evaluation with frozen models (refuses to run twice)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402

from pave.data import ottawa as OT  # noqa: E402
from pave.features import physics as PH  # noqa: E402
from pave.features import physics_v3 as V3  # noqa: E402
from pave.signal.envelope import envelope  # noqa: E402
from pave.signal.physics import BearingGeometry, orders  # noqa: E402

OUT = ROOT / "results" / "phase4"
RAW = ROOT / "01_RAW_DATA" / "Ottawa_data"
BAND = (1000.0, 20000.0)
KIN_OT = {k: v for k, v in orders(BearingGeometry("Ottawa 6203", 8, 6.77, 28.50, 0.0)).items() if k in ("BPFO", "BPFI")}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def env_spec(x, fs, n_fft, keep_hz=1000.0):
    e = envelope(x, fs, BAND)
    w = np.hanning(len(e))
    E = np.abs(np.fft.rfft(e * w, n_fft)) * 2.0 / w.sum()
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    k = f <= keep_hz + 1e-9
    return f[k], E[k]


def stage_audit():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in OT.find_files(RAW):
        info = OT.parse_stem(p.stem)
        a = OT.read_ottawa(p)
        sp = a[:, 2]
        rows.append({**info, "rpm_sample0": float(sp[0]), "load_sample0": float(a[0, 3]), "file": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "n_samples": a.shape[0],
                     "duration_s": a.shape[0] / OT.FS, "n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum()),
                     "speed_rpm_median": float(np.median(sp)), "speed_rpm_mean": float(sp.mean()), "speed_rpm_std": float(sp.std()),
                     "speed_rpm_p1": float(np.percentile(sp, 1)), "speed_rpm_p99": float(np.percentile(sp, 99)),
                     "speed_n_unique": int(len(np.unique(sp))), "load_N_median": float(np.median(a[:, 3])),
                     "accel_rms": float(np.sqrt(np.mean(a[:, 0] ** 2))), "accel_max_abs": float(np.abs(a[:, 0]).max()),
                     "temp_diff_median": float(np.median(a[:, 4]))})
    A = pd.DataFrame(rows).sort_values(["bearing", "state"])
    A.to_csv(OUT / "ottawa_audit.csv", index=False)
    print(A[["stem", "manufacturer", "n_samples", "n_nan", "speed_rpm_median", "speed_rpm_std", "speed_rpm_p1", "speed_rpm_p99",
             "speed_n_unique", "load_N_median"]].to_string())
    print("files", len(A), "bearings", A.bearing.nunique())


def label_free_speed_check(x, fs, fr_channel):
    """Shaft frequency from the raw accelerometer spectrum: strongest peak within +-10 % of the speed-channel value
    (label-free cross-check, reported only)."""
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x))))
    m = (f > fr_channel * 0.9) & (f < fr_channel * 1.1)
    return float(f[m][np.argmax(X[m])])


def stage_features():
    """Ottawa features with FINAL PU calibration (all 15 PU-BD5 bearings). Labels are not read."""
    from pave.experiments import phase2 as P2
    from pave.experiments import phase3a as P3A
    fin = json.loads((ROOT / "results/phase3b/folds/FINAL.json").read_text())
    reg4 = PH.Regions(**fin["regions"]["regions_4s"])
    reg1 = PH.Regions(**fin["regions"]["regions_1s"])
    reg2 = PH.Regions(**fin["regions"]["regions_v2"])
    rows, W = [], {"raw32": [], "v2": [], "v3_1s_c": [], "v3_1s_m": [], "v3_1s_q": []}
    S4 = {"c": [], "m": [], "q": []}
    for p in OT.find_files(RAW):
        info = OT.parse_stem(p.stem)
        a = OT.read_ottawa(p)
        x = a[:, 0].astype(np.float64)
        rpm0 = float(a[0, 2])  # EV-V1 sec.3: speed stored once per file at sample 0
        speed_flag = not (600.0 <= rpm0 <= 3000.0)
        fr = (rpm0 if not speed_flag else 1750.0) / 60.0
        sig = 0.5 / 60.0  # half of the 1-rpm reading resolution
        fr_vib = label_free_speed_check(x, OT.FS, fr)
        n4 = int(4 * OT.FS)
        for seg in range(2):
            xs = x[seg * n4:(seg + 1) * n4]
            f4, E4 = env_spec(xs, OT.FS, n4)
            r = V3.features_v3(f4, E4, fr, KIN_OT, reg4, sig)
            S4["c"].append(r.c)
            S4["m"].append(r.m)
            S4["q"].append(r.q)
            rs = V3.features_v3(f4, E4, fr, KIN_OT, reg4, 0.01 * fr)  # pre-specified sensitivity (sigma_r = 1 % of f_r)
            S4.setdefault("c_sens", []).append(rs.c)
            S4.setdefault("m_sens", []).append(rs.m)
            S4.setdefault("q_sens", []).append(rs.q)
            rows.append({**{k: info[k] for k in ("stem", "bearing", "state", "manufacturer")}, "segment4": seg, "fr_hz": fr,
                         "sigma_fr_hz": sig, "fr_vib_check_hz": fr_vib, "rpm_sample0": rpm0, "speed_flag": speed_flag, "load_sample0": float(a[0, 3]), "valid_BPFO": r.info["valid_BPFO"], "valid_BPFI": r.info["valid_BPFI"],
                         "delta_BPFO": r.info["delta_BPFO"], "delta_BPFI": r.info["delta_BPFI"]})
        n1 = int(OT.FS)
        for k in range(int(len(x) // n1)):
            xw = x[k * n1:(k + 1) * n1]
            W["raw32"].append(signal.resample_poly(xw, 16, 21).astype(np.float16)[None])
            f1, E1 = env_spec(xw, OT.FS, n1)
            d2 = PH.window_features(f1[:1001], E1[:1001], fr, KIN_OT, reg2)
            W["v2"].append([d2[c] for c in PH.FEATURE_NAMES])
            r1 = V3.features_v3(f1[:1001], E1[:1001], fr, KIN_OT, reg1, sig)
            W["v3_1s_c"].append(r1.c)
            W["v3_1s_m"].append(r1.m)
            W["v3_1s_q"].append(r1.q)
            W.setdefault("stem", []).append(info["stem"])
            W.setdefault("seg4", []).append(min(k // 4, 1) if k < 8 else -1)  # windows 8,9 not covered by the 4-s segments
    seg = pd.DataFrame(rows)
    seg.to_csv(OUT / "ottawa_segments4s.csv", index=False)
    np.savez_compressed(OUT / "ottawa_features.npz", C4=np.array(S4["c"]), M4=np.array(S4["m"]), Q4=np.array(S4["q"]),
                        C4_sens=np.array(S4["c_sens"]), M4_sens=np.array(S4["m_sens"]), Q4_sens=np.array(S4["q_sens"]),
                        raw32=np.stack(W["raw32"]), v2=np.array(W["v2"]), v3_1s_c=np.array(W["v3_1s_c"]),
                        v3_1s_m=np.array(W["v3_1s_m"]), v3_1s_q=np.array(W["v3_1s_q"]), stem=np.array(W["stem"]), seg4=np.array(W["seg4"]))
    print(seg.groupby("stem")[["fr_hz", "fr_vib_check_hz", "valid_BPFO", "valid_BPFI"]].mean().describe().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["audit", "features", "evaluate"])
    a = ap.parse_args()
    t0 = time.time()
    {"audit": stage_audit, "features": stage_features}.get(a.stage, lambda: (_ for _ in ()).throw(SystemExit("evaluate: see external_ottawa_eval.py")))()
    print(f"done in {time.time()-t0:.0f}s")
