"""Cross-file analyses for the Phase-1 audit (operates on cached per-file results).

Produces manifests (03_MANIFESTS), tables + figures (results/audit) and a
machine-readable ``audit_summary.json`` from which the written reports quote
numbers. No class label is used to compute any physical evidence; labels are
only used afterwards to TABULATE evidence against expectations.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pave.data.alignment import estimate_clock_offsets, overlap_fraction
from pave.data.conditions import parse_filename, parse_internal_label, same_condition
from pave.data.manifest import DATASET_DIR, DATASET_DOI, relpath
from pave.data.split import Rec, check_split, lolo_splits, permutation_splits, split_to_json, within_recording_split
from pave.signal.physics import NSK_6205_DDU, PAPER_RPM, all_frequencies, orders

PRIMARY_BAND = "FIXED"
PRIMARY_CHANNELS = ("x_housing_A", "y_housing_A")
DET_EXCESS_DB = 6.0           # target must exceed control median by >= 6 dB ...
DET_CTRL_EXCEED_MAX = 0.0     # ... and exceed ALL 24 control frequencies (record level)
WIN_CTRL_EXCEED_MAX = 1 / 24  # window level: at most one control >= target
TIME_TOL_S = 5.0              # clock-corrected start mismatch tolerated for concurrency
SESSION_NAMES = {"2021-06-25": "S1 (25 Jun)", "2021-06-28": "S2 (28 Jun)", "2021-06-30": "S3 (30 Jun)"}


# ----------------------------------------------------------------------------- loading
def _load(cache: Path):
    vib = {p.stem[4:]: json.loads(p.read_text()) for p in sorted(cache.glob("vib_*.json"))}
    tdms = {p.stem[5:]: json.loads(p.read_text()) for p in sorted(cache.glob("tdms_*.json"))}
    return vib, tdms


def _canon_stem(stem: str) -> str:
    return stem.replace("Unbalalnce", "Unbalance")


def _iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


def _motor_fr(t: dict) -> float:
    vals = []
    for c in t["current"].values():
        if c.get("available"):
            vals += [c["motor_fr_usb_hz"], c["motor_fr_lsb_hz"]]
    return float(np.median(vals)) if vals else np.nan


def _supply(t: dict) -> float:
    vals = [c["supply_hz"] for c in t["current"].values() if c.get("available")]
    return float(np.median(vals)) if vals else np.nan


# ----------------------------------------------------------------------------- pairing
def build_pairs(vib: dict, tdms: dict) -> tuple[pd.DataFrame, dict]:
    vrows = []
    for stem, v in vib.items():
        c = parse_filename(_canon_stem(stem))
        ci = parse_internal_label(v["run_name"])
        vrows.append({"vib_stem": stem, "canon": _canon_stem(stem), "cond": c, "vib_internal": v["run_name"],
                      "vib_internal_cond": ci, "vib_start": _iso(v["absolute_time"]), "vib_dur": v["duration_s"],
                      "fr_vib": v["fr_hz"]})
    trows = {}
    for stem, t in tdms.items():
        c = parse_filename(stem)
        ci = parse_internal_label(t["file_name_prop"])
        trows[stem] = {"cond": c, "internal": t["file_name_prop"], "internal_cond": ci,
                       "start": _iso(t["temp_start_utc"]),
                       "dur": t["temperature"]["T_housing_A"]["duration_s"],
                       "motor_fr": _motor_fr(t), "supply": _supply(t), "run": t["run_number"],
                       "phases": sum(1 for c_ in t["current"].values() if c_.get("available"))}

    same = same_condition

    # 1) filename-matched candidates and label consistency
    base = []
    for r in vrows:
        t = trows[r["canon"]]
        off = (r["vib_start"] - t["start"]).total_seconds()
        vib_lbl_ok = r["vib_internal_cond"] is None or same(r["cond"], r["vib_internal_cond"])
        tdms_lbl_ok = same(t["cond"], t["internal_cond"])
        base.append({**r, "tdms_by_name": r["canon"], "offset_by_name_s": off,
                     "vib_label_ok": vib_lbl_ok, "tdms_label_ok": tdms_lbl_ok,
                     "session": r["vib_start"].date().isoformat()})
    sessions = [b["session"] for b in base]
    offs = [b["offset_by_name_s"] for b in base]
    ok = [b["vib_label_ok"] and b["tdms_label_ok"] for b in base]
    sess_off = estimate_clock_offsets(sessions, offs, ok, max_dev_s=TIME_TOL_S)
    # gear ratio from label- and time-consistent pairs
    ratios = []
    for b in base:
        so = sess_off[b["session"]].offset_s
        if b["vib_label_ok"] and b["tdms_label_ok"] and abs(b["offset_by_name_s"] - so) <= TIME_TOL_S:
            ratios.append(b["fr_vib"] / trows[b["tdms_by_name"]]["motor_fr"])
    G = float(np.median(ratios))
    G_spread = float(np.std(ratios))

    rows = []
    for b in base:
        so = sess_off[b["session"]]
        # all TDMS of the same calendar day (TDMS clock) as candidates
        cands = []
        for ts, t in trows.items():
            if t["start"].date().isoformat() != b["session"]:
                continue
            resid = (b["vib_start"] - t["start"]).total_seconds() - so.offset_s
            dfr = b["fr_vib"] - G * t["motor_fr"]
            cands.append((ts, resid, dfr))
        by_time = min(cands, key=lambda c: abs(c[1]))
        by_speed = min(cands, key=lambda c: abs(c[2]))
        name_c = next(c for c in cands if c[0] == b["tdms_by_name"])
        t_name = trows[b["tdms_by_name"]]
        time_ok_name = abs(name_c[1]) <= TIME_TOL_S
        if time_ok_name and b["tdms_label_ok"]:
            status, resolved = "CONCURRENT_BY_NAME", b["tdms_by_name"]
        elif abs(by_time[1]) <= TIME_TOL_S and same(trows[by_time[0]]["internal_cond"], b["cond"]):
            status, resolved = "FILENAME_SWAP_RESOLVED", by_time[0]
        else:
            status, resolved = "NON_CONCURRENT", b["tdms_by_name"]
        tr = trows[resolved]
        res_c = next(c for c in cands if c[0] == resolved)
        vib_on_tdms_clock = (b["vib_start"] - dt.timedelta(seconds=so.offset_s))
        ovl = overlap_fraction(0.0, b["vib_dur"], (tr["start"] - vib_on_tdms_clock).total_seconds(), tr["dur"])
        rows.append({
            "recording_id": b["cond"].recording_id, "vib_file": b["vib_stem"] + ".mat",
            "vib_internal_label": b["vib_internal"], "vib_label_consistent": b["vib_label_ok"],
            "session": b["session"], "vib_start_utc": b["vib_start"].isoformat(),
            "tdms_by_filename": b["tdms_by_name"] + ".tdms", "tdms_by_filename_internal": t_name["internal"],
            "tdms_by_filename_label_consistent": b["tdms_label_ok"],
            "raw_offset_by_filename_s": round(b["offset_by_name_s"], 3),
            "session_clock_offset_s": round(so.offset_s, 3), "session_clock_offset_sigma_s": round(so.sigma_s, 3),
            "residual_by_filename_s": round(name_c[1], 3),
            "nearest_in_time_tdms": by_time[0] + ".tdms", "nearest_in_time_residual_s": round(by_time[1], 3),
            "best_speed_match_tdms": by_speed[0] + ".tdms",
            "resolved_tdms": resolved + ".tdms", "resolved_tdms_internal": tr["internal"],
            "pairing_status": status,
            "resolved_residual_s": round(res_c[1], 3),
            "temp_overlap_fraction_if_clock_offset": round(ovl, 3),
            "fr_vib_hz": round(b["fr_vib"], 5), "fr_from_current_hz": round(G * tr["motor_fr"], 5),
            "speed_mismatch_hz": round(res_c[2], 5), "supply_hz": round(tr["supply"], 4),
            "tdms_current_phases": tr["phases"], "tdms_run_number": tr["run"],
        })
    df = pd.DataFrame(rows).sort_values("recording_id").reset_index(drop=True)
    info = {"gear_ratio": G, "gear_ratio_std": G_spread, "n_ratio_pairs": len(ratios),
            "session_offsets": {k: vars(v) for k, v in sess_off.items()}}
    return df, info


# ----------------------------------------------------------------------------- manifests
def build_recordings(vib: dict, tdms: dict, pairs: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    sha = dict(zip(inv.file_name, inv.sha256))
    rel = dict(zip(inv.file_name, inv.relative_path))
    P = pairs.set_index("recording_id")
    rows = []
    for stem, v in vib.items():
        c = parse_filename(_canon_stem(stem))
        p = P.loc[c.recording_id]
        t = tdms[p.resolved_tdms[:-5]]
        start = _iso(v["absolute_time"])
        rows.append({
            "recording_id": c.recording_id, "dataset": "KAIST varying-load (Jung et al. 2023)",
            "dataset_version": DATASET_DOI + " (version from blueprint; not encoded in local files)",
            "load_Nm": c.load_Nm, "fault_group": c.fault_group, "severity_value": c.severity_value,
            "severity_unit": c.severity_unit, "label7": c.label7, "label3": c.label3,
            "condition_id": c.condition_id, "in_bearing_scope": c.in_bearing_scope,
            "vib_relative_path": rel[v["file"]], "vib_sha256": sha[v["file"]],
            "vib_channels": "|".join(f"{a}->{b}" for a, b in zip(v["channel_labels"], v["channel_names"])),
            "vib_n_samples": v["n_samples"], "vib_fs_hz": v["fs_hz"], "vib_duration_s": v["duration_s"],
            "vib_time_start_utc": start.isoformat(),
            "vib_time_end_utc": (start + dt.timedelta(seconds=v["duration_s"])).isoformat(),
            "vib_unit_label": v["unit_label"], "vib_stored_unit_per_metadata": "m/s^2 (Test.Lab MKS; factor to g = %.8f)" % v["unit_factor"],
            "shaft_fr_measured_hz": round(v["fr_hz"], 5),
            "tdms_relative_path": rel[p.resolved_tdms], "tdms_sha256": sha[p.resolved_tdms],
            "tdms_by_filename": p.tdms_by_filename, "pairing_status": p.pairing_status,
            "temp_concurrent": p.pairing_status in ("CONCURRENT_BY_NAME", "FILENAME_SWAP_RESOLVED"),
            "temp_n_samples": t["temperature"]["T_housing_A"]["n_samples"],
            "temp_nominal_rate_hz": round(t["temp_nominal_rate_hz"], 3),
            "temp_effective_rate_hz": round(t["temperature"]["T_housing_A"]["effective_rate_hz"], 3),
            "temp_duration_s": t["temperature"]["T_housing_A"]["duration_s"],
            "temp_time_start_utc_tdms_clock": t["temp_start_utc"], "temp_unit": "degC",
            "session": p.session,
        })
    df = pd.DataFrame(rows).sort_values(["in_bearing_scope", "load_Nm", "condition_id"],
                                        ascending=[False, True, True]).reset_index(drop=True)
    # acquisition order within session (vib clock)
    df["acq_order_in_session"] = df.groupby("session")["vib_time_start_utc"].rank().astype(int)
    first = df.groupby("session")["vib_time_start_utc"].transform("min")
    df["minutes_since_session_first"] = [(_iso(a) - _iso(b)).total_seconds() / 60 for a, b in zip(df.vib_time_start_utc, first)]
    return df


def signal_quality_table(vib: dict) -> pd.DataFrame:
    rows = []
    for stem, v in vib.items():
        c = parse_filename(_canon_stem(stem))
        for ch, s in v["channels"].items():
            rows.append({"recording_id": c.recording_id, "label7": c.label7, "condition_id": c.condition_id,
                         "load_Nm": c.load_Nm, "channel": ch, **{k: s[k] for k in (
                             "n", "n_nan", "n_inf", "mean", "std", "rms", "min", "max", "peak_abs", "p2p",
                             "crest_factor", "kurtosis", "skewness", "longest_constant_run", "quant_step",
                             "is_constant", "extreme_frac", "extreme_longest_run", "clipping_suspect",
                             "seg_rms_cv", "seg_rms_drift_rel", "seg_kurt_median", "seg_kurt_iqr",
                             "seg_mean_abs_max")},
                         "fr_hz": v["fr_hz"], "fr_track_std_hz": v["fr_track_std_hz"],
                         "duplicate_channel_pairs": len(v["duplicate_channel_pairs"])})
    return pd.DataFrame(rows).sort_values(["recording_id", "channel"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- temperature
def temperature_tables(tdms: dict, rec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for stem, t in tdms.items():
        c = parse_filename(stem)
        for ch, s in t["temperature"].items():
            rows.append({"tdms_file": stem + ".tdms", "filename_recording_id": c.recording_id,
                         "internal_name": t["file_name_prop"], "channel": ch, **s,
                         "A_minus_B_mean": t["T_A_minus_B"]["mean"], "A_B_corr_detrended": t.get("T_A_B_corr_detrended"),
                         "updates_simultaneous_AB": t.get("T_updates_simultaneous"),
                         "start_utc": t["temp_start_utc"]})
    per_file = pd.DataFrame(rows)
    # record-level (using RESOLVED pairing)
    R = []
    for _, r in rec.iterrows():
        t = tdms[Path(r.tdms_relative_path).stem]
        a, b = t["temperature"]["T_housing_A"], t["temperature"]["T_housing_B"]
        R.append({"recording_id": r.recording_id, "label7": r.label7, "condition_id": r.condition_id,
                  "fault_group": r.fault_group, "load_Nm": r.load_Nm, "in_bearing_scope": r.in_bearing_scope,
                  "session": r.session, "minutes_since_session_first": r.minutes_since_session_first,
                  "temp_concurrent": r.temp_concurrent,
                  "TA_mean": a["mean"], "TB_mean": b["mean"], "TA_first5": a["first5s_mean"], "TB_first5": b["first5s_mean"],
                  "TA_slope_C_min": a["slope_C_per_min"], "TB_slope_C_min": b["slope_C_per_min"],
                  "TA_rise": a["rise_C"], "TB_rise": b["rise_C"], "A_minus_B": a["mean"] - b["mean"],
                  "TA_noise": a["resid_noise_std_C"], "TB_noise": b["resid_noise_std_C"],
                  "temp_duration_s": a["duration_s"], "tdms_start_utc": t["temp_start_utc"]})
    rl = pd.DataFrame(R)
    # variance decomposition (descriptive eta^2) on the 21 in-scope recordings
    ins = rl[rl.in_bearing_scope].copy()
    eta = {}
    for y in ("TA_mean", "TB_mean", "A_minus_B", "TA_slope_C_min"):
        tot = ((ins[y] - ins[y].mean()) ** 2).sum()
        eta[y] = {}
        for f in ("label7", "fault_group", "load_Nm", "session"):
            g = ins.groupby(f)[y]
            ssb = (g.count() * (g.mean() - ins[y].mean()) ** 2).sum()
            eta[y][f] = float(ssb / tot) if tot > 0 else np.nan
    # within-session association of temperature with acquisition time
    assoc = {}
    for s, g in rl.groupby("session"):
        if len(g) > 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(g.minutes_since_session_first, g.TA_first5)
            assoc[s] = {"n": len(g), "spearman_TAstart_vs_time": float(rho), "p": float(p),
                        "TA_first5_min": float(g.TA_first5.min()), "TA_first5_max": float(g.TA_first5.max())}
    # class-vs-session confounding table
    conf = pd.crosstab(ins.fault_group, ins.session)
    info = {"eta2": eta, "temp_vs_time_by_session": assoc, "class_by_session": conf.to_dict()}
    return per_file, rl, info


# ----------------------------------------------------------------------------- physics
def physics_tables(vib: dict, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, wrows = [], []
    for stem, v in vib.items():
        c = parse_filename(_canon_stem(stem))
        for key, e in v["evidence"].items():
            ch, band, tgt = key.split("|")
            rows.append({"recording_id": c.recording_id, "label7": c.label7, "fault_group": c.fault_group,
                         "condition_id": c.condition_id, "severity": c.severity_value, "load_Nm": c.load_Nm,
                         "in_bearing_scope": c.in_bearing_scope, "channel": ch, "band": band, "target": tgt,
                         "fr_hz": v["fr_hz"], **{k: e.get(k) for k in (
                             "snr_db", "peak_hz", "peak_ratio", "ctrl_median_db", "ctrl_q95_db", "ctrl_max_db",
                             "excess_db", "robust_z", "ctrl_exceed_frac", "excess_db_tol0.005", "excess_db_tol0.02")}})
        npz = np.load(cache / f"vib_{stem}.npz")
        for key, s in v["window_evidence_summary"].items():
            ch, band, tgt = key.split("|")
            W = npz[f"win|{key}"]
            det = (W[:, 0] >= DET_EXCESS_DB) & (W[:, 2] <= WIN_CTRL_EXCEED_MAX + 1e-9)
            wrows.append({"recording_id": c.recording_id, "label7": c.label7, "fault_group": c.fault_group,
                          "condition_id": c.condition_id, "load_Nm": c.load_Nm, "in_bearing_scope": c.in_bearing_scope,
                          "channel": ch, "band": band, "target": tgt, "n_windows": int(W.shape[0]),
                          "frac_detected": float(det.mean()), "median_excess_db": float(np.median(W[:, 0])),
                          "q10_excess_db": float(np.quantile(W[:, 0], 0.1)),
                          "q90_excess_db": float(np.quantile(W[:, 0], 0.9))})
    ev = pd.DataFrame(rows)
    wev = pd.DataFrame(wrows)
    # record-level decision per recording (primary definition, label-free)
    dec = []
    for rid, g in ev[(ev.band == PRIMARY_BAND) & ev.channel.isin(PRIMARY_CHANNELS)].groupby("recording_id"):
        def det(t):
            s = g[g.target == t]
            ok = (s.excess_db >= DET_EXCESS_DB) & (s.ctrl_exceed_frac <= DET_CTRL_EXCEED_MAX)
            return bool(ok.any()), float(s.excess_db.max())
        o, oe = det("BPFO_1")
        i, ie = det("BPFI_1")
        o2, _ = det("BPFO_2")
        i2, _ = det("BPFI_2")
        first = g.iloc[0]
        call = "both" if (o and i) else "OR-like" if o else "IR-like" if i else "none"
        expected = {"Healthy": "none", "IR": "IR-like", "OR": "OR-like"}.get(first.fault_group, "none")
        dec.append({"recording_id": rid, "label7": first.label7, "fault_group": first.fault_group,
                    "condition_id": first.condition_id, "load_Nm": first.load_Nm,
                    "in_bearing_scope": first.in_bearing_scope, "fr_hz": first.fr_hz,
                    "BPFO_1_detected": o, "BPFO_1_max_excess_db": oe, "BPFI_1_detected": i,
                    "BPFI_1_max_excess_db": ie, "BPFO_2_detected": o2, "BPFI_2_detected": i2,
                    "physics_call": call, "expected_call": expected, "agrees_with_label": call == expected})
    return ev, wev, pd.DataFrame(dec).sort_values(["in_bearing_scope", "load_Nm", "condition_id"],
                                                  ascending=[False, True, True])


def collision_table(fr: float = PAPER_RPM / 60.0, tol: float = 0.01) -> pd.DataFrame:
    """Proximity of each target to the other fault family's components and to shaft harmonics."""
    o = orders()
    rows = []
    fam = {"BPFO": o["BPFO"], "BPFI": o["BPFI"]}
    for key, base in fam.items():
        other = "BPFI" if key == "BPFO" else "BPFO"
        for h in (1, 2, 3):
            t = h * base
            comps = [(f"{hh}x{other}{'+' if m >= 0 else ''}{m}fr" if m else f"{hh}x{other}", hh * fam[other] + m)
                     for hh in (1, 2, 3) for m in (-2, -1, 0, 1, 2)]
            nearest = min(comps, key=lambda c: abs(c[1] - t))
            k = round(t)
            rows.append({"target": f"{key}_{h}", "order": t, "hz_at_paper_speed": t * fr,
                         "tolerance_hz": tol * t * fr,
                         "nearest_shaft_harmonic": k, "dist_to_shaft_harmonic_hz": abs(t - k) * fr,
                         "nearest_other_family_component": nearest[0],
                         "dist_to_other_family_hz": abs(nearest[1] - t) * fr,
                         "ambiguous_within_2tol": abs(nearest[1] - t) * fr < 2 * tol * t * fr})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- splits
def build_splits(rec: pd.DataFrame, man: Path) -> dict:
    sd = man / "splits"
    sd.mkdir(parents=True, exist_ok=True)
    ins = rec[rec.in_bearing_scope]
    recs = [Rec(r.recording_id, int(r.load_Nm), float(r.vib_duration_s)) for r in ins.itertuples()]
    sps = lolo_splits(recs, val_frac=0.2, purge_s=2.0) + permutation_splits(recs) + [within_recording_split(recs)]
    index = []
    meta = ins.set_index("recording_id")
    for sp in sps:
        check_split(sp, max_window_s=2.0)
        d = json.loads(split_to_json(sp))
        for a in d["assignments"]:
            m = meta.loc[a["recording_id"]]
            a.update({"label7": m.label7, "label3": m.label3, "load_Nm": int(m.load_Nm),
                      "vib_relative_path": m.vib_relative_path, "vib_sha256": m.vib_sha256,
                      "tdms_relative_path": m.tdms_relative_path, "temp_concurrent": bool(m.temp_concurrent)})
        (sd / f"split_{sp.fold_id}.json").write_text(json.dumps(d, indent=2))
        roles = {r: sorted(sp.recordings(r)) for r in ("train", "val", "test")}
        secs = {r: float(sum(a.interval.duration_s for a in sp.roles(r))) for r in ("train", "val", "test")}
        index.append({"protocol": sp.protocol, "fold_id": sp.fold_id, "content_sha256": d["content_sha256"],
                      "n_train_rec": len(roles["train"]), "n_val_rec": len(roles["val"]), "n_test_rec": len(roles["test"]),
                      "train_s": secs["train"], "val_s": secs["val"], "test_s": secs["test"],
                      "leakage_check": "PASS"})
    pd.DataFrame(index).to_csv(sd / "split_index.csv", index=False)
    # out-of-scope challenge set (never used for tuning)
    oos = rec[~rec.in_bearing_scope][["recording_id", "condition_id", "load_Nm", "vib_relative_path", "vib_sha256",
                                       "tdms_relative_path", "temp_concurrent"]]
    oos.to_csv(sd / "challenge_set_out_of_scope.csv", index=False)
    # class-time balance per LOLO fold
    bal = []
    for sp in sps[:3]:
        for a in sp.assignments:
            bal.append({"fold_id": sp.fold_id, "role": a.role, "label7": meta.loc[a.recording_id].label7,
                        "seconds": a.interval.duration_s})
    bal = pd.DataFrame(bal).pivot_table(index=["fold_id", "role"], columns="label7", values="seconds", aggfunc="sum")
    bal.to_csv(sd / "lolo_class_seconds.csv")
    return {"index": index, "n_out_of_scope": len(oos)}


# ----------------------------------------------------------------------------- main
def run_all(cache: Path, out: Path, man: Path, root: Path) -> None:
    from . import figures

    tab = out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    vib, tdms = _load(cache)
    assert len(vib) == 45 and len(tdms) == 45, (len(vib), len(tdms))
    inv = pd.read_csv(man / "file_inventory.csv")
    pairs, pinfo = build_pairs(vib, tdms)
    pairs.to_csv(man / "pairs.csv", index=False)
    rec = build_recordings(vib, tdms, pairs, inv)
    rec.to_csv(man / "recordings.csv", index=False)
    sq = signal_quality_table(vib)
    sq.to_csv(tab / "signal_quality.csv", index=False)
    tf, trec, tinfo = temperature_tables(tdms, rec)
    tf.to_csv(tab / "temperature_per_file.csv", index=False)
    trec.to_csv(tab / "temperature_per_recording.csv", index=False)
    ev, wev, dec = physics_tables(vib, cache)
    ev.to_csv(tab / "physics_evidence_record_level.csv", index=False)
    wev.to_csv(tab / "physics_evidence_window_level.csv", index=False)
    dec.to_csv(tab / "physics_calls_record_level.csv", index=False)
    col = collision_table()
    col.to_csv(tab / "frequency_collisions.csv", index=False)
    # speed / bearing frequencies at measured speed
    spd = rec[["recording_id", "label7", "condition_id", "load_Nm", "shaft_fr_measured_hz"]].copy()
    for k in ("BPFO", "BPFI"):
        spd[f"{k}_measured_speed_hz"] = [all_frequencies(f)[k] for f in spd.shaft_fr_measured_hz]
        spd[f"{k}_paper_hz"] = all_frequencies(PAPER_RPM / 60)[k]
        spd[f"{k}_shift_hz"] = spd[f"{k}_measured_speed_hz"] - spd[f"{k}_paper_hz"]
    spd.to_csv(tab / "shaft_speed_and_fault_frequencies.csv", index=False)
    splits = build_splits(rec, man)
    # dataset distribution table
    dist = rec.pivot_table(index="condition_id", columns="load_Nm", values="vib_duration_s", aggfunc="sum")
    dist.to_csv(tab / "duration_by_condition_load.csv")

    # summary numbers
    ins = rec[rec.in_bearing_scope]
    win_primary = wev[(wev.band == PRIMARY_BAND) & wev.channel.isin(PRIMARY_CHANNELS)]
    summary = {
        "n_vibration_files": len(vib), "n_tdms_files": len(tdms),
        "n_recordings_total": len(rec), "n_recordings_in_scope": int(len(ins)),
        "in_scope_by_class": ins.label7.value_counts().to_dict(),
        "in_scope_by_load": ins.load_Nm.value_counts().to_dict(),
        "durations_s_in_scope": ins.groupby("label7").vib_duration_s.apply(list).to_dict(),
        "pairing": pairs.pairing_status.value_counts().to_dict(),
        "pairing_info": pinfo,
        "non_concurrent": pairs[pairs.pairing_status == "NON_CONCURRENT"][["recording_id", "resolved_residual_s", "speed_mismatch_hz"]].to_dict("records"),
        "swaps": pairs[pairs.pairing_status == "FILENAME_SWAP_RESOLVED"][["recording_id", "tdms_by_filename", "resolved_tdms"]].to_dict("records"),
        "speed_mismatch_abs_max_hz_concurrent": float(pairs[pairs.pairing_status != "NON_CONCURRENT"].speed_mismatch_hz.abs().max()),
        "vib_any_nan_inf": int(sq.n_nan.sum() + sq.n_inf.sum()),
        "vib_clipping_suspect": sq[sq.clipping_suspect].recording_id.unique().tolist(),
        "vib_constant": sq[sq.is_constant].recording_id.unique().tolist(),
        "vib_duplicate_channels": int(sq.duplicate_channel_pairs.sum()),
        "vib_fs_unique": sorted({v["fs_hz"] for v in vib.values()}),
        "temp_effective_rate_hz_unique": sorted(set(rec.temp_effective_rate_hz.round(2))),
        "temp_nominal_rate_hz_unique": sorted(set(rec.temp_nominal_rate_hz.round(2))),
        "temp_eta2": tinfo["eta2"], "temp_vs_time": tinfo["temp_vs_time_by_session"],
        "temp_class_by_session": tinfo["class_by_session"],
        "physics_calls_in_scope": dec[dec.in_bearing_scope].groupby(["fault_group", "physics_call"]).size().unstack(fill_value=0).to_dict(),
        "physics_agreement_in_scope": float(dec[dec.in_bearing_scope].agrees_with_label.mean()),
        "physics_calls_out_of_scope": dec[~dec.in_bearing_scope].physics_call.value_counts().to_dict(),
        "window_detect_primary": win_primary[win_primary.in_bearing_scope].groupby(["label7", "target"]).frac_detected.max().unstack().to_dict(),
        "speed_by_load": rec.groupby("load_Nm").shaft_fr_measured_hz.agg(["min", "max", "mean"]).to_dict(),
        "splits": splits,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }
    summary["family_search"] = family_analysis(vib, cache, tab)
    summary["temperature_shortcut_probe"] = temperature_shortcut_probe(trec, tab)
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    figures.make_all(cache, out / "figures", rec, pairs, trec, ev, wev, dec, sq, tdms, vib)
    print("analysis complete")


# ----------------------------------------------------------------------------- family-search physics (post-hoc)
FAM_CONTRAST_DB = 6.0
FAM_HEALTHY_Q = 0.99


def _fam_windows(vib: dict, cache: Path, band: str) -> pd.DataFrame:
    """Window-level family scores (max over housing-A x/y). score = mean(fundamental SNR, 2nd-harmonic SNR)."""
    rows = []
    for stem, v in vib.items():
        c = parse_filename(_canon_stem(stem))
        z = np.load(cache / f"vib_{stem}.npz")
        sc, od = {}, {}
        for fam in ("BPFO", "BPFI"):
            arrs = [z[f"win|{ch}|{band}|FAM_{fam}"] for ch in PRIMARY_CHANNELS]
            s = np.stack([np.nanmean(a[:, [0, 2]], axis=1) for a in arrs])  # (2, k)
            j = np.argmax(s, axis=0)
            sc[fam] = s.max(axis=0)
            od[fam] = np.stack([a[:, 1] for a in arrs])[j, np.arange(s.shape[1])]
        for i in range(len(sc["BPFO"])):
            rows.append({"recording_id": c.recording_id, "label7": c.label7, "fault_group": c.fault_group,
                         "load_Nm": c.load_Nm, "in_bearing_scope": c.in_bearing_scope, "window": i,
                         "score_O": sc["BPFO"][i], "score_I": sc["BPFI"][i],
                         "order_O": od["BPFO"][i], "order_I": od["BPFI"][i]})
    return pd.DataFrame(rows)


def _call(sO, sI, tO, tI, contrast=FAM_CONTRAST_DB):
    o = (sO >= tO) & (sO - sI >= contrast)
    i = (sI >= tI) & (sI - sO >= contrast)
    amb = (sO >= tO) & (sI >= tI) & ~o & ~i
    return np.where(o, "OR-like", np.where(i, "IR-like", np.where(amb, "ambiguous", "none")))


def family_analysis(vib: dict, cache: Path, tab: Path) -> dict:
    res = {}
    for band in ("FIXED", "SK"):
        W = _fam_windows(vib, cache, band)
        H = W[W.fault_group == "Healthy"]
        tO, tI = float(H.score_O.quantile(FAM_HEALTHY_Q)), float(H.score_I.quantile(FAM_HEALTHY_Q))
        W["call"] = _call(W.score_O.values, W.score_I.values, tO, tI)
        W["expected"] = W.fault_group.map({"Healthy": "none", "IR": "IR-like", "OR": "OR-like"}).fillna("none")
        W.to_csv(tab / f"physics_family_windows_{band}.csv", index=False)
        per_rec = (W.groupby(["recording_id", "label7", "fault_group", "load_Nm", "in_bearing_scope", "expected"])
                   .call.value_counts(normalize=True).unstack(fill_value=0).reset_index())
        for col in ("none", "IR-like", "OR-like", "ambiguous"):
            if col not in per_rec:
                per_rec[col] = 0.0
        per_rec["frac_expected"] = [r[r.expected] for _, r in per_rec.iterrows()]
        med = W.groupby("recording_id").agg(med_score_O=("score_O", "median"), med_score_I=("score_I", "median"),
                                            med_order_O=("order_O", "median"), med_order_I=("order_I", "median"),
                                            n_windows=("window", "count")).reset_index()
        per_rec = per_rec.merge(med, on="recording_id")
        per_rec.to_csv(tab / f"physics_family_per_recording_{band}.csv", index=False)
        # source-only LOLO: thresholds from healthy windows of the two source loads
        lolo = []
        for test_load in (0, 2, 4):
            Hs = W[(W.fault_group == "Healthy") & (W.load_Nm != test_load)]
            a, b = float(Hs.score_O.quantile(FAM_HEALTHY_Q)), float(Hs.score_I.quantile(FAM_HEALTHY_Q))
            T = W[(W.load_Nm == test_load) & W.in_bearing_scope]
            call = _call(T.score_O.values, T.score_I.values, a, b)
            ok = call == T.expected.values
            for lab in sorted(T.label7.unique()):
                m = T.label7.values == lab
                lolo.append({"test_load": test_load, "label7": lab, "n_windows": int(m.sum()),
                             "frac_expected_call": float(ok[m].mean()), "thr_O_src": a, "thr_I_src": b})
            oos = W[(W.load_Nm == test_load) & ~W.in_bearing_scope]
            c2 = _call(oos.score_O.values, oos.score_I.values, a, b)
            lolo.append({"test_load": test_load, "label7": "OUT_OF_SCOPE(fp)", "n_windows": len(oos),
                         "frac_expected_call": float(np.mean(c2 == "none")), "thr_O_src": a, "thr_I_src": b})
        lolo = pd.DataFrame(lolo)
        lolo.to_csv(tab / f"physics_family_lolo_source_thresholds_{band}.csv", index=False)
        # observed orders where the expected family was called
        ordr = []
        for g, sub in W[W.in_bearing_scope & (W.call == W.expected) & (W.fault_group != "Healthy")].groupby("label7"):
            col = "order_I" if g.startswith("IR") else "order_O"
            theo = orders()["BPFI" if g.startswith("IR") else "BPFO"]
            ordr.append({"label7": g, "n_windows": len(sub), "median_order": float(sub[col].median()),
                         "iqr_order": float(sub[col].quantile(0.75) - sub[col].quantile(0.25)),
                         "kinematic_order": theo, "rel_offset_pct": 100 * (float(sub[col].median()) / theo - 1)})
        ordr = pd.DataFrame(ordr)
        ordr.to_csv(tab / f"physics_family_observed_orders_{band}.csv", index=False)
        ins = per_rec[per_rec.in_bearing_scope]
        oos = W[~W.in_bearing_scope]
        res[band] = {
            "thr_O_healthy_q99": tO, "thr_I_healthy_q99": tI,
            "window_frac_expected_by_class": ins.groupby("label7").frac_expected.mean().round(3).to_dict(),
            "window_frac_expected_by_class_load": {f"{r.label7}@{r.load_Nm}": round(r.frac_expected, 3) for r in ins.itertuples()},
            "min_frac_expected_fault_recording": float(ins[ins.fault_group != "Healthy"].frac_expected.min()),
            "out_of_scope_window_false_call_rate": float(np.mean(oos.call != "none")),
            "out_of_scope_false_calls": {f"{a}|{b}": int(n) for (a, b), n in
                                         oos[oos.call != "none"].groupby("recording_id").call.value_counts().items()},
            "healthy_window_false_call_rate": float(np.mean(W[W.fault_group == "Healthy"].call != "none")),
            "observed_orders": ordr.to_dict("records"),
            "lolo_source_threshold": lolo.to_dict("records"),
        }
    return res


def temperature_shortcut_probe(trec: pd.DataFrame, tab: Path) -> list[dict]:
    """Parameter-free record-level 1-NN on temperature features, LOLO with source-only scaling.

    DIAGNOSTIC ONLY (n = 7 test recordings per fold): quantifies whether temperature
    can identify the condition, which - given class/session/installation confounding -
    would be a shortcut rather than fault physics. Not a model result.
    """
    t = trec[trec.in_bearing_scope]
    feats = {"A_minus_B": ["A_minus_B"], "TA_TB": ["TA_mean", "TB_mean"],
             "all": ["TA_mean", "TB_mean", "A_minus_B", "TA_slope_C_min", "TB_slope_C_min"]}
    rows = []
    for name, cols in feats.items():
        for L in (0, 2, 4):
            S, T = t[t.load_Nm != L], t[t.load_Nm == L]
            mu, sd = S[cols].mean(), S[cols].std().replace(0, 1)
            Xs, Xt = ((S[cols] - mu) / sd).values, ((T[cols] - mu) / sd).values
            pred = np.array([S.label7.values[np.argmin(((Xs - x) ** 2).sum(1))] for x in Xt])
            g = lambda a: np.array([p if p == "H" else p[:2] for p in a])  # noqa: E731
            rows.append({"features": name, "test_load": L, "n_test": len(T),
                         "acc7": float(np.mean(pred == T.label7.values)),
                         "acc3": float(np.mean(g(pred) == g(T.label7.values))),
                         "chance7": 1 / 7, "majority3": 3 / 7})
    df = pd.DataFrame(rows)
    df.to_csv(tab / "temperature_shortcut_probe_1nn.csv", index=False)
    return rows
