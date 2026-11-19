"""Phase 2 bounded baseline experiments (protocol V2 §4-§9).

Every fold is defined from frozen manifests at recording/bearing level; windows are assigned afterwards.
Physics regions are calibrated on TRAIN windows only. Nothing from validation/test enters any fitted
statistic (scalers, z-score, calibration). Validation is used only for deep-model epoch selection.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pave.data.paderborn import BD5_ROWS
from pave.evaluation.metrics import evaluate
from pave.features import physics as PH
from pave.signal.physics import BearingGeometry, orders

L3 = ["H", "IR", "OR"]
L7 = ["H", "IR_0p3", "IR_1p0", "IR_3p0", "OR_0p3", "OR_1p0", "OR_3p0"]


@dataclass
class DS:
    name: str
    meta: pd.DataFrame
    ometa: pd.DataFrame
    raw: np.ndarray
    order: np.ndarray
    hand: np.ndarray
    short: np.ndarray
    env: np.ndarray
    info: dict


def load_ds(root: Path, name: str) -> DS:
    d = root / name
    return DS(name, pd.read_csv(d / "meta.csv"), pd.read_csv(d / "order_meta.csv"),
              np.load(d / "raw.npy", mmap_mode="r"), np.load(d / "order.npy", mmap_mode="r"),
              np.load(d / "hand.npy"), np.load(d / "short.npy"), np.load(d / "env.npy"),
              json.loads((d / "info.json").read_text()))


# ----------------------------------------------------------------------------- folds
def kaist_folds(ds: DS, split_dir: Path) -> list[dict]:
    out = []
    m = ds.meta
    for f in sorted(split_dir.glob("split_A_test*.json")):
        sp = json.loads(f.read_text())
        role = np.full(len(m), "", dtype=object)
        for a in sp["assignments"]:
            sel = (m.recording_id == a["recording_id"]) & (m.t_start >= a["start_s"] - 1e-9) & (m.t_end <= a["end_s"] + 1e-9)
            if (role[sel.to_numpy()] != "").any():
                raise AssertionError("window assigned twice")
            role[sel.to_numpy()] = a["role"]
        idx = {r: np.flatnonzero(role == r) for r in ("train", "val", "test")}
        out.append({"protocol": "A_LOLO", "fold": sp["fold_id"], "split_sha256": sp["content_sha256"],
                    "raw": idx, "order": idx, "challenge": {"raw": np.array([], int), "order": np.array([], int)}})
    return out


def pu_bd5_sets() -> list[dict]:
    rows = [set(r) for r in BD5_ROWS]
    out = []
    for k in range(5):
        v = (k + 1) % 5
        tr = set().union(*[rows[i] for i in range(5) if i not in (k, v)])
        out.append({"fold": f"BD5_f{k+1}", "train": tr, "val": rows[v], "test": rows[k]})
    return out


PU_A2R = {"fold": "A2R", "train": {"KA01", "KA05", "KA07", "KA08", "KA09", "KI01", "KI05", "KI07", "K002", "K003", "K004", "K005"},
          "val": {"KA03", "KA06", "KI03", "KI08", "K006"},
          "test": {"K001", "KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"}}
PU_CHALLENGE = {"KB23", "KB24", "KB27"}


def check_bearing_disjoint(s: dict) -> None:
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"]), s["fold"]


def pu_folds(ds: DS, protocol: str) -> list[dict]:
    sets = pu_bd5_sets() if protocol == "PU_BD5" else [PU_A2R]
    out = []
    for s in sets:
        check_bearing_disjoint(s)
        f = {"protocol": protocol, "fold": s["fold"], "bearings": {k: sorted(s[k]) for k in ("train", "val", "test")}}
        for key, mm in (("raw", ds.meta), ("order", ds.ometa)):
            f[key] = {r: np.flatnonzero(mm.bearing.isin(s[r]).to_numpy()) for r in ("train", "val", "test")}
        f["challenge"] = {"raw": np.flatnonzero(ds.meta.bearing.isin(PU_CHALLENGE).to_numpy()),
                          "order": np.flatnonzero(ds.ometa.bearing.isin(PU_CHALLENGE).to_numpy())}
        out.append(f)
    return out


# ----------------------------------------------------------------------------- physics
def kin_orders_per_window(ds: DS, bearings_csv: Path | None) -> np.ndarray:
    """(N, 2) kinematic orders [BPFO, BPFI] for each raw window."""
    if ds.name == "kaist":
        o = orders()
        return np.tile([o["BPFO"], o["BPFI"]], (len(ds.meta), 1))
    b = pd.read_csv(bearings_csv).set_index("bearing")
    lut = {}
    for code, r in b.iterrows():
        g = BearingGeometry(code, int(r.geom_n_balls), float(r.geom_ball_diameter_mm), float(r.geom_pitch_diameter_mm),
                            float(r.geom_contact_angle_deg))
        o = orders(g)
        lut[code] = (o["BPFO"], o["BPFI"])
    return np.array([lut[c] for c in ds.meta.bearing])


def physics_scan(ds: DS, kin: np.ndarray) -> pd.DataFrame:
    """Label-free scan per window and channel; keeps the channel with the larger scan SNR per family."""
    f = np.arange(ds.env.shape[2], dtype=float)
    rows = []
    for i in range(len(ds.meta)):
        fr = ds.meta.fr_hz.iat[i]
        ko = {"BPFO": kin[i, 0], "BPFI": kin[i, 1]}
        best = {}
        for c in range(ds.env.shape[1]):
            s = PH.window_line_scan(f, ds.env[i, c].astype(float), fr, ko)
            for fam in PH.FAMILIES:
                if fam not in best or (np.nan_to_num(s[f"{fam}_scan_snr"], nan=-99) > np.nan_to_num(best[fam][0], nan=-99)):
                    best[fam] = (s[f"{fam}_scan_snr"], s[f"{fam}_scan_r"])
        rows.append({f"{fam}_scan_snr": best[fam][0] for fam in PH.FAMILIES} | {f"{fam}_scan_r": best[fam][1] for fam in PH.FAMILIES})
    return pd.DataFrame(rows)


def physics_features(ds: DS, kin: np.ndarray, reg: PH.Regions) -> np.ndarray:
    f = np.arange(ds.env.shape[2], dtype=float)
    out = np.empty((len(ds.meta), len(PH.FEATURE_NAMES)))
    for i in range(len(ds.meta)):
        fr = ds.meta.fr_hz.iat[i]
        ko = {"BPFO": kin[i, 0], "BPFI": kin[i, 1]}
        per = [PH.window_features(f, ds.env[i, c].astype(float), fr, ko, reg) for c in range(ds.env.shape[1])]
        d = {}
        for k in PH.FEATURE_NAMES:
            if k != "contrast":
                d[k] = np.nanmax([p[k] for p in per]) if any(np.isfinite(p[k]) for p in per) else np.nan
        d["contrast"] = d["BPFO_harm"] - d["BPFI_harm"]
        out[i] = [d[k] for k in PH.FEATURE_NAMES]
    return out


def physics_auroc(feat: np.ndarray, y3: np.ndarray) -> dict:
    """V3 diagnostics on a set of windows: contrast OR-vs-IR, max family evidence damaged-vs-healthy."""
    c = feat[:, PH.FEATURE_NAMES.index("contrast")]
    hmax = np.nanmax(feat[:, [PH.FEATURE_NAMES.index("BPFO_harm"), PH.FEATURE_NAMES.index("BPFI_harm")]], axis=1)
    out = {}
    m = (y3 == 1) | (y3 == 2)
    if len(np.unique(y3[m])) == 2:
        ok = m & np.isfinite(c)
        out["auroc_contrast_OR_vs_IR"] = float(roc_auc_score(y3[ok] == 2, c[ok]))
    if (y3 == 0).any() and m.any():
        ok = np.isfinite(hmax)
        out["auroc_family_damaged_vs_healthy"] = float(roc_auc_score(y3[ok] != 0, hmax[ok]))
    return out


# ----------------------------------------------------------------------------- classical models
def classical(kind: str, seed: int = 42):
    if kind == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if kind == "RF":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=8, random_state=seed)
    if kind == "SVM":
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=seed))
    raise ValueError(kind)


def impute_train_median(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    X = X.copy()
    med = np.nanmedian(X[tr], axis=0)
    r, c = np.where(~np.isfinite(X))
    X[r, c] = med[c]
    return X


def full_probs(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    p = model.predict_proba(X)
    cls = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    out = np.zeros((len(X), n_classes))
    out[:, cls] = p
    return out
