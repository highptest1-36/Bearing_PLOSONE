"""POST HOC, EXPLORATORY (pre-submission method audit, 2026-11-19). Not pre-specified.

Arithmetic on already-reported confusion matrices only: results/phase4/tables/external_primary.csv (frozen single
Ottawa evaluation, 20 H / 5 IR / 5 OR recordings) and results/phase3b/tables/per_bearing_predictions.csv (PU-BD5,
5 / 5 / 5 bearings). No model is run, no Ottawa signal or label file is read, nothing frozen is modified.

Purpose: separate the effect of class prevalence on macro-F1 from changes in per-class error rates. Each confusion
matrix is row-normalised to per-class rates and re-weighted to (a) 5/5/5 and (b) 20/5/5 class counts.
Balanced accuracy (mean recall) is prevalence-invariant and is reported alongside.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase5_audit"
L3 = ["H", "IR", "OR"]


def macro_f1(C):
    C = np.asarray(C, float)
    f = []
    for i in range(3):
        tp, p, r = C[i, i], C[:, i].sum(), C[i].sum()
        prec = tp / p if p else 0.0
        rec = tp / r if r else 0.0
        f.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f))


def reweight(C, counts):
    C = np.asarray(C, float)
    R = C / C.sum(1, keepdims=True)
    return R * np.asarray(counts, float)[:, None]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ext = pd.read_csv(ROOT / "results/phase4/tables/external_primary.csv")
    pbp = pd.read_csv(ROOT / "results/phase3b/tables/per_bearing_predictions.csv", index_col=0)
    rows = []
    for _, r in ext.iterrows():
        m = r["model"]
        Co = np.array(json.loads(r["confusion"]))
        row = {"model": m, "ottawa_counts": Co.sum(1).tolist(),
               "ottawa_macroF1_reported": round(float(r["macro_f1"]), 3),
               "ottawa_macroF1_recomputed": round(macro_f1(Co), 3),
               "ottawa_macroF1_at_5_5_5": round(macro_f1(reweight(Co, [5, 5, 5])), 3),
               "ottawa_bal_acc": round(float(np.mean(np.diag(Co) / Co.sum(1))), 3),
               "ottawa_FA": round(1 - Co[0, 0] / Co[0].sum(), 3)}
        if m in pbp.columns:
            Cp = np.zeros((3, 3), int)
            for t, p in zip(pbp["true"], pbp[m]):
                Cp[L3.index(t), L3.index(p)] += 1
            row.update({"pu_macroF1": round(macro_f1(Cp), 3),
                        "pu_macroF1_at_20_5_5": round(macro_f1(reweight(Cp, [20, 5, 5])), 3),
                        "pu_bal_acc": round(float(np.mean(np.diag(Cp) / Cp.sum(1))), 3),
                        "pu_FA": round(1 - Cp[0, 0] / Cp[0].sum(), 3), "pu_confusion": Cp.tolist()})
        rows.append(row)
    df = pd.DataFrame(rows)
    assert (abs(df.ottawa_macroF1_reported - df.ottawa_macroF1_recomputed) < 1e-3).all()
    df.to_csv(OUT / "prevalence_reweighting.csv", index=False)
    print(df.drop(columns=["pu_confusion"], errors="ignore").to_string())


if __name__ == "__main__":
    main()
