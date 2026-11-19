"""POST HOC, EXPLORATORY (pre-submission method audit, 2026-11-19). Not pre-specified in any protocol.

Question: the frozen inclusion rule (protocol V2.2/V2.3) was applied ONCE to the 60 pooled inner-OOF bearing predictions of
all five outer folds. Every PU-BD5 bearing appears in the inner-validation sets of the four outer folds in which it is a
training bearing, so the labels of the outer-test bearings of fold k informed the global decision through folds j != k.

This script re-applies the SAME rule separately inside each outer fold, using only that fold's 12 inner-OOF bearing
predictions, and reports which model each fold would have selected and the outer-test macro-F1 of that fold-locally
selected procedure. It reads only stored Phase 3B outputs (results/phase3b/folds/O*.npz, per_bearing_predictions.csv):
no retraining, no new predictions, no Ottawa data. Outputs go to results/phase5_audit/ (new directory).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from pave.experiments import phase3a as P3A  # noqa: E402

P3B = ROOT / "results" / "phase3b"
OUT = ROOT / "results" / "phase5_audit"
L3 = ["H", "IR", "OR"]


def macro(y, p):
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "results/phase2/windows/pu/meta.csv")
    frozen = json.loads((P3B / "tables" / "inclusion_decision.json").read_text())
    rows, choice = [], {}
    pooled = {m: ([], [], []) for m in ("A2", "A4", "B1-V3-1s", "B1-V3-4s")}
    for k in range(1, 6):
        js = json.loads((P3B / "folds" / f"O{k}.json").read_text())
        z = np.load(P3B / "folds" / f"O{k}.npz")
        y = z["oof_y"]
        b = meta.bearing.to_numpy()[z["oof__idx"]]
        res = {}
        for m in pooled:
            P = z[f"oof_{m}"]
            bt = P3A.bearing_table(P, y, b)
            res[m] = {"f1": macro(bt.y, bt.pred), "nll": nll(P, y), "n_bearings": int(len(bt))}
            pooled[m][0].append(P)
            pooled[m][1].append(y)
            pooled[m][2].append(np.array([f"{js['fold']}|{x}" for x in b]))
        keep = res["A4"]["f1"] - res["A2"]["f1"] >= 0.05 and res["A4"]["nll"] <= res["A2"]["nll"]
        phys = "B1-V3-4s" if res["B1-V3-4s"]["f1"] >= res["B1-V3-1s"]["f1"] else "B1-V3-1s"
        sel = "A4" if keep else phys
        choice[js["fold"]] = sel
        rows.append({"outer_fold": js["fold"], "test_bearings": " ".join(sorted(P3A.outer_folds()[k - 1]["test"])), "tie_4s_vs_1s": res["B1-V3-4s"]["f1"] == res["B1-V3-1s"]["f1"],
                     **{f"{m}_innerF1": round(v["f1"], 4) for m, v in res.items()},
                     **{f"{m}_innerNLL": round(v["nll"], 4) for m, v in res.items()},
                     "keep_latent": keep, "physics_choice": phys, "selected": sel})
    # sanity: pooled recomputation must reproduce the frozen decision numbers
    for m, (Ps, Ys, Bs) in pooled.items():
        P, y, bb = np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Bs)
        bt = P3A.bearing_table(P, y, bb)
        f = macro(bt.y, bt.pred)
        assert abs(f - frozen["inner_oof"][m]["inner_oof_bearing_macro_f1"]) < 1e-9, (m, f)
        assert abs(nll(P, y) - frozen["inner_oof"][m]["inner_oof_window_nll"]) < 1e-6, m
    # outer-test result of the fold-locally selected procedure
    pbp = pd.read_csv(P3B / "tables" / "per_bearing_predictions.csv", index_col=0)
    fold_of = {}
    for o in P3A.outer_folds():
        for bname in o["test"]:
            fold_of[bname] = o["fold"]
    y_true = [L3.index(pbp.loc[bname, "true"]) for bname in pbp.index]
    y_sel = [L3.index(pbp.loc[bname, choice[fold_of[bname]]]) for bname in pbp.index]
    y_glob = [L3.index(pbp.loc[bname, frozen["final_recommended_model"]]) for bname in pbp.index]
    summary = {"status": "POST HOC EXPLORATORY - not pre-specified; does not replace the frozen decision",
               "frozen_global_choice": frozen["final_recommended_model"],
               "fold_local_choices": choice,
               "all_folds_agree_with_global": all(v == frozen["final_recommended_model"] for v in choice.values()),
               "outer_macro_f1_global_choice": macro(y_true, y_glob),
               "outer_macro_f1_fold_local_procedure": macro(y_true, y_sel),
               "n_bearings": len(y_true)}
    pd.DataFrame(rows).to_csv(OUT / "foldlocal_selection.csv", index=False)
    (OUT / "foldlocal_selection_summary.json").write_text(json.dumps(summary, indent=1))
    print(pd.DataFrame(rows).to_string())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
