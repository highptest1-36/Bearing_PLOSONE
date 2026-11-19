"""Aggregate Phase 3A nested baselines (bearing = unit). Writes results/phase3a/tables + figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments import phase3a as P3  # noqa: E402

BASE = ROOT / "results" / "phase3a" / "baselines"
OUT = ROOT / "results" / "phase3a"
MODELS = ["B0", "B1", "B2", "B3", "B1_default_C1", "B3_default", "B2_s42", "B2_s43", "B2_s44", "B3_s42", "B3_s43", "B3_s44",
          "B3_default_s42", "B3_default_s43", "B3_default_s44"]
TXT = {"B0": "B0 physics rule", "B1": "B1 = P1 (LR, nested C)", "B2": "B2 concept MLP (nested)", "B3": "B3 CNN raw (nested)",
       "B1_default_C1": "B1 default C=1", "B3_default": "B3 default (lr 1e-3, 25 ep)"}


def macro(bt: pd.DataFrame) -> float:
    return float(f1_score(bt.y, bt.pred, labels=[0, 1, 2], average="macro", zero_division=0))


def main() -> None:
    meta = pd.read_csv(ROOT / "results/phase2/windows/pu/meta.csv")
    folds = [json.loads((BASE / f"fold_O{k}.json").read_text()) for k in range(1, 6)]
    assert not any(f.get("smoke") for f in folds), "smoke results present"
    Z = {f["fold"]: np.load(BASE / f"fold_{f['fold']}.npz") for f in folds}
    te = np.concatenate([Z[f["fold"]]["te"] for f in folds])
    y = np.concatenate([Z[f["fold"]]["y"] for f in folds])
    b = meta.bearing.to_numpy()[te]
    fold_of = np.concatenate([[f["fold"]] * len(Z[f["fold"]]["te"]) for f in folds])
    P = {m: np.concatenate([Z[f["fold"]][m] for f in folds]) for m in MODELS + ["B1_TS", "B2_TS", "B3_TS"]}
    rows, bts = [], {}
    for m in ["B1"] + [x for x in MODELS if x != "B1"]:  # B1 first: it is the comparator
        bt = P3.bearing_table(P[m], y, b)
        bts[m] = bt
        bm = P3.bearing_metrics(bt)
        wm = P3.window_metrics(P[m], y)
        per_fold = [P3.window_metrics(P[m][fold_of == f["fold"]], y[fold_of == f["fold"]])["macro_f1"] for f in folds]
        lo, hi = P3.stratified_bootstrap(bt, macro, n=10000, seed=1)
        r = {"model": m, "bearing_macro_f1": bm["macro_f1"], "bearing_macro_f1_ci95": [lo, hi], "bearing_acc": bm["acc"],
             "bearing_bal_acc": bm["bal_acc"], "bearing_recall_H_IR_OR": bm["recall"], "bearing_confusion": bm["confusion"],
             "bearing_failures": bm["failures"], "window_macro_f1_pooled": wm["macro_f1"],
             "window_macro_f1_fold_mean": float(np.mean(per_fold)), "window_macro_f1_fold_sd": float(np.std(per_fold, ddof=1)),
             "window_bal_acc": wm["bal_acc"], "ece": wm["ece"] if not m.startswith("B0") else None,
             "nll": wm["nll"] if not m.startswith("B0") else None,
             "healthy_specificity": float(np.mean(bt[bt.y == 0].pred == 0)),
             "faulty_sensitivity": float(np.mean(bt[bt.y != 0].pred != 0))}
        if m in ("B1", "B2", "B3"):
            wts = P3.window_metrics(P[f"{m}_TS"], y)
            r["ece_TS"], r["nll_TS"] = wts["ece"], wts["nll"]
        if m != "B1":
            a_ok = bts["B1"].set_index("bearing").correct
            o_ok = bt.set_index("bearing").correct.reindex(a_ok.index)
            r["vs_B1_mcnemar"] = P3.mcnemar_exact(o_ok.to_numpy(), a_ok.to_numpy())
            both = bts["B1"].merge(bt, on=["bearing", "y"], suffixes=("_a", "_b"))
            rng = np.random.default_rng(7)
            groups = [both[both.y == k].index.to_numpy() for k in range(3)]
            diffs = []
            for _ in range(10000):
                idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
                s = both.loc[idx]
                diffs.append(f1_score(s.y, s.pred_b, labels=[0, 1, 2], average="macro", zero_division=0)
                             - f1_score(s.y, s.pred_a, labels=[0, 1, 2], average="macro", zero_division=0))
            r["diff_vs_B1_bearing_macro_f1"] = r["bearing_macro_f1"] - macro(bts["B1"])
            r["diff_vs_B1_ci95"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
        rows.append(r)
    # metadata-only rules (S-4a)
    for key in ("manufacturer", "year"):
        pred = {bb: P3.LIDX[lab] for f in folds for bb, lab in f["metadata_rules"][key].items()}
        bt = pd.DataFrame({"bearing": list(pred), "y": [P3.LIDX[BEARINGS[bb]["label3"]] for bb in pred], "pred": list(pred.values())})
        bt["correct"] = bt.y == bt.pred
        bm = P3.bearing_metrics(bt)
        rows.append({"model": f"META_{key}_only", "bearing_macro_f1": bm["macro_f1"], "bearing_acc": bm["acc"],
                     "bearing_bal_acc": bm["bal_acc"], "bearing_recall_H_IR_OR": bm["recall"], "bearing_confusion": bm["confusion"],
                     "bearing_failures": bm["failures"], "healthy_specificity": float(np.mean(bt[bt.y == 0].pred == 0)),
                     "faulty_sensitivity": float(np.mean(bt[bt.y != 0].pred != 0))})
        bts[f"META_{key}_only"] = bt
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables" / "baselines_summary.json").write_text(json.dumps(rows, indent=1, default=float))
    flat = pd.DataFrame([{k: (v if not isinstance(v, (list, dict)) else json.dumps(v)) for k, v in r.items()} for r in rows])
    flat.to_csv(OUT / "tables" / "baselines_summary.csv", index=False)
    # per-bearing outcome matrix
    pbm = pd.DataFrame({m: bts[m].set_index("bearing").pred.map(dict(enumerate(P3.L3))) for m in bts})
    pbm.insert(0, "true", [BEARINGS[x]["label3"] for x in pbm.index])
    pbm.insert(1, "damage", [BEARINGS[x]["method"].replace("accelerated lifetime; ", "") + f" L{BEARINGS[x]['extent']}" for x in pbm.index])
    pbm.to_csv(OUT / "tables" / "baselines_per_bearing_predictions.csv")
    # per-bearing probability table for main models
    pp = []
    for m in ("B0", "B1", "B2", "B3"):
        t = bts[m].copy()
        t["model"] = m
        pp.append(t)
    pd.concat(pp).round(3).to_csv(OUT / "tables" / "baselines_per_bearing_probs.csv", index=False)
    # selection / inner-loop diagnostics
    sel = []
    for f in folds:
        b1 = {k: (np.mean(v), np.max(v) - np.min(v)) for k, v in f["inner"]["B1"].items()}
        b3 = f["inner"]["B3_curve_per_inner"]
        e3 = f["selected"]["B3"]["config"][1]
        lr3 = f["selected"]["B3"]["config"][0]
        sp3 = b3.get(str((lr3, e3)))
        sel.append({"fold": f["fold"], "B1_C": f["selected"]["B1"]["config"][0], "B1_inner_mean": f["selected"]["B1"]["inner_mean_macro_f1"],
                    "B1_inner_spread_selected": b1[str((f["selected"]["B1"]["config"][0],))][1],
                    "B1_inner_spread_min_over_configs": min(v[1] for v in b1.values()),
                    "B2_config": f["selected"]["B2"]["config"], "B2_inner_mean": f["selected"]["B2"]["inner_mean_macro_f1"],
                    "B3_lr": lr3, "B3_epochs": e3, "B3_inner_mean": f["selected"]["B3"]["inner_mean_macro_f1"],
                    "B3_inner_spread_selected": (max(sp3) - min(sp3)) if sp3 else None,
                    "T_B1": f["temperature"]["B1"], "T_B2": f["temperature"]["B2"], "T_B3": f["temperature"]["B3"],
                    "B0_tau_dB": f["b0_tau"], "BPFO_centre_pct": 100 * f["regions"]["centre"]["BPFO"],
                    "BPFI_centre_pct": 100 * f["regions"]["centre"]["BPFI"], "seconds": f["seconds"]})
    S = pd.DataFrame(sel)
    S.round(4).to_csv(OUT / "tables" / "nested_selection.csv", index=False)
    # figure: confusion matrices (bearing level) + per-bearing correctness
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.9))
    for ax, m in zip(axes, ["B0", "B1", "B2", "B3", "META_year_only"]):
        cmx = np.array(next(r for r in rows if r["model"] == m)["bearing_confusion"])
        ax.imshow(cmx, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7f7f5", "#256abf"]), vmin=0, vmax=5)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cmx[i, j], ha="center", va="center", fontsize=7, color="white" if cmx[i, j] >= 4 else "#0b0b0b")
        ax.set_xticks(range(3), P3.L3, fontsize=6)
        ax.set_yticks(range(3), P3.L3, fontsize=6)
        ax.set_title(m.replace("META_", "").replace("_only", "-only"), fontsize=7)
        ax.grid(False)
    axes[0].set_ylabel("true", fontsize=6)
    fig.suptitle("Bearing-level confusion (15 PU-BD5 bearings, nested outer test); columns = predicted", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_bearing_confusion.png", dpi=300)
    plt.close(fig)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = ["model", "bearing_macro_f1", "bearing_macro_f1_ci95", "bearing_acc", "bearing_recall_H_IR_OR", "window_macro_f1_pooled",
            "window_macro_f1_fold_mean", "window_macro_f1_fold_sd", "ece", "ece_TS", "nll", "nll_TS", "healthy_specificity",
            "faulty_sensitivity", "diff_vs_B1_bearing_macro_f1", "diff_vs_B1_ci95", "vs_B1_mcnemar", "bearing_failures"]
    print(flat[[c for c in show if c in flat]].round(3).to_string())
    print(S.round(3).to_string())
    print(pbm.to_string())


if __name__ == "__main__":
    main()
