"""Aggregate Phase 2 runs into separate classification / shortcut-control / physics tables + gate evaluation."""
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

from pave.features.physics import FEATURE_NAMES  # noqa: E402

RUNS = ROOT / "results" / "phase2" / "runs"
OUT = ROOT / "results" / "phase2"
MODEL_ORDER = ["S1_speed_LR", "S2_rms_LR", "C1_hand_RF", "C2_hand_SVM", "P1_phys_LR", "D1_cnn_raw", "D2_cnn_order",
               "D3_cnn_order_rmsnorm"]
MODEL_TXT = {"S1_speed_LR": "S1 speed-only", "S2_rms_LR": "S2 RMS-only", "C1_hand_RF": "C1 hand+RF",
             "C2_hand_SVM": "C2 hand+SVM", "P1_phys_LR": "P1 physics+LR", "D1_cnn_raw": "D1 CNN raw",
             "D2_cnn_order": "D2 CNN order", "D3_cnn_order_rmsnorm": "D3 CNN order+RMSnorm"}


def load() -> tuple[pd.DataFrame, list[dict]]:
    rows, diags = [], []
    for f in sorted(RUNS.glob("*/*/runs.jsonl")):
        if "_smoke" in f.parts:
            continue
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if r["run_key"].endswith("physics_diag"):
                r["dataset"] = f.parts[-3]
                diags.append(r)
                continue
            gl = {k: v for k, v in r["test"].items() if k != "window"}
            row = {"dataset": r["dataset"], "protocol": r["protocol"], "fold": r["fold"], "task": r["task"],
                   "model": r["model"], "seed": r["seed"], "win_macro_f1": r["test"]["window"]["macro_f1"],
                   "win_acc": r["test"]["window"]["acc"], "win_bal_acc": r["test"]["window"]["bal_acc"],
                   "win_ece": r["test"]["window"]["ece"], "n_test_windows": r["test"]["window"]["n"],
                   "recall": r["test"]["window"]["recall_per_class"],
                   "val_win_macro_f1": r["val"]["window"]["macro_f1"]}
            for g, v in gl.items():
                row[f"{g}_acc"] = v["acc"]
                row[f"{g}_macro_f1"] = v["macro_f1"]
                row[f"{g}_n"] = v["n"]
            if "challenge" in r:
                row["challenge"] = r["challenge"]
            rows.append(row)
    return pd.DataFrame(rows), diags


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    gcols = [c for c in df.columns if c.endswith("_acc") and not c.startswith("win")] + \
            [c for c in df.columns if c.endswith("_macro_f1") and not c.startswith("win") and not c.startswith("val")]
    agg = {"win_macro_f1": ["mean", "std", "min"], "win_bal_acc": ["mean"], "win_ece": ["mean"], "seed": ["nunique"],
           "fold": ["nunique"]}
    for c in gcols:
        agg[c] = ["mean", "min"]
    t = df.groupby(["dataset", "protocol", "task", "model"]).agg(agg)
    t.columns = ["_".join(c) for c in t.columns]
    t = t.reset_index()
    t["model"] = pd.Categorical(t.model, MODEL_ORDER, ordered=True)
    return t.sort_values(["dataset", "protocol", "task", "model"]).round(3)


def per_fold(df: pd.DataFrame) -> pd.DataFrame:
    gl = [c for c in df.columns if c.endswith("_acc") and not c.startswith("win")]
    t = df.groupby(["dataset", "protocol", "task", "model", "fold"])[["win_macro_f1"] + gl].mean().reset_index()
    return t.round(3)


def physics_table(diags: list[dict]) -> pd.DataFrame:
    rows = []
    for d in diags:
        r = {"dataset": d["dataset"], "protocol": d["protocol"], "fold": d["fold"]}
        for fam in ("BPFO", "BPFI"):
            r[f"{fam}_centre_pct"] = 100 * d["regions"]["centre"][fam]
            r[f"{fam}_half_pct"] = 100 * d["regions"]["half"][fam]
            r[f"{fam}_n_cal"] = d["regions"]["n_used"][fam]
            r[f"{fam}_fallback"] = d["regions"]["fallback"][fam]
        for split in ("train", "test"):
            for k, v in d[f"auroc_{split}"].items():
                r[f"{split}_{k}"] = v
        if d.get("bearings"):
            r["test_bearings"] = ",".join(d["bearings"]["test"])
        rows.append(r)
    return pd.DataFrame(rows).round(4)


def physics_feature_medians(ds: str, protocol: str) -> pd.DataFrame | None:
    """Median physics feature by class on TEST windows (feature values use the fold's train calibration)."""
    meta = pd.read_csv(OUT / "windows" / ds / "meta.csv")
    pdir = RUNS / ds / protocol / "preds"
    out = []
    for f in sorted(pdir.glob("*physics_diag.npz")):
        fold = f.name.split("__")[0]
        z = np.load(f)
        runs = [json.loads(l) for l in (RUNS / ds / protocol / "runs.jsonl").read_text().splitlines()]
        d = next(r for r in runs if r["run_key"] == f"{fold}|physics_diag")
        tests = set(d["bearings"]["test"]) if d.get("bearings") else None
        if tests is None:
            continue
        m = meta.bearing.isin(tests).to_numpy()
        P = pd.DataFrame(z["phys"][m], columns=FEATURE_NAMES)
        P["label3"] = meta.label3[m].to_numpy()
        P["fold"] = fold
        out.append(P)
    if not out:
        return None
    A = pd.concat(out)
    return A.groupby("label3")[FEATURE_NAMES].median().round(2)


def gates(t: pd.DataFrame, phys: pd.DataFrame) -> dict:
    g = {}
    pu = phys[(phys.dataset == "pu") & (phys.protocol == "PU_BD5")]
    if len(pu):
        c = float(pu["test_auroc_contrast_OR_vs_IR"].mean())
        h = float(pu["test_auroc_family_damaged_vs_healthy"].mean())
        g["V3"] = {"mean_auroc_contrast_OR_vs_IR": c, "mean_auroc_family_damaged_vs_healthy": h,
                   "per_fold_contrast": pu["test_auroc_contrast_OR_vs_IR"].round(3).tolist(),
                   "per_fold_family": pu["test_auroc_family_damaged_vs_healthy"].round(3).tolist(),
                   "threshold": 0.75, "pass": bool(c >= 0.75 and h >= 0.75)}
    s1 = t[(t.dataset == "pu") & (t.protocol == "PU_BD5") & (t.model == "S1_speed_LR")]
    if len(s1):
        v = float(s1.win_macro_f1_mean.iloc[0])
        g["V4"] = {"S1_speed_only_window_macro_f1": v, "threshold_max": 0.45, "pass": bool(v <= 0.45)}
    return g


def fig_bars(t: pd.DataFrame, df: pd.DataFrame) -> None:
    panels = [("kaist", "A_LOLO", "3way"), ("kaist", "A_LOLO", "7way"), ("pu", "PU_BD5", "3way"), ("pu", "PU_A2R", "3way")]
    panels = [p for p in panels if ((df.dataset == p[0]) & (df.protocol == p[1]) & (df.task == p[2])).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (ds, pr, task) in zip(axes, panels):
        sub = df[(df.dataset == ds) & (df.protocol == pr) & (df.task == task)]
        models = [m for m in MODEL_ORDER if m in set(sub.model)]
        for i, m in enumerate(models):
            v = sub[sub.model == m].groupby("fold").win_macro_f1.mean()
            col = "#b5b4ad" if m.startswith("S") else ("#1baf7a" if m.startswith("P") else ("#eb6834" if m.startswith("C") else "#2a78d6"))
            ax.bar(i, v.mean(), color=col, width=0.7)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, s=8, color="#0b0b0b", zorder=3)
        ax.set_xticks(range(len(models)), [MODEL_TXT[m] for m in models], rotation=60, ha="right", fontsize=6.5)
        ax.set_title(f"{ds.upper()} {pr} {task}", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.axhline(1 / (3 if task == "3way" else 7), color="#52514e", lw=0.6, ls=":")
    axes[0].set_ylabel("test window macro-F1
(bar = mean; dots = folds)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_p2_classification.png", dpi=300)
    plt.close(fig)


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    df, diags = load()
    t = summarise(df)
    t.to_csv(OUT / "tables" / "T_classification_summary.csv", index=False)
    per_fold(df).to_csv(OUT / "tables" / "T_classification_per_fold.csv", index=False)
    sc = t[t.model.isin(["S1_speed_LR", "S2_rms_LR", "D1_cnn_raw", "D2_cnn_order", "D3_cnn_order_rmsnorm"])]
    sc.to_csv(OUT / "tables" / "T_shortcut_controls.csv", index=False)
    phys = physics_table(diags)
    phys.to_csv(OUT / "tables" / "T_physics_features.csv", index=False)
    meds = {}
    for ds, pr in (("pu", "PU_BD5"), ("pu", "PU_A2R")):
        if (RUNS / ds / pr).exists():
            m = physics_feature_medians(ds, pr)
            if m is not None:
                m.to_csv(OUT / "tables" / f"T_physics_feature_medians_{ds}_{pr}.csv")
                meds[f"{ds}_{pr}"] = m.to_dict()
    ch = df[df.get("challenge").notna()] if "challenge" in df else pd.DataFrame()
    if len(ch):
        crow = []
        for r in ch.itertuples():
            for b, d in r.challenge.items():
                crow.append({"protocol": r.protocol, "fold": r.fold, "model": r.model, "seed": r.seed, "bearing": b, **d})
        pd.DataFrame(crow).groupby(["protocol", "model", "bearing"])[["H", "IR", "OR"]].mean().round(3).to_csv(
            OUT / "tables" / "T_challenge_KB.csv")
    g = gates(t, phys)
    (OUT / "gates.json").write_text(json.dumps(g, indent=2))
    fig_bars(t, df)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t[["dataset", "protocol", "task", "model", "fold_nunique", "seed_nunique", "win_macro_f1_mean", "win_macro_f1_std",
             "win_macro_f1_min"] + [c for c in t.columns if c.endswith("_acc_mean")]].to_string())
    print(phys.to_string())
    print(json.dumps(g, indent=1))


if __name__ == "__main__":
    main()
