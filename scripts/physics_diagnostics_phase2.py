"""Real-data physics-feature diagnostics for Phase 2 (part E) + metadata-confound controls (post-hoc).

Uses the per-fold physics features saved by run_phase2.py (computed with TRAIN-only calibrated regions).
"""
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments.phase2 import PU_A2R, kin_orders_per_window, load_ds, pu_bd5_sets  # noqa: E402
from pave.features.physics import FEATURE_NAMES  # noqa: E402

OUT = ROOT / "results" / "phase2"
RUNS = OUT / "runs"
COL = {"H": "#2a78d6", "IR": "#eb6834", "OR": "#1baf7a"}


def fold_features(ds: str, protocol: str) -> list[tuple[str, dict, np.ndarray]]:
    runs = [json.loads(l) for l in (RUNS / ds / protocol / "runs.jsonl").read_text().splitlines()]
    out = []
    for r in runs:
        if r["run_key"].endswith("physics_diag"):
            z = np.load(RUNS / ds / protocol / "preds" / f"{r['run_key'].replace('|', '__')}.npz")
            out.append((r["fold"], r, z["phys"]))
    return out


def auc(y, s):
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if len(np.unique(y[ok])) == 2 else np.nan


def main() -> None:
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    pu = load_ds(OUT / "windows", "pu")
    meta = pu.meta
    ic, io_, ii = FEATURE_NAMES.index("contrast"), FEATURE_NAMES.index("BPFO_harm"), FEATURE_NAMES.index("BPFI_harm")
    res = {}
    # ---- PU-BD5: test windows of every fold (each bearing tested once)
    rows, wins = [], []
    for fold, d, F in fold_features("pu", "PU_BD5"):
        test = set(d["bearings"]["test"])
        m = meta.bearing.isin(test).to_numpy()
        T = meta[m].copy()
        T["contrast"] = F[m, ic]
        T["family_max"] = np.nanmax(F[m][:, [io_, ii]], axis=1)
        T["nharm_O"] = F[m, FEATURE_NAMES.index("BPFO_nharm")]
        T["nharm_I"] = F[m, FEATURE_NAMES.index("BPFI_nharm")]
        T["sb_I"] = F[m, FEATURE_NAMES.index("BPFI_sb")]
        T["fold"] = fold
        wins.append(T)
    W = pd.concat(wins)
    W.to_csv(OUT / "tables" / "T_physics_pu_bd5_test_windows.csv", index=False)
    per_b = W.groupby(["bearing", "label3", "fold"]).agg(contrast_med=("contrast", "median"), contrast_q25=("contrast", lambda s: s.quantile(.25)),
                                                         contrast_q75=("contrast", lambda s: s.quantile(.75)),
                                                         family_max_med=("family_max", "median"),
                                                         nharm_O_med=("nharm_O", "median"), nharm_I_med=("nharm_I", "median"),
                                                         sb_I_med=("sb_I", "median"), n=("contrast", "size")).reset_index()
    per_b["manufacturer"] = [pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing").fs_manufacturer[b] for b in per_b.bearing]
    per_b["damage"] = [BEARINGS[b]["method"] + f" | level {BEARINGS[b]['extent']} | {BEARINGS[b]['combination']}" for b in per_b.bearing]
    per_b.round(2).to_csv(OUT / "tables" / "T_physics_pu_bd5_per_bearing.csv", index=False)
    # per setting AUROC (pooled over folds; each window scored with its own fold's calibration)
    st = []
    for s, g in W.groupby("setting"):
        dmg = g[g.label3 != "H"]
        st.append({"setting": s, "n": len(g), "auroc_contrast_OR_vs_IR": auc((dmg.label3 == "OR").to_numpy(), dmg.contrast.to_numpy()),
                   "auroc_family_damaged_vs_healthy": auc((g.label3 != "H").to_numpy(), g.family_max.to_numpy())})
    pd.DataFrame(st).round(3).to_csv(OUT / "tables" / "T_physics_pu_bd5_by_setting.csv", index=False)
    res["bd5_by_setting"] = st
    # per-bearing "expected-sign" fraction (contrast >0 for OR, <0 for IR)
    W["sign_ok"] = np.where(W.label3 == "OR", W.contrast > 0, np.where(W.label3 == "IR", W.contrast < 0, np.nan))
    res["bd5_sign_ok_by_bearing"] = W[W.label3 != "H"].groupby("bearing").sign_ok.mean().round(3).to_dict()
    # ---- A2R: calibration on artificial, evaluated on real
    a2r = fold_features("pu", "PU_A2R")
    if a2r:
        fold, d, F = a2r[0]
        res["a2r_regions"] = d["regions"]
        res["a2r_auroc_test_real"] = d["auroc_test"]
        res["a2r_auroc_train_artificial"] = d["auroc_train"]
    # ---- KAIST
    kd = fold_features("kaist", "A_LOLO")
    res["kaist_regions"] = {f: d["regions"] for f, d, _ in kd}
    res["kaist_auroc_test"] = {f: d["auroc_test"] for f, d, _ in kd}
    # ---- metadata-confound controls (POST-HOC; analytic rules, no training)
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    dates = pd.read_csv(OUT / "pu_audit" / "measurement_dates_per_bearing.csv", index_col=0)
    ctrl = []
    for s in pu_bd5_sets():
        for b in sorted(s["test"]):
            man = bear.fs_manufacturer[b]
            yr = str(dates.loc[b, "date"])[:4]
            train = sorted(s["train"])
            # rule learnt from TRAIN bearings only: majority label per manufacturer / per year
            tm = pd.Series({t: bear.fs_manufacturer[t] for t in train})
            ty = pd.Series({t: str(dates.loc[t, 'date'])[:4] for t in train})
            tl = pd.Series({t: BEARINGS[t]["label3"] for t in train})
            pm = tl[tm == man].mode().iloc[0] if (tm == man).any() else "IR"
            py = tl[ty == yr].mode().iloc[0] if (ty == yr).any() else "IR"
            ctrl.append({"fold": s["fold"], "bearing": b, "label3": BEARINGS[b]["label3"], "manufacturer": man, "year": yr,
                         "pred_manufacturer_rule": pm, "pred_year_rule": py})
    C = pd.DataFrame(ctrl)
    C.to_csv(OUT / "tables" / "T_metadata_confound_controls.csv", index=False)
    res["metadata_rule_bearing_acc"] = {"manufacturer_only": float((C.pred_manufacturer_rule == C.label3).mean()),
                                        "year_only": float((C.pred_year_rule == C.label3).mean()),
                                        "healthy_recall_manufacturer": float((C[C.label3 == "H"].pred_manufacturer_rule == "H").mean()),
                                        "healthy_recall_year": float((C[C.label3 == "H"].pred_year_rule == "H").mean())}
    (OUT / "physics_diagnostics.json").write_text(json.dumps(res, indent=2, default=float))
    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    pb = per_b.sort_values(["label3", "bearing"])
    for ax, col, lab in ((axes[0], "contrast_med", "family contrast BPFO−BPFI [dB]"), (axes[1], "family_max_med", "max family score [dB]")):
        for i, r in enumerate(pb.itertuples()):
            ax.bar(i, getattr(r, col), color=COL[r.label3])
            if col == "contrast_med":
                ax.plot([i, i], [r.contrast_q25, r.contrast_q75], color="#0b0b0b", lw=0.8)
        ax.set_xticks(range(len(pb)), pb.bearing, rotation=90, fontsize=6.5)
        ax.set_ylabel(lab, fontsize=7)
        ax.axhline(0, color="#52514e", lw=0.6)
    handles = [matplotlib.patches.Patch(color=c, label=k) for k, c in COL.items()]
    axes[1].legend(handles=handles, fontsize=6.5)
    fig.suptitle("Paderborn PU-BD5: physics evidence on UNSEEN test bearings (train-only calibration; median, IQR)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_p2_pu_physics_per_bearing.png", dpi=300)
    plt.close(fig)
    # example envelope spectra of the fold-1 test bearings at N15_M07_F10
    kin = kin_orders_per_window(pu, ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv")
    f1 = fold_features("pu", "PU_BD5")[0][1]
    reg = f1["regions"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True)
    for ax, b in zip(axes, sorted(f1["bearings"]["test"], key=lambda c: BEARINGS[c]["label3"])):
        m = ((meta.bearing == b) & (meta.setting == "N15_M07_F10")).to_numpy()
        E = pu.env[m, 0].mean(0)
        fr = meta.fr_hz[m].mean()
        ko = kin[np.flatnonzero(m)[0]]
        ax.plot(np.arange(len(E)), E, color=COL[BEARINGS[b]["label3"]], lw=0.6)
        for fam, k, c in (("BPFO", ko[0], COL["OR"]), ("BPFI", ko[1], COL["IR"])):
            for h in (1, 2, 3):
                fc = h * k * fr * (1 + reg["centre"][fam])
                hw = h * k * fr * reg["half"][fam]
                ax.axvspan(fc - hw, fc + hw, color=c, alpha=0.18, lw=0)
        ax.set_xlim(0, 500)
        ax.set_yticks([])
        ax.text(0.99, 0.85, f"{b} ({BEARINGS[b]['label3']}, {BEARINGS[b]['method']})", transform=ax.transAxes, ha="right", fontsize=7)
    axes[-1].set_xlabel("envelope frequency [Hz] — mean over 1-s windows, N15_M07_F10; shaded = fold-1 train-calibrated BPFO (aqua) / BPFI (orange) regions, h = 1..3")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig_p2_pu_envelope_examples.png", dpi=300)
    plt.close(fig)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
