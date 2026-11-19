"""Phase 3A-C: physics concept reliability audit + pre-specified confound analyses (protocol V2.1 §9 S-4b, S-4c, S-5).

Concept values are always computed with physics regions calibrated on the OUTER-TRAIN bearings of a fold and
evaluated on bearings outside that training set (BD5 outer-test bearings; artificial bearings, K006, KI17).
The session-identity probe (S-4c) uses label-free kinematic regions (no calibration) and leave-one-bearing-out.
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments import phase2 as P2  # noqa: E402
from pave.experiments import phase3a as P3  # noqa: E402
from pave.features import physics as PH  # noqa: E402

WIN = ROOT / "results" / "phase2" / "windows"
OUT = ROOT / "results" / "phase3a" / "concepts"
FN = PH.FEATURE_NAMES
# expected direction: (positive class, negative class) for which the concept should be HIGHER in the positive class
EXPECT = {"BPFO_snr1": ("OR", "H"), "BPFO_harm": ("OR", "H"), "BPFO_nharm": ("OR", "H"), "BPFO_sb": ("OR", "H"),
          "BPFI_snr1": ("IR", "H"), "BPFI_harm": ("IR", "H"), "BPFI_nharm": ("IR", "H"), "BPFI_sb": ("IR", "H"),
          "contrast": ("OR", "IR")}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = P2.load_ds(WIN, "pu")
    kin = P2.kin_orders_per_window(ds, ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv")
    scan = pd.read_csv(WIN / "pu" / "physics_scan.csv")
    meta = ds.meta
    oos_extra = {b for b, v in BEARINGS.items() if "artificial" in v["group"]} | {"K006", "KI17"}
    eval_bearings = set().union(*P3.ROWS) | oos_extra
    idx = np.flatnonzero(meta.bearing.isin(eval_bearings).to_numpy())
    cache = P3.PhysicsCache(ds, kin, scan, idx)
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    dates = pd.read_csv(ROOT / "results/phase2/pu_audit/measurement_dates_per_bearing.csv", index_col=0)
    rows = []
    for o in P3.outer_folds():
        F, reg = cache.features(o["train"])
        # training-only healthy reference: q95 of each concept over outer-train HEALTHY windows
        trH = meta.bearing.isin([b for b in o["train"] if BEARINGS[b]["label3"] == "H"]).to_numpy()
        q95 = np.nanquantile(F[trH], 0.95, axis=0)
        for b in sorted(o["test"] | oos_extra):
            m = (meta.bearing == b).to_numpy()
            for i in np.flatnonzero(m):
                r = {"fold": o["fold"], "bearing": b, "role": "BD5_test" if b in o["test"] else "extra_oos",
                     "label3": BEARINGS[b]["label3"], "origin": "artificial" if "artificial" in BEARINGS[b]["group"] else
                     ("healthy" if BEARINGS[b]["group"] == "healthy" else "real"),
                     "setting": meta.setting.iat[i], "manufacturer": bear.fs_manufacturer[b], "year": str(dates.loc[b, "min"])[:4]}
                r.update({k: F[i, j] for j, k in enumerate(FN)})
                r.update({f"{k}_gt_trainHq95": float(F[i, j] > q95[j]) if np.isfinite(F[i, j]) else np.nan for j, k in enumerate(FN)})
                rows.append(r)
    W = pd.DataFrame(rows)
    W.to_csv(OUT / "concept_windows.csv", index=False)
    bd5 = W[W.role == "BD5_test"]
    # ---------- reliability tables
    rel = []

    def add(scope, sub, unit):
        for k in FN:
            p, n = EXPECT[k]
            if unit == "window":
                a = auc(sub[sub.label3 == p][k].to_numpy(), sub[sub.label3 == n][k].to_numpy())
                hf = auc(sub[sub.label3 != "H"][k].to_numpy(), sub[sub.label3 == "H"][k].to_numpy()) if k != "contrast" else np.nan
                oi = auc(sub[sub.label3 == "OR"][k].to_numpy(), sub[sub.label3 == "IR"][k].to_numpy())
            else:
                med = sub.groupby(["bearing", "label3"])[k].median().reset_index()
                a = auc(med[med.label3 == p][k].to_numpy(), med[med.label3 == n][k].to_numpy())
                hf = auc(med[med.label3 != "H"][k].to_numpy(), med[med.label3 == "H"][k].to_numpy()) if k != "contrast" else np.nan
                oi = auc(med[med.label3 == "OR"][k].to_numpy(), med[med.label3 == "IR"][k].to_numpy())
            rel.append({"scope": scope, "unit": unit, "concept": k, "expected": f"{p}>{n}", "auroc_expected": a,
                        "auroc_faulty_vs_healthy": hf, "auroc_OR_vs_IR": oi,
                        "n_bearings": sub.bearing.nunique(), "n_windows": len(sub)})

    add("BD5 real (outer-test, 15 bearings)", bd5, "window")
    add("BD5 real (outer-test, 15 bearings)", bd5, "bearing")
    for s, g in bd5.groupby("setting"):
        add(f"BD5 real, setting {s}", g, "window")
    # artificial: damaged artificial windows (each fold's calibration) vs that fold's BD5 healthy test bearing
    art = []
    for fold, g in W.groupby("fold"):
        sub = pd.concat([g[(g.origin == "artificial")], g[(g.role == "BD5_test") & (g.label3 == "H")]])
        sub = sub.assign(fold=fold)
        art.append(sub)
    A = pd.concat(art)
    for k in FN:
        p, n = EXPECT[k]
        per_fold = [auc(g[g.label3 == p][k].to_numpy(), g[g.label3 == n][k].to_numpy()) for _, g in A.groupby("fold")]
        per_fold_oi = [auc(g[g.label3 == "OR"][k].to_numpy(), g[g.label3 == "IR"][k].to_numpy()) for _, g in A.groupby("fold")]
        rel.append({"scope": "ARTIFICIAL damaged vs fold BD5-test healthy (mean of 5 calibrations)", "unit": "window", "concept": k,
                    "expected": f"{p}>{n}", "auroc_expected": float(np.nanmean(per_fold)), "auroc_faulty_vs_healthy": np.nan,
                    "auroc_OR_vs_IR": float(np.nanmean(per_fold_oi)), "n_bearings": A.bearing.nunique(), "n_windows": len(A) // 5})
    R = pd.DataFrame(rel)
    R.round(3).to_csv(OUT / "concept_reliability.csv", index=False)
    # ---------- per-bearing table (BD5 outer-test + extra OOS averaged over folds)
    pb = W.groupby(["bearing", "label3", "origin", "manufacturer", "year", "role"]).agg(
        **{f"{k}_med": (k, "median") for k in FN},
        **{f"{k}_frac_gt_trainHq95": (f"{k}_gt_trainHq95", "mean") for k in ("BPFO_harm", "BPFI_harm")},
        n_windows=("fold", "size")).reset_index()
    pb["own_family_frac_gt_trainHq95"] = np.where(pb.label3 == "OR", pb.BPFO_harm_frac_gt_trainHq95,
                                                  np.where(pb.label3 == "IR", pb.BPFI_harm_frac_gt_trainHq95, np.nan))
    pb["contrast_sign_ok"] = np.where(pb.label3 == "OR", pb.contrast_med > 0, np.where(pb.label3 == "IR", pb.contrast_med < 0, np.nan))
    pb.round(3).to_csv(OUT / "concept_per_bearing.csv", index=False)
    # per bearing x setting stability of own-family evidence (BD5 test)
    st = bd5.copy()
    st["own_harm"] = np.where(st.label3 == "OR", st.BPFO_harm, np.where(st.label3 == "IR", st.BPFI_harm, np.nanmax(st[["BPFO_harm", "BPFI_harm"]], axis=1)))
    bys = st.groupby(["bearing", "label3", "setting"]).own_harm.median().unstack().round(2)
    bys.to_csv(OUT / "own_family_harm_by_bearing_setting.csv")
    # ---------- S-4b within-stratum healthy-vs-faulty (bearing medians)
    strata = {}
    fam = bd5.assign(fammax=np.nanmax(bd5[["BPFO_harm", "BPFI_harm"]], axis=1))
    for name, faulty in (("IBU: K001-K005 vs KA22", ["KA22"]), ("2015: K001-K005 vs KA30, KI04", ["KA30", "KI04"])):
        H = fam[fam.label3 == "H"].groupby("bearing").fammax.median()
        Fm = fam[fam.bearing.isin(faulty)].groupby("bearing").fammax.median()
        strata[name] = {"healthy_median_fammax_dB": H.round(2).to_dict(), "faulty_median_fammax_dB": Fm.round(2).to_dict(),
                        "bearing_level_auroc": auc(Fm.to_numpy(), H.to_numpy()),
                        "window_level_auroc": auc(fam[fam.bearing.isin(faulty)].fammax.to_numpy(), fam[fam.label3 == "H"].fammax.to_numpy())}
    # ---------- S-4c session-identity probe (label-free kinematic regions; LOBO)
    reg0 = PH.Regions()
    fgrid = np.arange(ds.env.shape[2], dtype=float)
    probe_sets = {
        "year_2014_vs_2015_within_IR_bearings": ([b for b, v in BEARINGS.items() if v["label3"] == "IR"],
                                                 lambda b: int(str(dates.loc[b, "min"])[:4] == "2015")),
        "IBU_vs_nonIBU_within_damaged_bearings": ([b for b, v in BEARINGS.items() if v["label3"] in ("IR", "OR")],
                                                  lambda b: int(bear.fs_manufacturer[b] == "IBU")),
        "EXPLORATORY_IBU_vs_MTK_within_artificial": ([b for b, v in BEARINGS.items() if "artificial" in v["group"]],
                                                     lambda b: int(bear.fs_manufacturer[b] == "IBU")),
    }
    probe = {}
    for name, (bset, target) in probe_sets.items():
        m = meta.bearing.isin(bset).to_numpy()
        ii = np.flatnonzero(m)
        Xh = ds.hand[ii]
        Xc = np.array([[PH.window_features(fgrid, ds.env[i, 0].astype(float), meta.fr_hz.iat[i],
                                           {"BPFO": kin[i, 0], "BPFI": kin[i, 1]}, reg0)[k] for k in FN] for i in ii])
        bb = meta.bearing.to_numpy()[ii]
        yy = np.array([target(b) for b in bb])
        res = {"n_bearings": len(bset), "n_positive_bearings": int(sum(target(b) for b in bset)),
               "positive_bearings": sorted(b for b in bset if target(b)),
               "artificial_share_pos": float(np.mean(["artificial" in BEARINGS[b]["group"] for b in bset if target(b)])),
               "artificial_share_neg": float(np.mean(["artificial" in BEARINGS[b]["group"] for b in bset if not target(b)]))}
        for fname, X, mk in (("handcrafted_RF", Xh, lambda: RandomForestClassifier(300, min_samples_leaf=2, n_jobs=4, random_state=42)),
                             ("concepts_LR", Xc, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)))):
            bscore = {}
            for b in sorted(set(bb)):
                te = bb == b
                Xtr, Xte = P3.impute(X[~te], X[te])
                mdl = mk().fit(Xtr, yy[~te])
                bscore[b] = float(mdl.predict_proba(Xte)[:, 1].mean())
            ys = np.array([target(b) for b in bscore])
            res[fname] = {"bearing_level_auroc": float(roc_auc_score(ys, list(bscore.values()))),
                          "bearing_scores": {k: round(v, 3) for k, v in bscore.items()}}
        probe[name] = res
    (OUT / "confound_analyses.json").write_text(json.dumps({"S-4b_within_stratum": strata, "S-4c_session_identity_probe": probe},
                                                          indent=2, default=float))
    # ---------- figure: bearing-level concept medians by class
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    col = {"H": "#2a78d6", "IR": "#eb6834", "OR": "#1baf7a"}
    bb5 = pb[pb.role == "BD5_test"]
    for ax, k in zip(axes, ("BPFO_harm_med", "BPFI_harm_med", "contrast_med")):
        for j, lab in enumerate(("H", "IR", "OR")):
            v = bb5[bb5.label3 == lab][k]
            ax.scatter(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), v, color=col[lab], s=20, zorder=3)
            for x, (bname, val) in zip(np.full(len(v), j) + np.linspace(-0.15, 0.15, len(v)), zip(bb5[bb5.label3 == lab].bearing, v)):
                ax.annotate(bname, (x, val), fontsize=5, xytext=(3, 0), textcoords="offset points")
        art = pb[pb.origin == "artificial"]
        for j, lab in ((1, "IR"), (2, "OR")):
            v = art[art.label3 == lab][k]
            ax.scatter(np.full(len(v), j + 0.35), v, facecolors="none", edgecolors=col[lab], s=16, zorder=3)
        ax.set_xticks([0, 1, 2], ["H", "IR", "OR"])
        ax.set_title(k.replace("_med", "") + " [dB]", fontsize=8)
        ax.axhline(0, color="#52514e", lw=0.5)
    fig.suptitle("Bearing-level concept medians (filled: BD5 real, outer-test calibration; open: artificial)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_concepts_bearing_level.png", dpi=300)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())
    print(json.dumps({"S-4b": strata, "S-4c": {k: {kk: (vv["bearing_level_auroc"] if isinstance(vv, dict) and "bearing_level_auroc" in vv else vv)
                                                      for kk, vv in v.items()} for k, v in probe.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
