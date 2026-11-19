"""Phase 3B analysis (protocol V2.2-V2.4): metrics, inclusion rule, PE-SD, shortcut probes. Bearing = unit.

Reads results/phase3b/folds/O*.npz|json; writes results/phase3b/tables/*.csv|json and figures.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pave.calibration.temperature import apply_temperature, fit_temperature  # noqa: E402
from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.evaluation.metrics import ece  # noqa: E402
from pave.experiments import phase3a as P3A  # noqa: E402
from pave.experiments import phase3b as P3B  # noqa: E402

OUT = ROOT / "results" / "phase3b"
TAB = OUT / "tables"
FIG = OUT / "figures"
MODELS = ["B0", "B1-V2", "B1-V3-1s", "B1-V3-4s", "B2", "B3", "B4", "A2", "A3", "A4", "A6"]
NAMES = {"B0": "B0 physics rule (4 s)", "B1-V2": "B1-V2 = P1 (LR, V2 concepts, 1 s)", "B1-V3-1s": "B1-V3-1s (LR)",
         "B1-V3-4s": "B1-V3-4s (LR)", "B2": "B2-V3 MLP (1 s)", "B3": "B3 CNN-1D (1 s raw)",
         "B4": "B4 = A1 physics head (4 s)", "A2": "A2 physics + reliability", "A3": "A3 physics + latent (no gate)",
         "A4": "A4 PAVE-Net (gated)", "A6": "A6 concept bottleneck (CNN-predicted)"}
WEAK = ["KA15", "KA22", "KA30", "KI14", "KI16", "KI21"]
STRONG = ["KA04", "KA16", "KI04", "KI18"]


def macro(y, p):
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def brier(P, y):
    Y = np.eye(3)[y]
    return float(np.mean(np.sum((P - Y) ** 2, 1)))


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def load():
    meta = pd.read_csv(ROOT / "results/phase2/windows/pu/meta.csv")
    folds = []
    for k in range(1, 6):
        js = json.loads((OUT / "folds" / f"O{k}.json").read_text())
        assert not js["smoke"]
        folds.append((js, np.load(OUT / "folds" / f"O{k}.npz")))
    return meta, folds


def bearing_level(P, y, b):
    bt = P3A.bearing_table(P, y, b)
    return bt


def boot_ci(bt, stat, n=10000, seed=1):
    return P3A.stratified_bootstrap(bt, stat, n=n, seed=seed)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    meta, folds = load()
    w2m = np.load(OUT / "env4s" / "win2meas.npy")
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    dates = pd.read_csv(ROOT / "results/phase2/pu_audit/measurement_dates_per_bearing.csv", index_col=0)
    year = {b: str(dates.loc[b, "min"])[:4] for b in dates.index}
    # ---------------- temperature scaling (inner OOF) + pooled test predictions
    T, P_test, P_raw, y_all, b_all, fold_of, seed_P = {}, {m: [] for m in MODELS}, {m: [] for m in MODELS}, [], [], [], {}
    for js, z in folds:
        f = js["fold"]
        yo = z["oof_y"]
        for m in MODELS:
            p = z[f"test_{m}"]
            P_raw[m].append(p)
            if m == "B0":
                P_test[m].append(p)
                continue
            t = fit_temperature(z[f"oof_{m}"], yo)
            T[(m, f)] = t
            P_test[m].append(apply_temperature(p, t))
            for s in (42, 43, 44):
                if f"test_{m}_s{s}" in z.files:
                    seed_P.setdefault((m, s), []).append(z[f"test_{m}_s{s}"])
        y_all.append(z["test_y"])
        b_all.append(meta.bearing.to_numpy()[z["test_idx"]])
        fold_of.append(np.full(len(z["test_y"]), f))
    y = np.concatenate(y_all)
    b = np.concatenate(b_all)
    fo = np.concatenate(fold_of)
    P = {m: np.concatenate(v) for m, v in P_test.items()}
    Praw = {m: np.concatenate(v) for m, v in P_raw.items()}
    # ---------------- main table
    rows, bts = [], {}
    for m in ["B1-V2"] + [x for x in MODELS if x != "B1-V2"]:
        bt = bearing_level(P[m], y, b)
        bts[m] = bt
        bm = P3A.bearing_metrics(bt)
        prec = []
        for k in range(3):
            sel = bt.pred == k
            prec.append(float(np.mean(bt.y[sel] == k)) if sel.any() else float("nan"))
        faulty = bt[bt.y != 0]
        lo, hi = boot_ci(bt, lambda d: macro(d.y, d.pred))
        r = {"model": m, "name": NAMES[m], "bearing_macro_f1": bm["macro_f1"], "ci95": [lo, hi], "bearing_bal_acc": bm["bal_acc"],
             "precision_H_IR_OR": prec, "recall_H_IR_OR": bm["recall"], "fault_sensitivity": float(np.mean(faulty.pred != 0)),
             "healthy_false_alarm": float(np.mean(bt[bt.y == 0].pred != 0)),
             "ir_vs_or_acc_on_detected_faulty": float(np.mean(faulty[faulty.pred != 0].pred == faulty[faulty.pred != 0].y)) if (faulty.pred != 0).any() else float("nan"),
             "confusion": bm["confusion"], "failures": bm["failures"],
             "window_macro_f1": macro(y, P[m].argmax(1)),
             "window_macro_f1_fold": [macro(y[fo == f], P[m][fo == f].argmax(1)) for f in sorted(set(fo))],
             "nll_ts": nll(P[m], y) if m != "B0" else None, "nll_raw": nll(Praw[m], y) if m != "B0" else None,
             "ece_ts": ece(P[m], y) if m != "B0" else None, "ece_raw": ece(Praw[m], y) if m != "B0" else None,
             "brier_ts": brier(P[m], y) if m != "B0" else None,
             "weak_bearings_correct": int(bt[bt.bearing.isin(WEAK)].correct.sum()),
             "strong_bearings_correct": int(bt[bt.bearing.isin(STRONG)].correct.sum()),
             "temperatures": [T.get((m, f)) for f in sorted(set(fo))]}
        if m != "B1-V2":
            for ref in ("B1-V2", "B1-V3-4s"):
                if m == ref or ref not in bts:
                    continue
                a = bts[ref].set_index("bearing")
                o = bt.set_index("bearing").reindex(a.index)
                r[f"vs_{ref}_mcnemar"] = P3A.mcnemar_exact(o.correct.to_numpy(), a.correct.to_numpy())
                both = bts[ref].merge(bt, on=["bearing", "y"], suffixes=("_a", "_b"))
                rng = np.random.default_rng(7)
                g = [both[both.y == k].index.to_numpy() for k in range(3)]
                d = []
                for _ in range(10000):
                    s = both.loc[np.concatenate([rng.choice(x, len(x)) for x in g])]
                    d.append(macro(s.y, s.pred_b) - macro(s.y, s.pred_a))
                r[f"diff_vs_{ref}"] = r["bearing_macro_f1"] - macro(bts[ref].y, bts[ref].pred)
                r[f"diff_vs_{ref}_ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
        rows.append(r)
    # per seed
    seed_rows = []
    for (m, s), v in seed_P.items():
        Ps = np.concatenate(v)
        bt = bearing_level(Ps, y, b)
        seed_rows.append({"model": m, "seed": s, "bearing_macro_f1": macro(bt.y, bt.pred), "bearing_acc": float(bt.correct.mean()),
                          "window_macro_f1": macro(y, Ps.argmax(1)), "failures": " ".join(bt[~bt.correct].bearing)})
    pd.DataFrame(seed_rows).sort_values(["model", "seed"]).round(4).to_csv(TAB / "per_seed.csv", index=False)
    # per fold window table
    pf = pd.DataFrame([{"model": r["model"], **{f"O{i+1}": v for i, v in enumerate(r["window_macro_f1_fold"])}} for r in rows])
    pf.round(4).to_csv(TAB / "per_fold_window_macro_f1.csv", index=False)
    (TAB / "main_results.json").write_text(json.dumps(rows, indent=1, default=float))
    flat = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])
    flat.to_csv(TAB / "main_results.csv", index=False)
    pbm = pd.DataFrame({m: bts[m].set_index("bearing").pred.map(dict(enumerate(P3B.L3))) for m in MODELS})
    pbm.insert(0, "true", [BEARINGS[x]["label3"] for x in pbm.index])
    pbm.insert(1, "evidence", ["weak" if x in WEAK else ("strong" if x in STRONG else "-") for x in pbm.index])
    pbm.to_csv(TAB / "per_bearing_predictions.csv")
    pd.concat([bts[m].assign(model=m) for m in MODELS]).round(4).to_csv(TAB / "per_bearing_probs.csv", index=False)
    # ---------------- inclusion rule (inner OOF only)
    inc = {}
    for m in ("A2", "A4", "A3", "B4", "B1-V3-1s", "B1-V3-4s", "B3", "A6"):
        Ys, Ps_, Bs = [], [], []
        for js, z in folds:
            Ys.append(z["oof_y"])
            Ps_.append(z[f"oof_{m}"])
            Bs.append(np.array([f"{js['fold']}|{x}" for x in meta.bearing.to_numpy()[z["oof__idx"]]]))
        Yc, Pc, Bc = np.concatenate(Ys), np.concatenate(Ps_), np.concatenate(Bs)
        bt = bearing_level(Pc, Yc, Bc)
        inc[m] = {"inner_oof_bearing_macro_f1": macro(bt.y, bt.pred), "n_bearing_predictions": int(len(bt)),
                  "inner_oof_window_nll": nll(Pc, Yc)}
    keep_latent = (inc["A4"]["inner_oof_bearing_macro_f1"] - inc["A2"]["inner_oof_bearing_macro_f1"] >= 0.05
                   and inc["A4"]["inner_oof_window_nll"] <= inc["A2"]["inner_oof_window_nll"])
    phys_final = "B1-V3-4s" if inc["B1-V3-4s"]["inner_oof_bearing_macro_f1"] >= inc["B1-V3-1s"]["inner_oof_bearing_macro_f1"] else "B1-V3-1s"
    final = "A4" if keep_latent else phys_final
    decision = {"inner_oof": inc, "rule": "keep latent iff A4 - A2 >= 0.05 bearing macro-F1 and NLL(A4) <= NLL(A2) on pooled inner OOF",
                "keep_latent_branch": bool(keep_latent), "physics_only_final": phys_final, "final_recommended_model": final}
    (TAB / "inclusion_decision.json").write_text(json.dumps(decision, indent=1, default=float))
    # ---------------- PE-SD
    sd_rows, curves = [], {}
    for base in sorted({"B1-V3-4s", "A4", final}):
        win_scores = {k: [] for k in ("S1", "S2", "S3", "S4")}
        for js, z in folds:
            f = js["fold"]
            t = T[(base, f)]
            Pt = apply_temperature(z[f"test_{base}"], t)
            Po = apply_temperature(z[f"oof_{base}"], t)
            C4, M4, Q4 = z["meas_C4"], z["meas_M4"], z["meas_Q4"]
            tau = js["b0_tau_dB"]
            mt = w2m[z["test_idx"]]
            kap_t = P3B.compatibility(C4[mt], M4[mt], Pt.argmax(1), tau)
            kap_o = P3B.compatibility(z["oof_C4"], z["oof_M4"], Po.argmax(1), z["oof_tau"])  # per-row inner tau
            Xo = np.c_[Po.max(1), kap_o, z["oof_Q4"][:, 2]]
            Xt = np.c_[Pt.max(1), kap_t, Q4[mt, 2]]
            s4 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Xo, Po.argmax(1) == z["oof_y"])
            win_scores["S4"].append(s4.predict_proba(Xt)[:, 1])
            win_scores["S1"].append(Pt.max(1))
            ent = -np.sum(Pt * np.log(np.clip(Pt, 1e-12, 1)), 1)
            win_scores["S2"].append(-ent)
            win_scores["S3"].append(Q4[mt, 2] * np.abs(Q4[mt, 3] - tau))
        Pb = P[base]
        bt = bearing_level(Pb, y, b).set_index("bearing")
        for sname, v in win_scores.items():
            sc = pd.Series(np.concatenate(v)).groupby(b).mean()
            if sname == "S1":
                sc = bt[["pH", "pIR", "pOR"]].max(1)
            if sname == "S2":
                pp = bt[["pH", "pIR", "pOR"]].to_numpy()
                sc = pd.Series(np.sum(pp * np.log(np.clip(pp, 1e-12, 1)), 1), index=bt.index)
            d = bt.join(sc.rename("score"))
            order = d.sort_values("score", ascending=False)
            err = (~order.correct).to_numpy().astype(float)
            risk = np.cumsum(err) / np.arange(1, len(err) + 1)
            curves[(base, sname)] = risk
            r = {"base": base, "score": sname, "AURC": float(risk.mean()),
                 "risk_at_12_of_15": float(risk[11]), "accepted_errors_12": int(err[:12].sum()),
                 "risk_at_9_of_15": float(risk[8]), "accepted_errors_9": int(err[:9].sum()),
                 "rejected_at_12": " ".join(order.index[12:]), "rejected_at_9": " ".join(order.index[9:]),
                 "rejected_classes_at_9": "".join(order.y.map(dict(enumerate(P3B.L3))).iloc[9:]),
                 "error_detection_auroc": float(roc_auc_score(d.correct, d.score)) if d.correct.nunique() == 2 else float("nan"),
                 "n_errors_total": int((~d.correct).sum())}
            sd_rows.append(r)
    pd.DataFrame(sd_rows).round(4).to_csv(TAB / "selective_diagnosis.csv", index=False)
    fig, axes = plt.subplots(1, len({"B1-V3-4s", "A4", final}), figsize=(7.2, 2.6), sharey=True)
    for ax, base in zip(np.atleast_1d(axes), sorted({"B1-V3-4s", "A4", final})):
        for sname, col in (("S1", "#2a78d6"), ("S2", "#86b6ef"), ("S3", "#eb6834"), ("S4", "#1baf7a")):
            r = curves[(base, sname)]
            ax.plot(np.arange(1, 16) / 15, r, marker="o", ms=3, color=col, label=sname)
        ax.set_title(base, fontsize=8)
        ax.set_xlabel("bearing coverage")
    np.atleast_1d(axes)[0].set_ylabel("selective risk (bearing errors / accepted)")
    np.atleast_1d(axes)[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_risk_coverage.png", dpi=300)
    fig.savefig(FIG / "fig_risk_coverage.pdf")
    plt.close(fig)
    # ---------------- shortcut probes (per fold, bearings unseen by that fold's models)
    probe_rows = []
    for js, z in folds:
        ex = z["extra_idx"]
        bb = meta.bearing.to_numpy()[ex]
        reps = {"B3_embedding": z["extra_B3_h"], "A4_latent": z["extra_A4_h"], "A6_predicted_concepts": z["extra_A6_c_hat"],
                "V3_concepts_4s": np.hstack([z["meas_C4"][w2m[ex]], z["meas_M4"][w2m[ex]]])}
        ub = sorted(set(bb))
        targets = {"manufacturer_IBU": {x: int(bear.fs_manufacturer[x] == "IBU") for x in ub},
                   "year_2015": {x: int(year[x] == "2015") for x in ub},
                   "origin_artificial(damaged only)": {x: int("artificial" in BEARINGS[x]["group"]) for x in ub if BEARINGS[x]["label3"] in ("IR", "OR")}}
        for rn, R in reps.items():
            Rb = pd.DataFrame(R).groupby(bb).mean()
            for tn, tg in targets.items():
                keys = [x for x in ub if x in tg]
                yy = np.array([tg[x] for x in keys])
                if len(set(yy)) < 2:
                    continue
                sc = []
                for x in keys:
                    trk = [k for k in keys if k != x]
                    ytr = np.array([tg[k] for k in trk])
                    if len(set(ytr)) < 2:
                        sc.append(0.5)
                        continue
                    mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Rb.loc[trk].to_numpy(), ytr)
                    sc.append(mdl.predict_proba(Rb.loc[[x]].to_numpy())[0, 1])
                probe_rows.append({"fold": js["fold"], "representation": rn, "target": tn, "n_bearings": len(keys),
                                   "n_positive": int(yy.sum()), "bearing_auroc": float(roc_auc_score(yy, sc))})
            # window-level operating-condition probes (grouped by bearing)
            st = meta.setting.to_numpy()[ex]
            spd = (meta.fr_hz.to_numpy()[ex] < 20).astype(int)
            from sklearn.model_selection import GroupKFold
            acc, auc_s = [], []
            for tr, te in GroupKFold(5).split(R, st, bb):
                m1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[tr], st[tr])
                acc.append(np.mean(m1.predict(R[te]) == st[te]))
                m2 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(R[tr], spd[tr])
                auc_s.append(roc_auc_score(spd[te], m2.predict_proba(R[te])[:, 1]))
            probe_rows.append({"fold": js["fold"], "representation": rn, "target": "setting_4class_window_acc", "n_bearings": len(ub),
                               "n_positive": None, "bearing_auroc": float(np.mean(acc))})
            probe_rows.append({"fold": js["fold"], "representation": rn, "target": "speed_900rpm_window_auroc", "n_bearings": len(ub),
                               "n_positive": None, "bearing_auroc": float(np.mean(auc_s))})
    PR = pd.DataFrame(probe_rows)
    PR.round(4).to_csv(TAB / "shortcut_probes_per_fold.csv", index=False)
    PR.groupby(["representation", "target"]).bearing_auroc.agg(["mean", "min", "max"]).round(3).to_csv(TAB / "shortcut_probes_summary.csv")
    # ---------------- A4 gate diagnostics
    gate_rows = []
    for js, z in folds:
        ex = z["extra_idx"]
        g = z["extra_A4_gate"].ravel()
        bb = meta.bearing.to_numpy()[ex]
        for x in sorted(set(bb) & set(js["test_bearings"])):
            gate_rows.append({"fold": js["fold"], "bearing": x, "gate_mean": float(g[bb == x].mean()),
                              "latent_weight_mean": float(1 - g[bb == x].mean()),
                              "evidence": "weak" if x in WEAK else ("strong" if x in STRONG else "healthy")})
    pd.DataFrame(gate_rows).round(4).to_csv(TAB / "a4_gate_by_bearing.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(flat[["model", "bearing_macro_f1", "ci95", "bearing_bal_acc", "recall_H_IR_OR", "fault_sensitivity", "healthy_false_alarm",
                "window_macro_f1", "nll_ts", "ece_ts", "weak_bearings_correct", "strong_bearings_correct", "failures"]].to_string())
    print(json.dumps(decision, indent=1, default=float))
    print(pd.DataFrame(sd_rows)[["base", "score", "AURC", "risk_at_12_of_15", "risk_at_9_of_15", "error_detection_auroc", "n_errors_total", "rejected_at_9"]].round(3).to_string())
    print(PR.groupby(["representation", "target"]).bearing_auroc.mean().round(3).unstack().to_string())
    print(pd.DataFrame(gate_rows).groupby("evidence").gate_mean.mean().round(3).to_string())


if __name__ == "__main__":
    main()
