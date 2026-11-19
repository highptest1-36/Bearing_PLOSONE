"""Single external evaluation on Ottawa (protocol EV-V1). Refuses to run twice (results/phase4/EVALUATED.lock).

Requires: results/phase3b/folds/FINAL.{json,npz} (models trained on all 15 PU-BD5 bearings), O1-O5 fold outputs,
results/phase4/ottawa_features.npz (from external_ottawa.py --stage features) and the EV-V1 freeze file.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pave.calibration.temperature import apply_temperature, fit_temperature  # noqa: E402
from pave.data import ottawa as OT  # noqa: E402
from pave.evaluation.metrics import ece  # noqa: E402
from pave.experiments import phase2 as P2  # noqa: E402
from pave.experiments import phase3a as P3A  # noqa: E402
from pave.experiments import phase3b as P3B  # noqa: E402
from pave.features import physics_v3 as V3  # noqa: E402
from pave.models.cnn1d import CNN1D  # noqa: E402
from pave.models.mlp import ConceptMLP  # noqa: E402
from pave.models.pavenet import ConceptBottleneck, PAVENet  # noqa: E402
from pave.training import multi as TR  # noqa: E402

OUT = ROOT / "results" / "phase4"
TAB = OUT / "tables"
F3B = ROOT / "results" / "phase3b"
MODELS = ["B0", "B1-V2", "B1-V3-1s", "B1-V3-4s", "B2", "B3", "B4", "A2", "A3", "A4", "A6"]
NEURAL = {"B2": ("mlp", lambda: ConceptMLP(16, 3, 64, 0.1)), "B3": ("cnn", lambda: CNN1D(1, 3)),
          "B4": ("pave", lambda: PAVENet(16, 7, 3, mode="physics")), "A2": ("pave", lambda: PAVENet(16, 7, 3, mode="physics", head_q=True)),
          "A3": ("pave", lambda: PAVENet(16, 7, 3, mode="always")), "A4": ("pave", lambda: PAVENet(16, 7, 3, mode="gated")),
          "A6": ("cbm", lambda: ConceptBottleneck(8, 3, 1))}
L3 = P3B.L3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def macro(y, p):
    return float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def verify_freeze():
    fr = (ROOT / "docs/PROTOCOL_FREEZE_EV_V1.txt").read_text()
    for line in fr.splitlines():
        if line.startswith("FILE "):
            _, rel, h = line.split()
            if sha(ROOT / rel) != h:
                raise SystemExit(f"hash mismatch for {rel}: evaluation aborted")
    return fr


def pu_artifacts(device):
    """Rebuild PU-final logistic models, standardisation statistics, temperatures and S4 (all from PU only)."""
    ds = P2.load_ds(ROOT / "results/phase2/windows", "pu")
    meta = ds.meta
    meas = pd.read_csv(F3B / "env4s/meas.csv")
    w2m = np.load(F3B / "env4s/win2meas.npy")
    fin = json.loads((F3B / "folds/FINAL.json").read_text())
    z = np.load(F3B / "folds/FINAL.npz")
    bd5 = sorted(fin["train_bearings"])
    tr_w = np.flatnonzero(meta.bearing.isin(bd5).to_numpy())
    tr_m = np.flatnonzero(meas.bearing.isin(bd5).to_numpy())
    y_w = np.array([P3B.LIDX.get(v, -1) if isinstance(v, str) else -1 for v in meta.label3])
    y_m = np.array([P3B.LIDX.get(v, -1) if isinstance(v, str) else -1 for v in meas.label3.fillna("")])
    C4, M4, Q4 = z["meas_C4"], z["meas_M4"], z["meas_Q4"]
    Z4, mu4, sd4 = V3.standardise(C4, M4, tr_m)
    qmu, qsd = Q4[tr_m].mean(0), np.where(Q4[tr_m].std(0) > 1e-8, Q4[tr_m].std(0), 1.0)
    art = {"mu4": mu4, "sd4": sd4, "qmu": qmu, "qsd": qsd, "tau": fin["b0_tau_dB"]}
    art["lr4"] = P3A.lr_model(1.0).fit(np.hstack([Z4, M4])[tr_m], y_m[tr_m])
    # 1-s V3 and V2 (rebuild caches with the same code paths as run_phase3b)
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    from pave.signal.physics import BearingGeometry, orders
    kin_b = {c: (lambda o: (o["BPFO"], o["BPFI"]))(orders(BearingGeometry(c, int(r.geom_n_balls), r.geom_ball_diameter_mm, r.geom_pitch_diameter_mm, r.geom_contact_angle_deg)))
             for c, r in bear.iterrows()}
    v1 = P3B.V3Cache(np.arange(1001, dtype=float), ds.env[:, 0], meta.bearing.to_numpy(), meta.label3.fillna("").to_numpy(), meta.fr_hz.to_numpy(),
                     meas.sigma_fr_hz.to_numpy()[w2m], np.array([kin_b[b] for b in meta.bearing]), rows=tr_w)
    C1, M1, _, _ = v1.features(bd5)
    Z1, mu1, sd1 = V3.standardise(C1, M1, tr_w)
    art.update(mu1=mu1, sd1=sd1, lr1=P3A.lr_model(1.0).fit(np.hstack([Z1, M1])[tr_w], y_w[tr_w]))
    kin_v2 = P2.kin_orders_per_window(ds, ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv")
    v2 = P3A.PhysicsCache(ds, kin_v2, pd.read_csv(ROOT / "results/phase2/windows/pu/physics_scan.csv"), tr_w)
    F2, _ = v2.features(frozenset(bd5))
    F2tr, = P3A.impute(F2[tr_w])
    art["v2_median"] = np.where(np.isfinite(np.nanmedian(F2[tr_w], 0)), np.nanmedian(F2[tr_w], 0), 0.0)
    art["lr_v2"] = P3A.lr_model(1.0).fit(F2tr, y_w[tr_w])
    # temperatures + S4 + PU acceptance thresholds from pooled outer-fold test predictions
    temps, s4_X, s4_y, pu_scores = {}, [], [], {}
    folds = [(json.loads((F3B / f"folds/O{k}.json").read_text()), np.load(F3B / f"folds/O{k}.npz")) for k in range(1, 6)]
    yt = np.concatenate([zz["test_y"] for _, zz in folds])
    bt_b = np.concatenate([meta.bearing.to_numpy()[zz["test_idx"]] for _, zz in folds])
    for m in MODELS:
        if m == "B0":
            continue
        temps[m] = fit_temperature(np.concatenate([zz[f"test_{m}"] for _, zz in folds]), yt)
    art["temps"] = temps
    return art, folds, meta, w2m, yt, bt_b


def s4_fit(base, art, folds, meta, w2m, yt):
    X, y = [], []
    for js, zz in folds:
        P = apply_temperature(zz[f"test_{base}"], art["temps"][base])
        mt = w2m[zz["test_idx"]]
        k = P3B.compatibility(zz["meas_C4"][mt], zz["meas_M4"][mt], P.argmax(1), js["b0_tau_dB"])
        X.append(np.c_[P.max(1), k, zz["meas_Q4"][mt, 2]])
        y.append(P.argmax(1) == zz["test_y"])
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(np.concatenate(X), np.concatenate(y))


def bearing_scores(P, C, M, Q, tau, s4, units):
    pred = P.argmax(1)
    kap = P3B.compatibility(C, M, pred, tau)
    w = pd.DataFrame({"u": units, "S4": s4.predict_proba(np.c_[P.max(1), kap, Q[:, 2]])[:, 1], "S3": Q[:, 2] * np.abs(Q[:, 3] - tau),
                      "p0": P[:, 0], "p1": P[:, 1], "p2": P[:, 2]})
    g = w.groupby("u").mean()
    pp = g[["p0", "p1", "p2"]].to_numpy()
    g["S1"] = pp.max(1)
    g["S2"] = np.sum(pp * np.log(np.clip(pp, 1e-12, 1)), 1)
    return g


def main():
    global OUT, TAB
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", default=None, help="SYNTHETIC code-path test: read features from and write to this dir; no lock")
    args = ap.parse_args()
    if args.dry_run:
        OUT = Path(args.dry_run)
        TAB = OUT / "tables"
        TAB.mkdir(parents=True, exist_ok=True)
        run(lock=None, freeze="DRY RUN")
        return
    lock = OUT / "EVALUATED.lock"
    if lock.exists():
        raise SystemExit("Ottawa has already been evaluated (EV-V1 single-evaluation rule).")
    freeze = verify_freeze()
    TAB.mkdir(parents=True, exist_ok=True)
    run(lock=lock, freeze=freeze)


def run(lock, freeze):
    if lock is not None:
        lock.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + freeze)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    art, folds, meta, w2m, yt, bt_b = pu_artifacts(device)
    F = np.load(OUT / "ottawa_features.npz")
    stems = F["stem"]
    seg4 = F["seg4"]
    # map each 1-s window to its 4-s record row: record rows are ordered (file, segment)
    seg_tab = pd.read_csv(OUT / "ottawa_segments4s.csv")
    rid = {(s, g): i for i, (s, g) in enumerate(zip(seg_tab.stem, seg_tab.segment4))}
    wseg = np.array([rid[(s, g if g >= 0 else 1)] for s, g in zip(stems, seg4)])
    C4, M4, Q4 = F["C4"], F["M4"], F["Q4"]
    Z4 = M4 * (C4 - art["mu4"]) / art["sd4"]
    Q4s = (Q4 - art["qmu"]) / art["qsd"]
    Z1 = F["v3_1s_m"] * (F["v3_1s_c"] - art["mu1"]) / art["sd1"]
    V2 = F["v2"].copy()
    r, c = np.where(~np.isfinite(V2))
    V2[r, c] = art["v2_median"][c]
    n = len(stems)
    P = {"B0": P3B.b0_predict(C4, M4, art["tau"])[wseg],
         "B1-V3-4s": P3A.full_proba(art["lr4"], np.hstack([Z4, M4]))[wseg],
         "B1-V3-1s": P3A.full_proba(art["lr1"], np.hstack([Z1, F["v3_1s_m"]])),
         "B1-V2": P3A.full_proba(art["lr_v2"], V2)}
    # pre-specified sensitivity: primary physics model with sigma_r = 1 % of f_r
    Z4s = F["M4_sens"] * (F["C4_sens"] - art["mu4"]) / art["sd4"]
    P["B1-V3-4s_sigma1pct"] = P3A.full_proba(art["lr4"], np.hstack([Z4s, F["M4_sens"]]))[wseg]
    D = {"raw": F["raw32"], "y": np.zeros(n, int), "zm": np.hstack([Z4, M4])[wseg].astype(np.float32), "q": Q4s[wseg].astype(np.float32)}
    Dm = dict(D, raw=None, zm=np.hstack([Z1, F["v3_1s_m"]]).astype(np.float32))
    idx = np.arange(n)
    parts_a4 = []
    ckrows = []
    for m, (kind, mk) in NEURAL.items():
        ps, ps_rec = [], []
        for s in (42, 43, 44):
            cp = F3B / "checkpoints" / m / "FINAL" / f"s{s}.pt"
            ckrows.append({"model": m, "seed": s, "sha256": sha(cp)})
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            net = mk()
            net.load_state_dict(ck["state_dict"])
            net = net.to(device)
            Dx = Dm if m == "B2" else D
            p, pr = TR.predict(net, kind, Dx, idx, device, rs=ck["raw_stats"], parts=True)
            ps.append(p)
            if m == "A4":
                parts_a4.append(pr["gate"].ravel())
            if m in ("B3", "A4", "A6"):  # pre-specified secondary: per-recording standardisation of CNN input
                raw = F["raw32"].astype(np.float32)
                zr = np.empty_like(raw)
                for st in np.unique(stems):
                    k = stems == st
                    zr[k] = (raw[k] - raw[k].mean()) / (raw[k].std() + 1e-8)
                ps_rec.append(TR.predict(net, kind, dict(Dx, raw=zr), idx, device, rs=None))
        P[m] = np.mean(ps, 0)
        if ps_rec:
            P[f"{m}_recnorm"] = np.mean(ps_rec, 0)
    pd.DataFrame(ckrows).to_csv(TAB / "checkpoints_evaluated.csv", index=False)
    for m in list(P):
        base = m.split("_")[0]  # variants (_recnorm, _sigma1pct) use the base model's PU temperature
        if base != "B0" and base in art["temps"]:
            P[f"{m}"] = apply_temperature(P[m], art["temps"][base])
    info = pd.DataFrame([OT.parse_stem(s) for s in stems])
    lab = [OT.label_mapping(OT.parse_stem(s)) for s in stems]
    info["primary"] = [d["primary"] for d in lab]
    info["role"] = [d["role"] for d in lab]
    info["dev_label"] = [d.get("developing_label") for d in lab]
    np.savez_compressed(OUT / "ottawa_window_probs.npz", **{k: v.astype(np.float32) for k, v in P.items()}, stem=stems)
    # ------------- recording-level evaluation
    rows, rec_tabs = [], {}
    prim = info.role == "primary"
    rec = info[prim].groupby("stem").first()
    yrec = rec.primary.map({k: i for i, k in enumerate(L3)}).to_numpy()
    rng = np.random.default_rng(3)
    bearings = sorted(rec.bearing.unique())
    for m, Pm in P.items():
        g = pd.DataFrame(Pm[prim.to_numpy()], columns=["pH", "pIR", "pOR"]).assign(stem=stems[prim.to_numpy()]).groupby("stem").mean().loc[rec.index]
        pred = g.to_numpy().argmax(1)
        rec_tabs[m] = g.assign(y=yrec, pred=pred, bearing=rec.bearing.to_numpy())
        yb, pb = yrec, pred
        boots = []
        for _ in range(10000):
            sel = rng.choice(bearings, len(bearings))
            ii = np.concatenate([np.flatnonzero(rec.bearing.to_numpy() == b) for b in sel])
            boots.append(macro(yb[ii], pb[ii]))
        faulty = yb != 0
        wy = info[prim].primary.map({k: i for i, k in enumerate(L3)}).to_numpy()
        wp = Pm[prim.to_numpy()]
        rows.append({"model": m, "macro_f1": macro(yb, pb), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                     "bal_acc": float(np.mean([np.mean(pb[yb == k] == k) for k in range(3)])),
                     "recall_H_IR_OR": [float(np.mean(pb[yb == k] == k)) for k in range(3)],
                     "healthy_false_alarm": float(np.mean(pb[yb == 0] != 0)), "fault_sensitivity": float(np.mean(pb[faulty] != 0)),
                     "localisation_on_detected": float(np.mean(pb[faulty & (pb != 0)] == yb[faulty & (pb != 0)])) if (faulty & (pb != 0)).any() else float("nan"),
                     "window_nll": float(-np.mean(np.log(np.clip(wp[np.arange(len(wy)), wy], 1e-12, 1)))) if not m.startswith("B0") else None,
                     "window_ece": ece(wp, wy) if not m.startswith("B0") else None,
                     "window_brier": float(np.mean(np.sum((wp - np.eye(3)[wy]) ** 2, 1))) if not m.startswith("B0") else None,
                     "confusion": pd.crosstab(pd.Series(yb, name="y"), pd.Series(pb, name="p")).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values.tolist()})
    R = pd.DataFrame(rows)
    (TAB / "external_primary.json").write_text(R.to_json(orient="records", indent=1))
    R.assign(**{c: R[c].astype(str) for c in ("ci95", "recall_H_IR_OR", "confusion")}).to_csv(TAB / "external_primary.csv", index=False)
    pd.concat([t.assign(model=m) for m, t in rec_tabs.items()]).round(4).to_csv(TAB / "external_recording_predictions.csv")
    # ------------- developing and OOD (secondary)
    sec = []
    for role in ("developing", "ood_faulty", "ood_developing"):
        k = (info.role == role).to_numpy()
        for m, Pm in P.items():
            g = pd.DataFrame(Pm[k], columns=["pH", "pIR", "pOR"]).assign(stem=stems[k]).groupby("stem").mean()
            pred = g.to_numpy().argmax(1)
            row = {"role": role, "model": m, "n_recordings": len(g), "pred_H": int((pred == 0).sum()), "pred_IR": int((pred == 1).sum()),
                   "pred_OR": int((pred == 2).sum())}
            if role == "developing":
                true = info[k].groupby("stem").dev_label.first().loc[g.index].map({"IR": 1, "OR": 2}).to_numpy()
                row["detected"] = float(np.mean(pred != 0))
                row["correct_type"] = float(np.mean(pred == true))
            sec.append(row)
    pd.DataFrame(sec).to_csv(TAB / "external_secondary.csv", index=False)
    # ------------- selective diagnosis (thresholds from PU)
    sd = []
    for base in ("B1-V3-4s", "A4"):
        s4 = s4_fit(base, art, folds, meta, w2m, yt)
        # PU thresholds at bearing coverage 12/15 and 9/15
        pu = []
        for js, zz in folds:
            Pp = apply_temperature(zz[f"test_{base}"], art["temps"][base])
            mt = w2m[zz["test_idx"]]
            pu.append(bearing_scores(Pp, zz["meas_C4"][mt], zz["meas_M4"][mt], zz["meas_Q4"][mt], js["b0_tau_dB"], s4,
                                     meta.bearing.to_numpy()[zz["test_idx"]]))
        pu = pd.concat(pu)
        k = prim.to_numpy()
        og = bearing_scores(P[base][k], C4[wseg][k], M4[wseg][k], Q4[wseg][k], art["tau"], s4, stems[k]).loc[rec.index]
        corr = (og[["p0", "p1", "p2"]].to_numpy().argmax(1) == yrec)
        for sname in ("S1", "S2", "S3", "S4"):
            th = pu[sname].sort_values(ascending=False).to_numpy()
            order = og[sname].sort_values(ascending=False).index
            err = (~pd.Series(corr, index=og.index).loc[order]).to_numpy().astype(float)
            risk = np.cumsum(err) / np.arange(1, len(err) + 1)
            row = {"base": base, "score": sname, "AURC": float(risk.mean()), "n_errors": int(err.sum()),
                   "error_detection_auroc": float(roc_auc_score(corr, og[sname])) if 0 < corr.sum() < len(corr) else float("nan")}
            for cov, kk in ((0.8, 12), (0.6, 9)):
                acc = og[sname] >= th[kk - 1]
                row[f"pu_threshold_cov{cov}_realised_coverage"] = float(acc.mean())
                row[f"pu_threshold_cov{cov}_risk"] = float(np.mean(~corr[acc.to_numpy()])) if acc.any() else float("nan")
                nm = int(round(cov * len(err)))
                row[f"matched_cov{cov}_risk"] = float(risk[nm - 1])
                row[f"matched_cov{cov}_errors"] = int(err[:nm].sum())
            sd.append(row)
    pd.DataFrame(sd).round(4).to_csv(TAB / "external_selective.csv", index=False)
    # ------------- concept validity + speed diagnostics
    seg_tab.groupby("stem")[["fr_hz", "fr_vib_check_hz", "valid_BPFO", "valid_BPFI"]].mean().round(4).to_csv(TAB / "external_concept_validity_speed.csv")
    if parts_a4:
        pd.DataFrame({"stem": stems, "gate": np.mean(parts_a4, 0)}).groupby("stem").gate.mean().round(4).to_csv(TAB / "external_a4_gate.csv")
    print(R[["model", "macro_f1", "ci95", "bal_acc", "recall_H_IR_OR", "healthy_false_alarm", "fault_sensitivity", "localisation_on_detected",
             "window_nll", "window_ece"]].round(3).to_string())
    print(pd.DataFrame(sd).round(3).to_string())


if __name__ == "__main__":
    main()
