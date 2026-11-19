"""Phase 3A nested baselines B0-B3 on PU-BD5 (protocol V2.1 §4-§9).

  python -I scripts/run_phase3a.py [--folds O1 O2 ...] [--smoke]

Per outer fold: inner leave-one-row-out selection (configs + epoch count) -> temperature on pooled inner OOF ->
refit on the 12 outer-train bearings -> predict the 3 outer-test bearings. Outputs: results/phase3a/baselines/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from pave.calibration.temperature import apply_temperature, fit_temperature  # noqa: E402
from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments import phase2 as P2  # noqa: E402
from pave.experiments import phase3a as P3  # noqa: E402
from pave.models.cnn1d import CNN1D  # noqa: E402
from pave.models.mlp import ConceptMLP  # noqa: E402
from pave.training.curves import fit_curve, fit_fixed  # noqa: E402
from pave.training.engine import predict  # noqa: E402

WIN = ROOT / "results" / "phase2" / "windows"
OUT = ROOT / "results" / "phase3a" / "baselines"
SEEDS = [42, 43, 44]
MLP_GRID = [(16, 1e-2), (16, 1e-4), (64, 1e-2), (64, 1e-4)]  # ordered simplest/most-regularised first
MLP_MAX, MLP_EVERY = 150, 5
CNN_GRID = [1e-3, 3e-4]
CNN_MAX = 20
TIE = 0.005


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mf1(P, y) -> float:
    return float(f1_score(y, P.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0))


def zstats(X16: np.ndarray, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    s = ss = 0.0
    n = 0
    for i in range(0, len(X16), chunk):
        x = X16[i:i + chunk].astype(np.float64)
        s += x.sum(axis=(0, 2))
        ss += (x ** 2).sum(axis=(0, 2))
        n += x.shape[0] * x.shape[2]
    mu = s / n
    sd = np.sqrt(ss / n - mu ** 2) + 1e-8
    return mu.astype(np.float32)[None, :, None], sd.astype(np.float32)[None, :, None]


def select(curves: dict[tuple, list[float]]) -> tuple:
    """curves: key -> per-inner-fold metric. Keys are pre-ordered simplest first; tie within TIE keeps the earlier key."""
    best_key, best = None, -np.inf
    for k, v in curves.items():
        m = float(np.mean(v))
        if m > best + TIE:
            best_key, best = k, m
    return best_key, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT / ("_smoke" if a.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = P2.load_ds(WIN, "pu")
    kin = P2.kin_orders_per_window(ds, ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv")
    scan = pd.read_csv(WIN / "pu" / "physics_scan.csv")
    bd5 = set().union(*P3.ROWS)
    idx_all = np.flatnonzero(ds.meta.bearing.isin(bd5).to_numpy())
    cache = P3.PhysicsCache(ds, kin, scan, idx_all)
    y_all = cache.y
    log(f"loading {len(idx_all)} raw BD5 windows (float16)...")
    X16 = np.asarray(ds.raw[idx_all])
    pos = {g: i for i, g in enumerate(idx_all)}
    bearing = ds.meta.bearing.to_numpy()
    mlp_max, cnn_max = (10, 2) if a.smoke else (MLP_MAX, CNN_MAX)
    mlp_every = 5 if not a.smoke else 5
    for outer in P3.outer_folds():
        if a.folds and outer["fold"] not in a.folds:
            continue
        fjson = out / f"fold_{outer['fold']}.json"
        if fjson.exists():
            log(f"{outer['fold']} done - skipped")
            continue
        P3.check_nested(outer)
        t0 = time.time()
        sel = {}
        inner_oof = {"B1": {}, "B2": {}, "B3": {}}
        curves_b1, curves_b2, curves_b3 = {}, {}, {}
        ys_inner = []
        for inner in P3.inner_folds(outer):
            tr = idx_all[np.isin(bearing[idx_all], list(inner["train"]))]
            va = idx_all[np.isin(bearing[idx_all], list(inner["val"]))]
            F, _ = cache.features(inner["train"])
            Ftr, Fva = P3.impute(F[tr], F[va])
            ytr, yva = y_all[tr], y_all[va]
            ys_inner.append(yva)
            for C in P3.LR_GRID:
                m = P3.lr_model(C).fit(Ftr, ytr)
                Pv = P3.full_proba(m, Fva)
                curves_b1.setdefault((C,), []).append(mf1(Pv, yva))
                inner_oof["B1"].setdefault((C,), []).append(Pv)
            mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
            Ztr, Zva = ((Ftr - mu) / sd).astype(np.float32), ((Fva - mu) / sd).astype(np.float32)
            for h, wd in MLP_GRID:
                r = fit_curve(lambda h=h: ConceptMLP(Ztr.shape[1], 3, h), Ztr, ytr, Zva, 3, 1e-3, wd, 256, mlp_max, 42, device,
                              amp=False, eval_every=mlp_every)
                for ep, Pv in r["val_probs"].items():
                    curves_b2.setdefault((h, wd, ep), []).append(mf1(Pv, yva))
                    inner_oof["B2"].setdefault((h, wd, ep), []).append(Pv)
            Xtr, Xva = X16[[pos[i] for i in tr]], X16[[pos[i] for i in va]]
            cm, cs = zstats(Xtr)
            tf = lambda b, cm=cm, cs=cs: (b.astype(np.float32) - cm) / cs  # noqa: E731
            for lr in CNN_GRID:
                r = fit_curve(lambda: CNN1D(1, 3), Xtr, ytr, Xva, 3, lr, 1e-2, 64, cnn_max, 42, device, transform=tf)
                for ep, Pv in r["val_probs"].items():
                    curves_b3.setdefault((lr, ep), []).append(mf1(Pv, yva))
                    inner_oof["B3"].setdefault((lr, ep), []).append(Pv)
                log(f"{inner['fold']} B3 lr={lr} inner curve done ({r['seconds']:.0f}s)")
            del Xtr, Xva
            torch.cuda.empty_cache()
        yoof = np.concatenate(ys_inner)
        # order keys: B2 simplest first then fewer epochs; B3 fewer epochs first within lr order
        b2_keys = sorted(curves_b2, key=lambda k: (MLP_GRID.index((k[0], k[1])), k[2]))
        b3_keys = sorted(curves_b3, key=lambda k: (k[1], CNN_GRID.index(k[0])))
        sel["B1"] = select(curves_b1)
        sel["B2"] = select({k: curves_b2[k] for k in b2_keys})
        sel["B3"] = select({k: curves_b3[k] for k in b3_keys})
        T = {m: fit_temperature(np.concatenate(inner_oof[m][sel[m][0]]), yoof) for m in ("B1", "B2", "B3")}
        inner_summary = {
            "B1": {str(k): [round(x, 4) for x in v] for k, v in curves_b1.items()},
            "B2_best_per_config": {str(c): max(((float(np.mean(curves_b2[(c[0], c[1], e)])), e) for e in range(mlp_every, mlp_max + 1, mlp_every)))
                                   for c in MLP_GRID},
            "B3_curve_mean": {str(lr): [round(float(np.mean(curves_b3[(lr, e)])), 4) for e in range(1, cnn_max + 1)] for lr in CNN_GRID},
            "B3_curve_per_inner": {str(k): [round(x, 4) for x in v] for k, v in curves_b3.items() if k[0] == sel["B3"][0][0]},
        }
        log(f"{outer['fold']} selected: {sel} | T {T}")
        # ---- outer refit
        tr = idx_all[np.isin(bearing[idx_all], list(outer["train"]))]
        te = idx_all[np.isin(bearing[idx_all], list(outer["test"]))]
        F, reg = cache.features(outer["train"])
        Ftr, Fte = P3.impute(F[tr], F[te])
        ytr, yte = y_all[tr], y_all[te]
        probs = {}
        tau = P3.b0_fit(Ftr, ytr)
        probs["B0"] = P3.b0_predict(Fte, tau)
        m1 = P3.lr_model(sel["B1"][0][0]).fit(Ftr, ytr)
        probs["B1"] = P3.full_proba(m1, Fte)
        probs["B1_default_C1"] = P3.full_proba(P3.lr_model(1.0).fit(Ftr, ytr), Fte)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
        Ztr, Zte = ((Ftr - mu) / sd).astype(np.float32), ((Fte - mu) / sd).astype(np.float32)
        h, wd, ep2 = sel["B2"][0]
        for s in SEEDS:
            net = fit_fixed(lambda: ConceptMLP(Ztr.shape[1], 3, h), Ztr, ytr, 3, 1e-3, wd, 256, ep2, s, device,
                            amp=False, sched_epochs=mlp_max)["model"]
            probs[f"B2_s{s}"] = predict(net, Zte, device)
        Xtr, Xte = X16[[pos[i] for i in tr]], X16[[pos[i] for i in te]]
        cm, cs = zstats(Xtr)
        tf = lambda b, cm=cm, cs=cs: (b.astype(np.float32) - cm) / cs  # noqa: E731
        lr3, ep3 = sel["B3"][0]
        for s in SEEDS:
            r = fit_fixed(lambda: CNN1D(1, 3), Xtr, ytr, 3, lr3, 1e-2, 64, ep3, s, device, transform=tf, sched_epochs=cnn_max)
            probs[f"B3_s{s}"] = predict(r["model"], Xte, device, transform=tf)
            torch.save(r["model"].state_dict(), out / f"B3_{outer['fold']}_s{s}.pt")
            rd = fit_fixed(lambda: CNN1D(1, 3), Xtr, ytr, 3, 1e-3, 1e-2, 64, 2 if a.smoke else 25, s, device, transform=tf)
            probs[f"B3_default_s{s}"] = predict(rd["model"], Xte, device, transform=tf)
            log(f"{outer['fold']} B3 seed {s} refit done ({r['seconds']:.0f}s)")
        del Xtr, Xte
        torch.cuda.empty_cache()
        for m in ("B2", "B3", "B3_default"):
            probs[m] = np.mean([probs[f"{m}_s{s}"] for s in SEEDS], axis=0)
        for m in ("B1", "B2", "B3"):
            probs[f"{m}_TS"] = apply_temperature(probs[m], T[m])
        # metadata-only rules (S-4a) learnt from outer-train bearings
        bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
        dates = pd.read_csv(ROOT / "results/phase2/pu_audit/measurement_dates_per_bearing.csv", index_col=0)
        meta_rules = {}
        for key, getter in (("manufacturer", lambda b: bear.fs_manufacturer[b]), ("year", lambda b: str(dates.loc[b, "min"])[:4])):
            lab = pd.Series({b: BEARINGS[b]["label3"] for b in outer["train"]})
            att = pd.Series({b: getter(b) for b in outer["train"]})
            meta_rules[key] = {b: (lab[att == getter(b)].mode().sort_values().iloc[0] if (att == getter(b)).any() else "IR")
                               for b in outer["test"]}
        np.savez_compressed(out / f"fold_{outer['fold']}.npz", te=te, y=yte, **{k: v.astype(np.float32) for k, v in probs.items()})
        rec = {"fold": outer["fold"], "test_bearings": sorted(outer["test"]), "train_bearings": sorted(outer["train"]),
               "selected": {k: {"config": list(v[0]), "inner_mean_macro_f1": v[1]} for k, v in sel.items()},
               "temperature": T, "b0_tau": tau, "regions": reg.to_dict(), "inner": inner_summary,
               "metadata_rules": meta_rules, "seconds": time.time() - t0, "smoke": a.smoke}
        fjson.write_text(json.dumps(rec, indent=1, default=float))
        log(f"{outer['fold']} finished in {time.time()-t0:.0f}s; test window macro-F1: " +
            ", ".join(f"{m} {mf1(probs[m], yte):.3f}" for m in ("B0", "B1", "B2", "B3", "B3_default")))


if __name__ == "__main__":
    main()
