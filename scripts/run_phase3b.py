"""Phase 3B (protocol V2.2 + V2.3): all models on PU-BD5, 5 outer folds; inner OOF (seed 42) for calibration only.

  python -I scripts/run_phase3b.py [--folds O1 ...] [--smoke] [--final]

--final trains the external-validation models on all 15 PU-BD5 bearings (only after the Phase 3B GO gate).
Outputs: results/phase3b/folds/<fold>.npz/.json, checkpoints results/phase3b/checkpoints/<model>/<fold>/s<seed>.pt
(+ checkpoints.csv with SHA-256). Existing fold outputs / checkpoints are never overwritten (resume = skip).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments import phase2 as P2  # noqa: E402
from pave.experiments import phase3a as P3A  # noqa: E402
from pave.experiments import phase3b as P3B  # noqa: E402
from pave.features import physics_v3 as V3  # noqa: E402
from pave.models.cnn1d import CNN1D  # noqa: E402
from pave.models.mlp import ConceptMLP  # noqa: E402
from pave.models.pavenet import ConceptBottleneck, PAVENet, n_params  # noqa: E402
from pave.training import multi as TR  # noqa: E402

WIN = ROOT / "results" / "phase2" / "windows"
E4 = ROOT / "results" / "phase3b" / "env4s"
OUT = ROOT / "results" / "phase3b"
SEEDS = [42, 43, 44]
CFG = {  # protocol V2.2 §5 (fixed)
    "B2": dict(kind="mlp", epochs=20, lr=1e-3, wd=1e-2, bs=256),
    "B3": dict(kind="cnn", epochs=25, lr=1e-3, wd=1e-2, bs=64),
    "B4": dict(kind="pave", epochs=25, lr=1e-3, wd=1e-2, bs=64, mode="physics", head_q=False),
    "A2": dict(kind="pave", epochs=25, lr=1e-3, wd=1e-2, bs=64, mode="physics", head_q=True),
    "A3": dict(kind="pave", epochs=25, lr=1e-3, wd=1e-2, bs=64, mode="always", head_q=False),
    "A4": dict(kind="pave", epochs=25, lr=1e-3, wd=1e-2, bs=64, mode="gated", head_q=False),
    "A6": dict(kind="cbm", epochs=25, lr=1e-3, wd=1e-2, bs=64),
}
NEED_RAW = {"B3", "A3", "A4", "A6"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def maker(name):
    c = CFG[name]
    if c["kind"] == "cnn":
        return lambda: CNN1D(1, 3)
    if c["kind"] == "mlp":
        return lambda: ConceptMLP(16, 3, 64, 0.1)
    if c["kind"] == "cbm":
        return lambda: ConceptBottleneck(8, 3, 1)
    return lambda: PAVENet(16, 7, 3, mode=c["mode"], head_q=c["head_q"])


def raw_stats(raw16, idx, chunk=512):
    s = ss = 0.0
    n = 0
    for i in range(0, len(idx), chunk):
        x = raw16[idx[i:i + chunk]].astype(np.float64)
        s += x.sum(axis=(0, 2))
        ss += (x ** 2).sum(axis=(0, 2))
        n += x.shape[0] * x.shape[2]
    mu = s / n
    return mu.astype(np.float32)[None, :, None], (np.sqrt(ss / n - mu ** 2) + 1e-8).astype(np.float32)[None, :, None]


def lr_probs(Xtr, ytr, Xev):
    m = P3A.lr_model(1.0).fit(Xtr, ytr)
    return P3A.full_proba(m, Xev), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    out = OUT / ("_smoke" if a.smoke else "")
    (out / "folds").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = P2.load_ds(WIN, "pu")
    meta = ds.meta
    meas = pd.read_csv(E4 / "meas.csv")
    env4 = np.load(E4 / "env4.npy")
    w2m = np.load(E4 / "win2meas.npy")
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    kin_b = {}
    from pave.signal.physics import BearingGeometry, orders
    for code, r in bear.iterrows():
        o = orders(BearingGeometry(code, int(r.geom_n_balls), r.geom_ball_diameter_mm, r.geom_pitch_diameter_mm, r.geom_contact_angle_deg))
        kin_b[code] = (o["BPFO"], o["BPFI"])
    bd5 = sorted(set().union(*P3A.ROWS))
    nonbd5 = sorted(b for b in BEARINGS if b not in bd5)
    win_bd5 = np.flatnonzero(meta.bearing.isin(bd5).to_numpy())
    win_all = np.arange(len(meta))
    y_win = np.array([P3B.LIDX.get(v, -1) if isinstance(v, str) else -1 for v in meta.label3])
    y_meas = np.array([P3B.LIDX.get(v, -1) if isinstance(v, str) else -1 for v in meas.label3.fillna("")])
    log("V3 scans (label-free) ...")
    v4 = P3B.V3Cache(np.arange(4001) * 0.25, env4, meas.bearing.to_numpy(), meas.label3.fillna("").to_numpy(),
                     meas.fr_hz.to_numpy(), meas.sigma_fr_hz.to_numpy(), np.array([kin_b[b] for b in meas.bearing]))
    sig_win = meas.sigma_fr_hz.to_numpy()[w2m]
    v1 = P3B.V3Cache(np.arange(1001, dtype=float), ds.env[:, 0], meta.bearing.to_numpy(), meta.label3.fillna("").to_numpy(),
                     meta.fr_hz.to_numpy(), sig_win, np.array([kin_b[b] for b in meta.bearing]), rows=win_bd5)
    kin_v2 = P2.kin_orders_per_window(ds, ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv")
    scan_v2 = pd.read_csv(WIN / "pu" / "physics_scan.csv")
    v2 = P3A.PhysicsCache(ds, kin_v2, scan_v2, win_bd5)
    log("loading raw windows (float16) ...")
    raw_all = np.asarray(ds.raw)  # (N, 1, 32000) float16
    env_info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "numpy": np.__version__}
    ck_rows = []

    def run_split(name, train_b, eval_w, seeds, fold_id, save_ckpt, extra_w=None):
        """Train all models on train_b; return probs on eval_w (and parts/embeddings on extra_w for outer)."""
        tr_w = np.flatnonzero(meta.bearing.isin(train_b).to_numpy())
        tr_m = np.flatnonzero(meas.bearing.isin(train_b).to_numpy())
        C4, M4, Q4, reg4 = v4.features(train_b)
        Z4, mu4, sd4 = V3.standardise(C4, M4, tr_m)
        Q4s = P3B.standardise_q(Q4, tr_m)
        C1, M1, Q1, reg1 = v1.features(train_b)
        Z1, _, _ = V3.standardise(C1, M1, tr_w)
        F2, reg2 = v2.features(frozenset(train_b))
        res = {"probs": {}, "info": {"regions_4s": reg4.to_dict(), "regions_1s": reg1.to_dict(), "regions_v2": reg2.to_dict()}}
        ev = eval_w
        # ---- classical
        tau = P3B.b0_fit(Q4[tr_m, 3], y_meas[tr_m])
        res["info"]["b0_tau_dB"] = tau
        res["probs"]["B0"] = P3B.b0_predict(C4, M4, tau)[w2m[ev]]
        F2tr, F2ev = P3A.impute(F2[tr_w], F2[ev])
        res["probs"]["B1-V2"], _ = lr_probs(F2tr, y_win[tr_w], F2ev)
        X1 = np.hstack([Z1, M1])
        res["probs"]["B1-V3-1s"], _ = lr_probs(X1[tr_w], y_win[tr_w], X1[ev])
        X4 = np.hstack([Z4, M4])
        p4, lr4 = lr_probs(X4[tr_m], y_meas[tr_m], X4)
        res["probs"]["B1-V3-4s"] = p4[w2m[ev]]
        res["lr4"] = {"coef": lr4[-1].coef_.tolist(), "intercept": lr4[-1].intercept_.tolist(),
                      "scaler_mean": lr4[0].mean_.tolist(), "scaler_scale": lr4[0].scale_.tolist()}
        res["phys4"] = {"C4": C4, "M4": M4, "Q4": Q4, "tau": tau, "mu4": mu4, "sd4": sd4}
        # ---- neural
        allw = np.unique(np.concatenate([tr_w, ev] + ([extra_w] if extra_w is not None else [])))
        D = {"raw": None, "y": y_win.copy(),
             "zm": np.zeros((len(meta), 16), np.float32), "q": np.zeros((len(meta), 7), np.float32)}
        D["zm"][allw] = np.hstack([Z4, M4])[w2m[allw]]
        D["q"][allw] = Q4s[w2m[allw]]
        Dm = {"raw": None, "y": y_win, "zm": np.zeros((len(meta), 16), np.float32), "q": D["q"]}
        Dm["zm"][win_bd5] = X1[win_bd5]
        Dr = dict(D)
        Dr["raw"] = raw_all
        rs = raw_stats(raw_all, tr_w)
        for mname, c in CFG.items():
            Dx = Dm if mname == "B2" else (Dr if mname in NEED_RAW else D)
            rsx = rs if mname in NEED_RAW else None
            ps = []
            for s in seeds:
                ep = 1 if a.smoke else c["epochs"]
                model, info = TR.fit(maker(mname), c["kind"], Dx, tr_w, ep, c["lr"], c["wd"], c["bs"], s, device, rs=rsx)
                p = TR.predict(model, c["kind"], Dx, ev, device, rs=rsx)
                ps.append(p)
                res["probs"][f"{mname}_s{s}"] = p
                info.update({"model": mname, "split": name, "fold": fold_id, "n_params": n_params(model),
                             "n_train_windows": int(len(tr_w)), "config": {k: v for k, v in c.items()}, **env_info})
                res["info"].setdefault("train", []).append(info)
                if save_ckpt:
                    cp = out / "checkpoints" / mname / fold_id / f"s{s}.pt"
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    if not cp.exists():
                        torch.save({"state_dict": model.state_dict(), "config": c, "raw_stats": rs if rsx is not None else None,
                                    "train_bearings": sorted(train_b)}, cp)
                    ck_rows.append({"model": mname, "fold": fold_id, "seed": s, "path": cp.relative_to(ROOT).as_posix(),
                                    "sha256": sha(cp), **{k: info[k] for k in ("train_seconds", "peak_gpu_mem_MB", "n_params")}})
                    if extra_w is not None and s == seeds[0]:
                        Pex, parts = TR.predict(model, c["kind"], Dx, extra_w, device, rs=rsx, parts=True)
                        res.setdefault("extra", {})[mname] = {"probs": Pex, **{k: v for k, v in parts.items() if k in ("phys", "latent", "gate", "c_hat", "logits")}}
                        if c["kind"] == "cnn":
                            with torch.no_grad():
                                H = []
                                for i in range(0, len(extra_w), 256):
                                    xb = torch.as_tensor((raw_all[extra_w[i:i + 256]].astype(np.float32) - rs[0]) / rs[1], device=device)
                                    H.append(model.head[:2](model.features(xb)).float().cpu().numpy())
                            res["extra"][mname]["h"] = np.concatenate(H)
                        elif "h" in parts:
                            res["extra"][mname]["h"] = parts["h"]
                del model
                torch.cuda.empty_cache()
            res["probs"][mname] = np.mean(ps, axis=0)
        return res

    folds = P3A.outer_folds()
    if a.final:
        folds = [{"fold": "FINAL", "train": frozenset(bd5), "test": frozenset(), "train_rows": list(range(5))}]
    for o in folds:
        if a.folds and o["fold"] not in a.folds:
            continue
        fz = out / "folds" / f"{o['fold']}.npz"
        if fz.exists():
            log(f"{o['fold']} exists - skipped")
            continue
        t0 = time.time()
        rec = {"fold": o["fold"], "train_bearings": sorted(o["train"]), "test_bearings": sorted(o["test"])}
        arrays = {}
        if not a.final:
            P3A.check_nested(o)
            oof = {}
            for inner in P3A.inner_folds(o):
                ev = np.flatnonzero(meta.bearing.isin(inner["val"]).to_numpy())
                r = run_split(inner["fold"], inner["train"], ev, [42], o["fold"], save_ckpt=False)
                for k, v in r["probs"].items():
                    if "_s" not in k:
                        oof.setdefault(k, []).append(v)
                oof.setdefault("_idx", []).append(ev)
                oof.setdefault("_kappa_in", []).append(np.stack([r["phys4"]["C4"][w2m[ev]].T.ravel()]))  # placeholder size check
                # physics features of inner-val windows (for S4 fitting)
                arrays.setdefault("oof_C4", []).append(r["phys4"]["C4"][w2m[ev]])
                arrays.setdefault("oof_M4", []).append(r["phys4"]["M4"][w2m[ev]])
                arrays.setdefault("oof_Q4", []).append(r["phys4"]["Q4"][w2m[ev]])
                arrays.setdefault("oof_tau", []).append(np.full(len(ev), r["phys4"]["tau"]))
                log(f"  inner {inner['fold']} done ({time.time()-t0:.0f}s)")
            for k, v in oof.items():
                if k.startswith("_kappa"):
                    continue
                arrays[f"oof_{k}"] = np.concatenate(v)
            for k in ("oof_C4", "oof_M4", "oof_Q4", "oof_tau"):
                arrays[k] = np.concatenate(arrays[k])
            arrays["oof_y"] = y_win[arrays["oof__idx"]]
            ev = np.flatnonzero(meta.bearing.isin(o["test"]).to_numpy())
        else:
            ev = np.flatnonzero(meta.bearing.isin(bd5).to_numpy())
        extra = np.flatnonzero(meta.bearing.isin(sorted(set(nonbd5) | set(o["test"]))).to_numpy())
        r = run_split(o["fold"], o["train"], ev, SEEDS, o["fold"], save_ckpt=True, extra_w=extra)
        for k, v in r["probs"].items():
            arrays[f"test_{k}"] = v
        arrays["test_idx"] = ev
        arrays["test_y"] = y_win[ev]
        arrays["extra_idx"] = extra
        for mname, d in r.get("extra", {}).items():
            for k, v in d.items():
                arrays[f"extra_{mname}_{k}"] = v
        for k in ("C4", "M4", "Q4"):
            arrays[f"meas_{k}"] = r["phys4"][k]
        rec.update({"b0_tau_dB": r["phys4"]["tau"], "regions": {k: r["info"][k] for k in ("regions_4s", "regions_1s", "regions_v2")},
                    "train": r["info"]["train"], "lr4": r["lr4"], "seconds": time.time() - t0, "smoke": a.smoke,
                    "protocol": "V2.2+V2.3"})
        np.savez_compressed(fz, **{k: np.asarray(v) for k, v in arrays.items()})
        (out / "folds" / f"{o['fold']}.json").write_text(json.dumps(rec, indent=1, default=float))
        pd.DataFrame(ck_rows).to_csv(out / "checkpoints.csv", mode="a", header=not (out / "checkpoints.csv").exists(), index=False)
        ck_rows.clear()
        log(f"{o['fold']} finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
