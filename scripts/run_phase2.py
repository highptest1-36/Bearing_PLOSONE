"""Run Phase 2 bounded baselines (protocol V2 §7).

  python -I scripts/run_phase2.py --dataset kaist --smoke          # 1 fold, 2 epochs, 1 seed
  python -I scripts/run_phase2.py --dataset kaist                  # full KAIST (3-way + exploratory 7-way)
  python -I scripts/run_phase2.py --dataset pu --protocol PU_BD5   # bearing-disjoint 5 folds
  python -I scripts/run_phase2.py --dataset pu --protocol PU_A2R

Results: results/phase2/runs/<ds>/<protocol>/runs.jsonl (+ predictions .npz). Re-running skips finished runs.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pave.evaluation.metrics import evaluate  # noqa: E402
from pave.experiments import phase2 as X  # noqa: E402
from pave.features import physics as PH  # noqa: E402
from pave.models.cnn1d import CNN1D, n_params  # noqa: E402
from pave.training.engine import TrainConfig, fit, predict  # noqa: E402

WIN = ROOT / "results" / "phase2" / "windows"
RUNS = ROOT / "results" / "phase2" / "runs"
PU_BEAR = ROOT / "03_MANIFESTS" / "paderborn" / "pu_bearings.csv"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["run_key"] for l in path.read_text().splitlines() if l.strip()}


def labels(meta, task: str) -> np.ndarray:
    names = X.L3 if task == "3way" else X.L7
    col = "label3" if task == "3way" else "label7"
    lut = {n: i for i, n in enumerate(names)}
    return np.array([lut.get(v, -1) if isinstance(v, str) else -1 for v in meta[col]])


def groups(ds, meta) -> dict:
    if ds.name == "kaist":
        return {"recording": meta.recording_id.to_numpy()}
    return {"bearing": meta.bearing.to_numpy(), "bearing_setting": (meta.bearing + "|" + meta.setting).to_numpy()}


def write(path: Path, rec: dict, pred: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    if pred is not None:
        pdir = path.parent / "preds"
        pdir.mkdir(exist_ok=True)
        np.savez_compressed(pdir / f"{rec['run_key'].replace('|', '__')}.npz", **pred)


def challenge_summary(probs: np.ndarray, meta) -> dict:
    pred = probs.argmax(1)
    out = {}
    for b in sorted(set(meta.bearing)):
        m = (meta.bearing == b).to_numpy()
        out[b] = {X.L3[k]: float(np.mean(pred[m] == k)) for k in range(3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["kaist", "pu"])
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = X.load_ds(WIN, a.dataset)
    protocol = a.protocol or ("A_LOLO" if a.dataset == "kaist" else "PU_BD5")
    out = (RUNS / "_smoke" if a.smoke else RUNS) / a.dataset / protocol / "runs.jsonl"
    keys = done_keys(out)
    kin = X.kin_orders_per_window(ds, PU_BEAR if a.dataset == "pu" else None)
    scan_file = WIN / a.dataset / "physics_scan.csv"
    if scan_file.exists():
        import pandas as pd
        scan = pd.read_csv(scan_file)
    else:
        log("physics scan (label-free)...")
        scan = X.physics_scan(ds, kin)
        scan.to_csv(scan_file, index=False)
    folds = X.kaist_folds(ds, ROOT / "03_MANIFESTS" / "splits") if a.dataset == "kaist" else X.pu_folds(ds, protocol)
    if a.smoke:
        folds = folds[:1]
        a.seeds = a.seeds[:1]
    tasks = ["3way", "7way"] if a.dataset == "kaist" else ["3way"]
    cfg = TrainConfig(max_epochs=2 if a.smoke else 25)
    env_info = {"python": platform.python_version(), "torch": torch.__version__, "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    for fold in folds:
        fid = fold["fold"]
        R = fold["raw"]
        y3_all = labels(ds.meta, "3way")
        reg = PH.calibrate_regions({f: scan[f"{f}_scan_r"].to_numpy()[R["train"]] for f in PH.FAMILIES},
                                   {f: scan[f"{f}_scan_snr"].to_numpy()[R["train"]] for f in PH.FAMILIES},
                                   np.array(X.L3 + ["?"])[np.where(y3_all[R["train"]] >= 0, y3_all[R["train"]], 3)])
        phys = X.physics_features(ds, kin, reg)
        diag = {"run_key": f"{fid}|physics_diag", "fold": fid, "protocol": protocol, "regions": reg.to_dict(),
                "auroc_test": X.physics_auroc(phys[R["test"]], y3_all[R["test"]]),
                "auroc_train": X.physics_auroc(phys[R["train"]], y3_all[R["train"]]),
                "n_windows": {k: int(len(v)) for k, v in R.items()}, "bearings": fold.get("bearings"),
                "split_sha256": fold.get("split_sha256")}
        if diag["run_key"] not in keys:
            write(out, diag, {"phys": phys.astype(np.float32), "idx": np.arange(len(phys))})
            log(f"{fid} physics: regions {reg.to_dict()} | test AUROC {diag['auroc_test']}")
        for task in tasks:
            y_all = labels(ds.meta, task)
            ncls = 3 if task == "3way" else 7
            feats = {"S1_speed_LR": (ds.short[:, :1], "LR"), "S2_rms_LR": (ds.short[:, 1:], "LR"),
                     "C1_hand_RF": (ds.hand, "RF"), "C2_hand_SVM": (ds.hand, "SVM"), "P1_phys_LR": (phys, "LR")}
            deep = {"D1_cnn_raw": "raw", "D2_cnn_order": "order", "D3_cnn_order_rmsnorm": "order_rms"}
            allowed = (list(feats) + list(deep)) if task == "3way" else ["C1_hand_RF", "P1_phys_LR", "D1_cnn_raw", "D2_cnn_order"]
            if a.models:
                allowed = [m for m in allowed if m in a.models]
            tr, va, te = R["train"], R["val"], R["test"]
            for mname in allowed:
                if mname in feats:
                    key = f"{fid}|{task}|{mname}|s42"
                    if key in keys:
                        continue
                    t0 = time.time()
                    Xf, kind = feats[mname]
                    Xf = X.impute_train_median(np.asarray(Xf, float), tr)
                    mdl = X.classical(kind)
                    mdl.fit(Xf[tr], y_all[tr])
                    P = {s: X.full_probs(mdl, Xf[ix], ncls) for s, ix in (("train", tr), ("val", va), ("test", te))}
                    rec = {"run_key": key, "dataset": a.dataset, "protocol": protocol, "fold": fid, "task": task, "model": mname,
                           "seed": 42, "fit_seconds": time.time() - t0,
                           "train": evaluate(P["train"], y_all[tr], {}, ncls)["window"],
                           "val": evaluate(P["val"], y_all[va], groups(ds, ds.meta.iloc[va]), ncls),
                           "test": evaluate(P["test"], y_all[te], groups(ds, ds.meta.iloc[te]), ncls), "env": env_info}
                    if len(fold["challenge"]["raw"]):
                        rec["challenge"] = challenge_summary(X.full_probs(mdl, Xf[fold["challenge"]["raw"]], ncls),
                                                             ds.meta.iloc[fold["challenge"]["raw"]])
                    write(out, rec, {"probs": P["test"].astype(np.float32), "y": y_all[te], "idx": te})
                    keys.add(key)
                    log(f"{key}: test win macroF1 {rec['test']['window']['macro_f1']:.3f} "
                        f"group acc {[round(v['acc'], 3) for k, v in rec['test'].items() if k != 'window']}")
                else:
                    rep = deep[mname]
                    if rep == "raw":
                        arr, mm, I = ds.raw, ds.meta, R
                        ych = labels(ds.meta, task)
                        chal = fold["challenge"]["raw"]
                    else:
                        arr, mm, I = ds.order, ds.ometa, fold["order"]
                        if a.dataset == "kaist":
                            mm = ds.meta
                        ych = labels(mm, task) if "label7" in mm or task == "3way" else None
                        chal = fold["challenge"]["order"]
                    Xtr = np.asarray(arr[I["train"]], np.float32)
                    Xva = np.asarray(arr[I["val"]], np.float32)
                    Xte = np.asarray(arr[I["test"]], np.float32)
                    Xch = np.asarray(arr[chal], np.float32) if len(chal) else None
                    if rep == "order_rms":
                        def tf(b):
                            return b / (b.std(axis=2, keepdims=True) + 1e-8)
                        for Z in (Xtr, Xva, Xte) + ((Xch,) if Xch is not None else ()):
                            Z[:] = tf(Z)
                    else:
                        mu = Xtr.mean(axis=(0, 2), keepdims=True)
                        sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
                        for Z in (Xtr, Xva, Xte) + ((Xch,) if Xch is not None else ()):
                            Z -= mu
                            Z /= sd
                    ytr, yva, yte = ych[I["train"]], ych[I["val"]], ych[I["test"]]
                    for seed in a.seeds:
                        key = f"{fid}|{task}|{mname}|s{seed}"
                        if key in keys:
                            continue
                        model = CNN1D(Xtr.shape[1], ncls)
                        info = fit(model, Xtr, ytr, Xva, yva, ncls, cfg, seed, device)
                        P = {"val": predict(model, Xva, device), "test": predict(model, Xte, device)}
                        rec = {"run_key": key, "dataset": a.dataset, "protocol": protocol, "fold": fid, "task": task,
                               "model": mname, "seed": seed, "n_params": n_params(model), "input_shape": list(Xtr.shape[1:]),
                               **{k: v for k, v in info.items()},
                               "val": evaluate(P["val"], yva, groups(ds, mm.iloc[I["val"]]), ncls),
                               "test": evaluate(P["test"], yte, groups(ds, mm.iloc[I["test"]]), ncls), "env": env_info}
                        if Xch is not None:
                            rec["challenge"] = challenge_summary(predict(model, Xch, device), mm.iloc[chal])
                        write(out, rec, {"probs": P["test"].astype(np.float32), "y": yte, "idx": I["test"]})
                        keys.add(key)
                        log(f"{key}: ep {info['best_epoch']} val {info['best_val_macro_f1']:.3f} test win macroF1 "
                            f"{rec['test']['window']['macro_f1']:.3f} group acc "
                            f"{[round(v['acc'], 3) for k, v in rec['test'].items() if k != 'window']} ({info['train_seconds']:.0f}s)")
                    del Xtr, Xva, Xte, Xch
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
