"""Phase 3A-E: real-data intervention study on PU-BD5.

S-6 (pre-specified, V2.1): manipulation check - do targeted envelope interventions remove the intended physics
evidence while controls do not, and how much distortion do they introduce?
EXPLORATORY: response of outer-fold models B1 (LR on concepts) and B3 (CNN, 3-seed ensemble) to targeted vs control
interventions. Bearing is the unit of analysis; windows are descriptive.
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
import torch  # noqa: E402
from scipy import signal  # noqa: E402

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments import phase2 as P2  # noqa: E402
from pave.experiments import phase3a as P3  # noqa: E402
from pave.explain import interventions as IV  # noqa: E402
from pave.features import physics as PH  # noqa: E402
from pave.models.cnn1d import CNN1D  # noqa: E402
from pave.training.engine import predict  # noqa: E402

WIN = ROOT / "results" / "phase2" / "windows"
VIB = ROOT / "results" / "phase2" / "cache_pu" / "vib"
BASE = ROOT / "results" / "phase3a" / "baselines"
OUT = ROOT / "results" / "phase3a" / "interventions"
FS = 64000.0
BAND = (1000.0, 20000.0)
N_MEAS = 5           # measurements per bearing x setting (window 0 of each)
STRENGTHS = (1.0, 0.5)


def zstats(raw, idx: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over the given (outer-train) windows, chunked to bound memory."""
    s = ss = 0.0
    n = 0
    for i in range(0, len(idx), chunk):
        x = np.asarray(raw[idx[i:i + chunk]], np.float64)
        s += x.sum(axis=(0, 2))
        ss += (x ** 2).sum(axis=(0, 2))
        n += x.shape[0] * x.shape[2]
    mu = s / n
    sd = np.sqrt(ss / n - mu ** 2) + 1e-8
    return mu.astype(np.float32)[None, :, None], sd.astype(np.float32)[None, :, None]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-bearings", nargs="*", default=None, help="debug: restrict to these bearings")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = P2.load_ds(WIN, "pu")
    meta = ds.meta
    kin_all = P2.kin_orders_per_window(ds, ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv")
    kin_b = {b: kin_all[np.flatnonzero((meta.bearing == b).to_numpy())[0]] for b in set().union(*P3.ROWS)}
    scan = pd.read_csv(WIN / "pu" / "physics_scan.csv")
    bd5_idx = np.flatnonzero(meta.bearing.isin(set().union(*P3.ROWS)).to_numpy())
    cache = P3.PhysicsCache(ds, kin_all, scan, bd5_idx)
    fgrid = np.arange(1001, dtype=float)
    rows = []
    rng = np.random.default_rng(2026)
    for o in P3.outer_folds():
        if args.only_bearings and not (set(args.only_bearings) & o["test"]):
            continue
        print(f"[{o['fold']}] preparing", flush=True)
        fold = json.loads((BASE / f"fold_{o['fold']}.json").read_text())
        if fold.get("smoke"):
            raise RuntimeError("smoke fold results present; run the full baseline first")
        reg = PH.Regions(**{k: fold["regions"][k] for k in ("centre", "half", "n_used", "fallback")})
        # B1 refit (deterministic) with the selected C on outer-train concepts
        F, _ = cache.features(o["train"])
        tr = bd5_idx[np.isin(meta.bearing.to_numpy()[bd5_idx], list(o["train"]))]
        Ftr, = P3.impute(F[tr])
        med = np.nanmedian(F[tr], axis=0)
        b1 = P3.lr_model(fold["selected"]["B1"]["config"][0]).fit(Ftr, cache.y[tr])
        # B3 models (3 seeds) + outer-train z-score stats
        nets = []
        for s in (42, 43, 44):
            p = BASE / f"B3_{o['fold']}_s{s}.pt"
            net = CNN1D(1, 3)
            net.load_state_dict(torch.load(p, map_location="cpu"))
            nets.append(net.to(device).eval())
        cm, cs = zstats(ds.raw, tr)

        def model_probs(xs: list[np.ndarray], fr: float, ko: dict) -> tuple[np.ndarray, np.ndarray]:
            C = []
            for x in xs:
                f, E = PH.envelope_spectrum_window(x, FS, BAND)
                d = PH.window_features(f[:1001], E[:1001], fr, ko, reg)
                C.append([d[k] for k in PH.FEATURE_NAMES])
            C = np.array(C, float)
            r, c = np.where(~np.isfinite(C))
            C[r, c] = med[c]
            p1 = P3.full_proba(b1, C)
            X = np.stack([signal.decimate(x, 2, ftype="fir", zero_phase=True)[None].astype(np.float32) for x in xs])
            X = (X - cm) / cs
            p3 = np.mean([predict(n, X, device) for n in nets], axis=0)
            return p1, p3

        import time as _t
        for b in sorted(o["test"]):
            if args.only_bearings and b not in args.only_bearings:
                continue
            _t0 = _t.time()
            lab = P3.LIDX[BEARINGS[b]["label3"]]
            kb = {"BPFO": kin_b[b][0], "BPFI": kin_b[b][1]}
            for setting in sorted(set(meta.setting)):
                fr = float(meta[(meta.bearing == b) & (meta.setting == setting)].fr_hz.iloc[0])
                guards = IV.shaft_guards(fr, 1000)
                fam = {"BPFO": IV.family_bands(fr, kb["BPFO"], reg.centre["BPFO"], reg.half["BPFO"]),
                       "BPFI": IV.family_bands(fr, kb["BPFI"], reg.centre["BPFI"], reg.half["BPFI"], sidebands=True)}
                for mnum in range(1, N_MEAS + 1):
                    p = VIB / b / f"{setting}_{b}_{mnum}.npy"
                    if not p.exists():
                        continue
                    x = np.load(p).astype(np.float64)[:64000]
                    variants = {"original": (x, {})}
                    dec = IV.decompose(x, FS, BAND)
                    for tgt in ("BPFO", "BPFI"):
                        other = "BPFI" if tgt == "BPFO" else "BPFO"
                        tb = IV.subtract_bands(fam[tgt], guards + fam[other])
                        ov = {"overlap_other_family_hz": IV.overlap_hz(fam[tgt], fam[other]),
                              "overlap_shaft_guard_hz": IV.overlap_hz(fam[tgt], guards),
                              "target_width_hz": sum(h - l for l, h in tb)}
                        forbidden = fam["BPFO"] + fam["BPFI"] + guards
                        hi = max(h for _, h in fam[tgt]) * 1.2
                        ctrl = IV.control_centres(tb, forbidden, 10.0, hi, rng)
                        for s in STRENGTHS:
                            rt = IV.suppress(x, FS, BAND, tb, s, fr, dec)
                            rw = IV.suppress(x, FS, BAND, ctrl, s, fr, dec)
                            re, ratio = IV.matched_energy_control(x, FS, BAND, rt.removed_env_energy, ctrl, forbidden, fr, s)
                            for name, r in ((f"T_{tgt}", rt), (f"Cw_{tgt}", rw), (f"Ce_{tgt}", re)):
                                variants[f"{name}|s{s}"] = (r.x, {"removed_env_energy": r.removed_env_energy, "env_energy": r.env_energy,
                                                                  "clip_fraction": r.clip_fraction, "oob_rel_change": r.out_of_band_rel_change,
                                                                  "in_band_snr_db": r.in_band_snr_db, "total_energy_ratio": r.total_energy_ratio,
                                                                  "energy_ratio_vs_target": ratio if name.startswith("Ce") else np.nan, **ov})
                    names = list(variants)
                    p1, p3 = model_probs([variants[n][0] for n in names], fr, kb)
                    base_c = None
                    for k, n in enumerate(names):
                        xv, info = variants[n]
                        f, E = PH.envelope_spectrum_window(xv, FS, BAND)
                        c = PH.window_features(f[:1001], E[:1001], fr, kb, reg)
                        if n == "original":
                            base_c = c
                        rows.append({"fold": o["fold"], "bearing": b, "label3": BEARINGS[b]["label3"], "setting": setting,
                                     "measurement": mnum, "variant": n.split("|")[0], "strength": float(n.split("|s")[1]) if "|s" in n else 0.0,
                                     **{f"c_{kk}": c[kk] for kk in ("BPFO_harm", "BPFI_harm", "contrast", "BPFI_sb", "BPFO_snr1", "BPFI_snr1")},
                                     **{f"d_{kk}": c[kk] - base_c[kk] for kk in ("BPFO_harm", "BPFI_harm", "contrast", "BPFI_sb")},
                                     "B1_p_true": p1[k, lab], "B3_p_true": p3[k, lab],
                                     "B1_pred": int(p1[k].argmax()), "B3_pred": int(p3[k].argmax()), **info})
                print(f"  {o['fold']} {b} {setting} done ({_t.time()-_t0:.0f}s)", flush=True)
            print(f"{o['fold']} {b} done", flush=True)
    D = pd.DataFrame(rows)
    base = D[D.variant == "original"].set_index(["bearing", "setting", "measurement"])[["B1_p_true", "B3_p_true"]]
    D = D.join(base, on=["bearing", "setting", "measurement"], rsuffix="_orig")
    D["dB1"] = D.B1_p_true - D.B1_p_true_orig
    D["dB3"] = D.B3_p_true - D.B3_p_true_orig
    D.to_csv(OUT / "intervention_windows.csv", index=False)
    # summaries (bearing = unit): own-family target vs controls
    D["own"] = np.where(D.label3 == "OR", "BPFO", np.where(D.label3 == "IR", "BPFI", ""))
    X = D[(D.variant != "original")].copy()
    X["target_fam"] = X.variant.str.split("_").str[1]
    X["kind"] = X.variant.str.split("_").str[0]
    X["relation"] = np.where(X.label3 == "H", "healthy", np.where(X.target_fam == X.own, "own_family", "other_family"))
    summ = X.groupby(["relation", "kind", "strength", "bearing"]).agg(
        d_own_harm=("d_BPFO_harm", "median"), d_BPFI_harm=("d_BPFI_harm", "median"), dB1=("dB1", "median"),
        dB3=("dB3", "median"), energy_ratio=("energy_ratio_vs_target", "median"), snr_db=("in_band_snr_db", "median"),
        clip=("clip_fraction", "max"), oob=("oob_rel_change", "max")).reset_index()
    summ.to_csv(OUT / "intervention_bearing_summary.csv", index=False)
    # manipulation check: own-family concept change, target vs controls (median over bearings)
    X["d_own"] = np.where(X.target_fam == "BPFO", X.d_BPFO_harm, X.d_BPFI_harm)
    X["d_other"] = np.where(X.target_fam == "BPFO", X.d_BPFI_harm, X.d_BPFO_harm)
    mc = X.groupby(["relation", "kind", "strength", "bearing"])[["d_own", "d_other", "dB1", "dB3"]].median().reset_index()
    agg = mc.groupby(["relation", "kind", "strength"]).agg(
        n_bearings=("bearing", "nunique"), d_target_family_harm_median=("d_own", "median"),
        d_other_family_harm_median=("d_other", "median"), dB1_median=("dB1", "median"), dB3_median=("dB3", "median")).round(3)
    agg.to_csv(OUT / "intervention_aggregate.csv")
    # targeted-minus-control per bearing (own family, strength 1)
    piv = mc[(mc.relation == "own_family") & (mc.strength == 1.0)].pivot(index="bearing", columns="kind", values=["d_own", "dB1", "dB3"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    for m in ("dB1", "dB3", "d_own"):
        piv[f"{m}_T_minus_Ce"] = piv[f"{m}_T"] - piv[f"{m}_Ce"]
        piv[f"{m}_T_minus_Cw"] = piv[f"{m}_T"] - piv[f"{m}_Cw"]
    piv.round(3).to_csv(OUT / "intervention_targeted_minus_control_per_bearing.csv")
    dist = X.groupby(["kind", "strength"]).agg(in_band_snr_db_median=("in_band_snr_db", "median"),
                                               clip_fraction_max=("clip_fraction", "max"), oob_rel_change_max=("oob_rel_change", "max"),
                                               total_energy_ratio_median=("total_energy_ratio", "median"),
                                               energy_ratio_vs_target_median=("energy_ratio_vs_target", "median"),
                                               energy_ratio_vs_target_p10=("energy_ratio_vs_target", lambda s: s.quantile(0.1)),
                                               overlap_shaft_guard_hz_max=("overlap_shaft_guard_hz", "max"),
                                               overlap_other_family_hz_max=("overlap_other_family_hz", "max")).round(4)
    dist.to_csv(OUT / "intervention_distortion.csv")
    by_setting = X[(X.relation == "own_family") & (X.strength == 1.0)].groupby(["setting", "kind"])[["d_own", "dB1", "dB3"]].median().round(3)
    by_setting.to_csv(OUT / "intervention_by_setting.csv")
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    colors = {"T": "#d03b3b", "Cw": "#b5b4ad", "Ce": "#52514e"}
    for ax, col, lab in zip(axes, ("d_own", "dB1", "dB3"), ("Δ own-family evidence [dB]", "Δ p(true) B1", "Δ p(true) B3")):
        sub = mc[(mc.relation == "own_family") & (mc.strength == 1.0)]
        for j, k in enumerate(("T", "Cw", "Ce")):
            v = sub[sub.kind == k][col]
            ax.scatter(np.full(len(v), j) + np.linspace(-0.2, 0.2, len(v)), v, s=14, color=colors[k], zorder=3)
            ax.plot([j - 0.3, j + 0.3], [v.median()] * 2, color="#0b0b0b", lw=1)
        ax.set_xticks([0, 1, 2], ["targeted", "ctrl width", "ctrl energy"], fontsize=7)
        ax.set_title(lab, fontsize=8)
        ax.axhline(0, color="#52514e", lw=0.5)
    fig.suptitle("PU-BD5 faulty bearings (dots = bearing medians), strength 1.0 — exploratory model response", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_interventions.png", dpi=300)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(agg.to_string())
    print(dist.to_string())
    print(piv.round(3).to_string())
    print(by_setting.to_string())


if __name__ == "__main__":
    main()
