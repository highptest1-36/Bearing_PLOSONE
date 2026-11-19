"""Pre-specified secondary analysis of EV-V1 §8: physics-evidence verification (PEV) on the 10 faulty IR/OR Ottawa recordings
for B1-V3-4s, A4 and B3 (frozen FINAL PU models; same intervention code and settings as Phase 3B PEV, strength 1.0 and 0.5).

Executed after the primary single evaluation (EVALUATED.lock) as a separate script; it trains nothing, tunes nothing and
cannot change the primary result. Labels are used only to read p(true class), exactly as in the PU PEV.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import signal  # noqa: E402

from pave.data import ottawa as OT  # noqa: E402
from pave.experiments import phase3a as P3A  # noqa: E402
from pave.experiments import phase3b as P3B  # noqa: E402
from pave.explain import interventions as IV  # noqa: E402
from pave.features import physics_v3 as V3  # noqa: E402
from pave.features.physics import Regions  # noqa: E402
from pave.models.cnn1d import CNN1D  # noqa: E402
from pave.models.pavenet import PAVENet  # noqa: E402
from pave.signal.envelope import envelope  # noqa: E402
from pave.signal.physics import BearingGeometry, orders  # noqa: E402

FS = OT.FS
N4 = int(4 * FS)
BAND = (1000.0, 20000.0)
OUT = ROOT / "results/phase4/pev"
F3B = ROOT / "results/phase3b"
RAW = ROOT / "01_RAW_DATA" / "Ottawa_data"
KIN = {k: v for k, v in orders(BearingGeometry("Ottawa 6203", 8, 6.77, 28.50, 0.0)).items() if k in ("BPFO", "BPFI")}
STRENGTHS = (1.0, 0.5)


def env4(x):
    e = envelope(x, FS, BAND)
    w = np.hanning(len(e))
    E = np.abs(np.fft.rfft(e * w, N4)) * 2.0 / w.sum()
    return np.arange(4001) * 0.25, E[:4001]


def main():
    assert (ROOT / "results/phase4/EVALUATED.lock").exists(), "primary evaluation must precede this secondary analysis"
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fin = json.loads((F3B / "folds/FINAL.json").read_text())
    z = np.load(F3B / "folds/FINAL.npz")
    meas = pd.read_csv(F3B / "env4s/meas.csv")
    tr_m = np.flatnonzero(meas.bearing.isin(sorted(fin["train_bearings"])).to_numpy())
    C4, M4, Q4 = z["meas_C4"], z["meas_M4"], z["meas_Q4"]
    Z4, mu4, sd4 = V3.standardise(C4, M4, tr_m)
    qmu, qsd = Q4[tr_m].mean(0), np.where(Q4[tr_m].std(0) > 1e-8, Q4[tr_m].std(0), 1.0)
    y_m = np.array([P3B.LIDX.get(v, -1) if isinstance(v, str) else -1 for v in meas.label3.fillna("")])
    lr4 = P3A.lr_model(1.0).fit(np.hstack([Z4, M4])[tr_m], y_m[tr_m])
    reg = Regions(**fin["regions"]["regions_4s"])
    models = {}
    for name, mk in (("A4", lambda: PAVENet(16, 7, 3, mode="gated")), ("B3", lambda: CNN1D(1, 3))):
        ms = []
        for s in (42, 43, 44):
            ck = torch.load(F3B / "checkpoints" / name / "FINAL" / f"s{s}.pt", map_location="cpu", weights_only=False)
            m = mk()
            m.load_state_dict(ck["state_dict"])
            ms.append((m.to(device).eval(), ck["raw_stats"]))
        models[name] = ms

    def evaluate(xs, fr, sig):
        Z, Mm, Qs, Cs = [], [], [], []
        for x in xs:
            f, E = env4(x)
            r = V3.features_v3(f, E, fr, KIN, reg, sig)
            Cs.append(r.c)
            Mm.append(r.m)
            Z.append(r.m * (r.c - mu4) / sd4)
            Qs.append((r.q - qmu) / qsd)
        Z, Mm, Qs, Cs = map(np.array, (Z, Mm, Qs, Cs))
        zm = np.hstack([Z, Mm]).astype(np.float32)
        out = {"concepts": Cs, "masks": Mm, "B1-V3-4s": P3A.full_proba(lr4, np.hstack([Z, Mm]))}
        n1 = int(FS)
        W = np.stack([signal.resample_poly(x[k * n1:(k + 1) * n1], 16, 21) for x in xs for k in range(4)])[:, None, :].astype(np.float32)
        zmw = torch.as_tensor(np.repeat(zm, 4, 0), device=device)
        qw = torch.as_tensor(np.repeat(Qs.astype(np.float32), 4, 0), device=device)
        with torch.no_grad():
            for name, ms in models.items():
                P, pp, pl = [], [], []
                for m, rs in ms:
                    xr = torch.as_tensor((W - rs[0]) / rs[1], device=device)
                    if name == "B3":
                        lg = m(xr)
                    else:
                        lg, pr = m(zmw, qw, xr, return_parts=True)
                        pp.append(pr["phys"].float().cpu().numpy())
                        pl.append(((1 - pr["gate"]) * pr["latent"]).float().cpu().numpy())
                    P.append(torch.softmax(lg.float(), 1).cpu().numpy())
                out[name] = np.mean(P, 0).reshape(len(xs), 4, 3).mean(1)
                if pp:
                    out[f"{name}_phys_logits"] = np.mean(pp, 0).reshape(len(xs), 4, 3).mean(1)
                    out[f"{name}_latent_contrib"] = np.mean(pl, 0).reshape(len(xs), 4, 3).mean(1)
        return out

    rng = np.random.default_rng(2026)
    F = np.load(ROOT / "results/phase4/ottawa_features.npz")
    seg_tab = pd.read_csv(ROOT / "results/phase4/ottawa_segments4s.csv")
    rows, t0 = [], time.time()
    for p in OT.find_files(RAW):
        info = OT.parse_stem(p.stem)
        lm = OT.label_mapping(info)
        if lm["role"] != "primary" or lm["primary"] == "H":
            continue
        lab = P3B.LIDX[lm["primary"]]
        a = OT.read_ottawa(p)
        x_all = a[:, 0].astype(np.float64)
        fr, sig = float(a[0, 2]) / 60.0, 0.5 / 60.0
        guards = IV.shaft_guards(fr, 1000)
        fam = {Fm: IV.family_bands(fr, KIN[Fm], reg.centre[Fm], reg.half[Fm], sidebands=(Fm == "BPFI"), min_half_hz=0.25) for Fm in ("BPFO", "BPFI")}
        for sg in range(2):
            x = x_all[sg * N4:(sg + 1) * N4]
            dec = IV.decompose(x, FS, BAND)
            variants = {"original|0": (x, {})}
            for tgt in ("BPFO", "BPFI"):
                other = "BPFI" if tgt == "BPFO" else "BPFO"
                tb = IV.subtract_bands(fam[tgt], guards + fam[other])
                forbidden = fam["BPFO"] + fam["BPFI"] + guards
                ctrl = IV.control_centres(tb, forbidden, 10.0, max(h for _, h in fam[tgt]) * 1.2, rng)
                for s in STRENGTHS:
                    rt = IV.suppress(x, FS, BAND, tb, s, fr, dec)
                    rw = IV.suppress(x, FS, BAND, ctrl, s, fr, dec)
                    re, ratio = IV.matched_energy_control(x, FS, BAND, rt.removed_env_energy, ctrl, forbidden, fr, s)
                    for kind, res in (("T", rt), ("Cw", rw), ("Ce", re)):
                        variants[f"{kind}_{tgt}|{s}"] = (res.x, {"in_band_snr_db": res.in_band_snr_db, "clip_fraction": res.clip_fraction,
                                                                 "oob_rel_change": res.out_of_band_rel_change,
                                                                 "energy_ratio_vs_target": ratio if kind == "Ce" else np.nan})
            names = list(variants)
            ev = evaluate([variants[n][0] for n in names], fr, sig)
            # consistency check: the original variant reproduces the frozen Ottawa features
            k0 = int(np.flatnonzero((seg_tab.stem == info["stem"]).to_numpy() & (seg_tab.segment4 == sg).to_numpy())[0])
            assert np.allclose(np.nan_to_num(ev["concepts"][0]), np.nan_to_num(F["C4"][k0]), atol=1e-6), info["stem"]
            for k, n in enumerate(names):
                row = {"stem": info["stem"], "bearing": info["bearing"], "label3": lm["primary"], "segment4": sg,
                       "variant": n.split("|")[0], "strength": float(n.split("|")[1]), **variants[n][1],
                       **{f"c_{c}": ev["concepts"][k, j] for j, c in enumerate(V3.CONCEPTS_V3)},
                       **{f"m_{c}": ev["masks"][k, j] for j, c in enumerate(V3.CONCEPTS_V3)}}
                for mname in ("B1-V3-4s", "A4", "B3"):
                    pv = ev[mname][k]
                    row[f"{mname}_p_true"] = pv[lab]
                    row[f"{mname}_pred"] = int(pv.argmax())
                for part in ("A4_phys_logits", "A4_latent_contrib"):
                    row[f"{part}_true"] = ev[part][k, lab]
                rows.append(row)
        print(f"{info['stem']} done ({time.time()-t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "pev_ottawa_measurements.csv", index=False)
    # summary: own family = BPFO for OR, BPFI for IR; strength 1.0; recording-level means then median over 10 recordings
    D["own"] = np.where(D.label3 == "OR", "BPFO", "BPFI")
    D["evidence"] = np.where(D.own == "BPFO", D.c_BPFO_harm, D.c_BPFI_harm)
    base = D[D.variant == "original"].groupby("stem").mean(numeric_only=True)
    out = []
    for kind in ("T", "Cw", "Ce"):
        sub = D[(D.strength == 1.0) & (D.variant == D.own.radd(kind + "_"))].groupby("stem").mean(numeric_only=True)
        dlt = sub - base.loc[sub.index]
        flips = D[(D.strength == 1.0) & (D.variant == D.own.radd(kind + "_"))]
        out.append({"variant": kind, "n_recordings": len(sub), "evidence_dB_median_change": float(dlt.evidence.median()),
                    **{f"{m}_dp_true_median": float(dlt[f"{m}_p_true"].median()) for m in ("B1-V3-4s", "A4", "B3")},
                    **{f"{m}_pred_change_rate": float(np.mean(flips[f"{m}_pred"].to_numpy() !=
                                                              D[D.variant == "original"].set_index(["stem", "segment4"]).loc[
                                                                  list(zip(flips.stem, flips.segment4)), f"{m}_pred"].to_numpy()))
                       for m in ("B1-V3-4s", "A4", "B3")},
                    "A4_phys_logit_median_change": float(dlt.A4_phys_logits_true.median()),
                    "A4_latent_contrib_median_change": float(dlt.A4_latent_contrib_true.median()),
                    "in_band_snr_db_median": float(sub.in_band_snr_db.median()) if kind != "orig" else np.nan,
                    "clip_fraction_max": float(sub.clip_fraction.max()), "oob_rel_change_max": float(sub.oob_rel_change.max())})
    S = pd.DataFrame(out)
    # paired targeted - energy-matched control, per recording, with bootstrap CI over the 10 bearings
    t = D[(D.strength == 1.0) & (D.variant == D.own.radd("T_"))].groupby("stem").mean(numeric_only=True)
    c = D[(D.strength == 1.0) & (D.variant == D.own.radd("Ce_"))].groupby("stem").mean(numeric_only=True).loc[t.index]
    rng2 = np.random.default_rng(7)
    diff = []
    for col, lab in (("evidence", "evidence_dB"), ("B1-V3-4s_p_true", "B1-V3-4s"), ("A4_p_true", "A4"), ("B3_p_true", "B3"),
                     ("A4_phys_logits_true", "A4_phys_logit"), ("A4_latent_contrib_true", "A4_latent_contrib")):
        d = (t[col] - c[col]).to_numpy()
        bs = [np.median(rng2.choice(d, len(d))) for _ in range(10000)]
        diff.append({"quantity": lab, "median_T_minus_Ce": float(np.median(d)), "ci95_lo": float(np.quantile(bs, .025)),
                     "ci95_hi": float(np.quantile(bs, .975)), "n_negative": int((d < 0).sum()), "n": len(d)})
    Dd = pd.DataFrame(diff)
    S.round(4).to_csv(OUT / "pev_ottawa_summary.csv", index=False)
    Dd.round(4).to_csv(OUT / "pev_ottawa_targeted_minus_control.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string())
    print(Dd.round(3).to_string())


if __name__ == "__main__":
    main()
