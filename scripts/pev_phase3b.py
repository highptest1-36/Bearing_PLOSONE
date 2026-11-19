"""Phase 3B physics-evidence verification (protocol V2.2 §7 + V2.3 §3).

For each outer fold, the 3 outer-test bearings: 5 measurements x 4 settings (first 4 s). Variants: original;
targeted / matched-width / matched-energy suppression of each family (BPFO; BPFI incl. +-fr sidebands) at strengths
1.0 and 0.5. Physics models recompute 4-s V3 concepts; CNN-based models average the four 1-s windows.
Models: B1-V3-4s (LR), A2, A4 (total + physics/latent logit parts), A6, B3 (3-seed ensembles).
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

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments import phase3a as P3A  # noqa: E402
from pave.experiments import phase3b as P3B  # noqa: E402
from pave.explain import interventions as IV  # noqa: E402
from pave.features import physics_v3 as V3  # noqa: E402
from pave.features.physics import Regions  # noqa: E402
from pave.models.cnn1d import CNN1D  # noqa: E402
from pave.models.pavenet import ConceptBottleneck, PAVENet  # noqa: E402

FS = 64000.0
N4 = 256000
BAND = (1000.0, 20000.0)
OUT = ROOT / "results" / "phase3b" / "pev"
CK = ROOT / "results" / "phase3b" / "checkpoints"
N_MEAS = 5
STRENGTHS = (1.0, 0.5)


def env4(x):
    from pave.signal.envelope import envelope
    e = envelope(x, FS, BAND)
    w = np.hanning(len(e))
    E = np.abs(np.fft.rfft(e * w, N4)) * 2.0 / w.sum()
    return np.arange(4001) * 0.25, E[:4001]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meas = pd.read_csv(ROOT / "results/phase3b/env4s/meas.csv")
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    from pave.signal.physics import BearingGeometry, orders
    kin_b = {c: orders(BearingGeometry(c, int(r.geom_n_balls), r.geom_ball_diameter_mm, r.geom_pitch_diameter_mm, 0.0))
             for c, r in bear.iterrows()}
    rng = np.random.default_rng(2026)
    rows = []
    t_start = time.time()
    for o in P3A.outer_folds():
        js = json.loads((ROOT / f"results/phase3b/folds/{o['fold']}.json").read_text())
        z = np.load(ROOT / f"results/phase3b/folds/{o['fold']}.npz")
        reg = Regions(**js["regions"]["regions_4s"])
        tau = js["b0_tau_dB"]
        C4, M4, Q4 = z["meas_C4"], z["meas_M4"], z["meas_Q4"]
        tr_m = np.flatnonzero(meas.bearing.isin(o["train"]).to_numpy())
        _, mu4, sd4 = V3.standardise(C4, M4, tr_m)
        qmu, qsd = Q4[tr_m].mean(0), np.where(Q4[tr_m].std(0) > 1e-8, Q4[tr_m].std(0), 1.0)
        y_m = np.array([P3B.LIDX.get(v, -1) if isinstance(v, str) else -1 for v in meas.label3.fillna("")])
        lr4 = P3A.lr_model(1.0).fit(np.hstack([V3.standardise(C4, M4, tr_m)[0], M4])[tr_m], y_m[tr_m])
        models = {}
        for name, mk in (("A2", lambda: PAVENet(16, 7, 3, mode="physics", head_q=True)), ("A4", lambda: PAVENet(16, 7, 3, mode="gated")),
                         ("A6", lambda: ConceptBottleneck(8, 3, 1)), ("B3", lambda: CNN1D(1, 3))):
            ms = []
            for s in (42, 43, 44):
                ck = torch.load(CK / name / o["fold"] / f"s{s}.pt", map_location="cpu", weights_only=False)
                m = mk()
                m.load_state_dict(ck["state_dict"])
                ms.append((m.to(device).eval(), ck["raw_stats"]))
            models[name] = ms

        def evaluate(xs, fr, kb, sig):
            """xs: list of 4-s signals -> dict model -> (probs (n,3), extra)"""
            Z, Mm, Qs, Cs = [], [], [], []
            for x in xs:
                f, E = env4(x)
                r = V3.features_v3(f, E, fr, kb, reg, sig)
                Cs.append(r.c)
                Mm.append(r.m)
                Z.append(r.m * (r.c - mu4) / sd4)
                Qs.append((r.q - qmu) / qsd)
            Z, Mm, Qs, Cs = map(np.array, (Z, Mm, Qs, Cs))
            zm = np.hstack([Z, Mm]).astype(np.float32)
            out = {"concepts": Cs, "masks": Mm, "B1-V3-4s": P3A.full_proba(lr4, np.hstack([Z, Mm]))}
            # 1-s windows (4 per signal), decimated to 32 kHz
            W = np.stack([signal.decimate(x[k * 64000:(k + 1) * 64000], 2, ftype="fir", zero_phase=True) for x in xs for k in range(4)])
            W = W[:, None, :].astype(np.float32)
            zmw = torch.as_tensor(np.repeat(zm, 4, 0), device=device)
            qw = torch.as_tensor(np.repeat(Qs.astype(np.float32), 4, 0), device=device)
            with torch.no_grad():
                for name, ms in models.items():
                    P, parts_p, parts_l = [], [], []
                    for m, rs in ms:
                        xr = torch.as_tensor((W - rs[0]) / rs[1], device=device) if rs is not None else None
                        if name == "B3":
                            lg = m(xr)
                        elif name == "A6":
                            lg = m(xr)
                        else:
                            lg, pr = m(zmw, qw, xr, return_parts=True)
                            parts_p.append(pr["phys"].float().cpu().numpy())
                            parts_l.append(((1 - pr["gate"]) * pr["latent"]).float().cpu().numpy())
                        P.append(torch.softmax(lg.float(), 1).cpu().numpy())
                    p = np.mean(P, 0).reshape(len(xs), 4, 3).mean(1)
                    out[name] = p
                    if parts_p:
                        out[f"{name}_phys_logits"] = np.mean(parts_p, 0).reshape(len(xs), 4, 3).mean(1)
                        out[f"{name}_latent_contrib"] = np.mean(parts_l, 0).reshape(len(xs), 4, 3).mean(1)
            return out

        for b in sorted(o["test"]):
            lab = P3B.LIDX[BEARINGS[b]["label3"]]
            kb = {"BPFO": kin_b[b]["BPFO"], "BPFI": kin_b[b]["BPFI"]}
            for setting in sorted(meas.setting.unique()):
                mm = meas[(meas.bearing == b) & (meas.setting == setting)].sort_values("measurement").head(N_MEAS)
                for _, r in mm.iterrows():
                    x = np.load(ROOT / "results/phase2/cache_pu/vib" / b / f"{setting}_{b}_{r.measurement}.npy").astype(np.float64)[:N4]
                    if len(x) < N4:
                        x = np.pad(x, (0, N4 - len(x)))
                    fr, sig = r.fr_hz, r.sigma_fr_hz
                    guards = IV.shaft_guards(fr, 1000)
                    fam = {F: IV.family_bands(fr, kb[F], reg.centre[F], reg.half[F], sidebands=(F == "BPFI"), min_half_hz=0.25)
                           for F in ("BPFO", "BPFI")}
                    dec = IV.decompose(x, FS, BAND)
                    variants = {"original|0": (x, {})}
                    for tgt in ("BPFO", "BPFI"):
                        other = "BPFI" if tgt == "BPFO" else "BPFO"
                        tb = IV.subtract_bands(fam[tgt], guards + fam[other])
                        forbidden = fam["BPFO"] + fam["BPFI"] + guards
                        ctrl = IV.control_centres(tb, forbidden, 10.0, max(h for _, h in fam[tgt]) * 1.2, rng)
                        ov = {"overlap_other_family_hz": IV.overlap_hz(fam[tgt], fam[other]), "overlap_shaft_guard_hz": IV.overlap_hz(fam[tgt], guards),
                              "target_width_hz": sum(h - l for l, h in tb)}
                        for s in STRENGTHS:
                            rt = IV.suppress(x, FS, BAND, tb, s, fr, dec)
                            rw = IV.suppress(x, FS, BAND, ctrl, s, fr, dec)
                            re, ratio = IV.matched_energy_control(x, FS, BAND, rt.removed_env_energy, ctrl, forbidden, fr, s)
                            for kind, res in (("T", rt), ("Cw", rw), ("Ce", re)):
                                variants[f"{kind}_{tgt}|{s}"] = (res.x, {"in_band_snr_db": res.in_band_snr_db, "clip_fraction": res.clip_fraction,
                                                                         "oob_rel_change": res.out_of_band_rel_change, "total_energy_ratio": res.total_energy_ratio,
                                                                         "removed_env_energy": res.removed_env_energy,
                                                                         "energy_ratio_vs_target": ratio if kind == "Ce" else np.nan, **ov})
                    names = list(variants)
                    ev = evaluate([variants[n][0] for n in names], fr, kb, sig)
                    for k, n in enumerate(names):
                        row = {"fold": o["fold"], "bearing": b, "label3": BEARINGS[b]["label3"], "setting": setting, "measurement": int(r.measurement),
                               "variant": n.split("|")[0], "strength": float(n.split("|")[1]), **variants[n][1],
                               **{f"c_{c}": ev["concepts"][k, j] for j, c in enumerate(V3.CONCEPTS_V3)},
                               **{f"m_{c}": ev["masks"][k, j] for j, c in enumerate(V3.CONCEPTS_V3)}}
                        for mname in ("B1-V3-4s", "A2", "A4", "A6", "B3"):
                            p = ev[mname][k]
                            row[f"{mname}_p_true"] = p[lab]
                            row[f"{mname}_logp_true"] = float(np.log(max(p[lab], 1e-12)))
                            row[f"{mname}_pred"] = int(p.argmax())
                        for part in ("A4_phys_logits", "A4_latent_contrib", "A2_phys_logits"):
                            row[f"{part}_true"] = ev[part][k, lab]
                        rows.append(row)
                print(f"{o['fold']} {b} done ({time.time()-t_start:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "pev_measurements.csv", index=False)
    print("saved", len(D), flush=True)


if __name__ == "__main__":
    main()
