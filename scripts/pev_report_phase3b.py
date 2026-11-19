"""Summarise Phase 3B PEV (bearing = unit). Writes results/phase3b/tables/pev_*.csv and figure."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = ROOT / "results/phase3b/pev"
TAB = ROOT / "results/phase3b/tables"
FIG = ROOT / "results/phase3b/figures"
MODELS = ["B1-V3-4s", "A2", "A4", "A6", "B3"]


def main():
    X = pd.read_csv(D / "pev_measurements.csv")
    key = ["bearing", "setting", "measurement"]
    base = X[X.variant == "original"].set_index(key)
    cols = [f"{m}_p_true" for m in MODELS] + [f"{m}_logp_true" for m in MODELS] + [f"{m}_pred" for m in MODELS] + \
           ["c_BPFO_harm", "c_BPFI_harm", "A4_phys_logits_true", "A4_latent_contrib_true"]
    X = X.join(base[cols], on=key, rsuffix="_orig")
    for m in MODELS:
        X[f"dp_{m}"] = X[f"{m}_p_true"] - X[f"{m}_p_true_orig"]
        X[f"dlogp_{m}"] = X[f"{m}_logp_true"] - X[f"{m}_logp_true_orig"]
        X[f"flip_{m}"] = (X[f"{m}_pred"] != X[f"{m}_pred_orig"]).astype(float)
    X["d_A4_phys"] = X["A4_phys_logits_true"] - X["A4_phys_logits_true_orig"]
    X["d_A4_latent"] = X["A4_latent_contrib_true"] - X["A4_latent_contrib_true_orig"]
    V = X[X.variant != "original"].copy()
    V["kind"] = V.variant.str.split("_").str[0]
    V["target"] = V.variant.str.split("_").str[1]
    own = {"OR": "BPFO", "IR": "BPFI"}
    V["relation"] = np.where(V.label3 == "H", "healthy", np.where(V.target == V.label3.map(own), "own_family", "other_family"))
    V["d_target_harm"] = np.where(V.target == "BPFO", V.c_BPFO_harm - V.c_BPFO_harm_orig, V.c_BPFI_harm - V.c_BPFI_harm_orig)
    V["d_other_harm"] = np.where(V.target == "BPFO", V.c_BPFI_harm - V.c_BPFI_harm_orig, V.c_BPFO_harm - V.c_BPFO_harm_orig)
    metrics = ["d_target_harm", "d_other_harm"] + [f"dp_{m}" for m in MODELS] + [f"dlogp_{m}" for m in MODELS] + \
              [f"flip_{m}" for m in MODELS] + ["d_A4_phys", "d_A4_latent"]
    B = V.groupby(["relation", "kind", "strength", "bearing"])[metrics].mean().reset_index()
    B.round(4).to_csv(TAB / "pev_bearing_level.csv", index=False)
    agg = B.groupby(["relation", "kind", "strength"])[metrics].median().round(4)
    agg.insert(0, "n_bearings", B.groupby(["relation", "kind", "strength"]).bearing.nunique())
    agg.to_csv(TAB / "pev_aggregate_medians.csv")
    # primary contrast: own family, strength 1, targeted - matched-energy control, per bearing; bootstrap over bearings
    rng = np.random.default_rng(11)
    prim = []
    for s in (1.0, 0.5):
        sub = B[(B.relation == "own_family") & (B.strength == s)].pivot(index="bearing", columns="kind")
        for m in ["d_target_harm"] + [f"dp_{x}" for x in MODELS] + [f"dlogp_{x}" for x in MODELS] + [f"flip_{x}" for x in MODELS] + ["d_A4_phys", "d_A4_latent"]:
            for ctrl in ("Ce", "Cw"):
                d = (sub[(m, "T")] - sub[(m, ctrl)]).dropna().to_numpy()
                bs = [np.mean(rng.choice(d, len(d))) for _ in range(10000)]
                prim.append({"strength": s, "metric": m, "control": ctrl, "n_bearings": len(d), "mean_T_minus_C": float(d.mean()),
                             "ci95_lo": float(np.quantile(bs, 0.025)), "ci95_hi": float(np.quantile(bs, 0.975)),
                             "n_bearings_T_lt_C": int((d < 0).sum())})
    P = pd.DataFrame(prim)
    P.round(4).to_csv(TAB / "pev_targeted_minus_control.csv", index=False)
    dist = V.groupby(["kind", "strength"]).agg(in_band_snr_db_median=("in_band_snr_db", "median"), clip_max=("clip_fraction", "max"),
                                                 oob_max=("oob_rel_change", "max"), energy_ratio_total_median=("total_energy_ratio", "median"),
                                                 energy_ratio_vs_target_median=("energy_ratio_vs_target", "median"),
                                                 energy_ratio_vs_target_p10=("energy_ratio_vs_target", lambda s: s.quantile(0.1)),
                                                 overlap_other_family_hz_max=("overlap_other_family_hz", "max"),
                                                 overlap_shaft_guard_hz_max=("overlap_shaft_guard_hz", "max")).round(4)
    dist.to_csv(TAB / "pev_distortion.csv")
    bys = V[(V.relation == "own_family") & (V.strength == 1.0)].groupby(["setting", "kind"])[["d_target_harm"] + [f"dp_{m}" for m in MODELS]].median().round(4)
    bys.to_csv(TAB / "pev_by_setting.csv")
    fig, axes = plt.subplots(1, len(MODELS) + 1, figsize=(7.2, 2.4), sharey=False)
    sub = B[(B.relation == "own_family") & (B.strength == 1.0)]
    col = {"T": "#d03b3b", "Cw": "#b5b4ad", "Ce": "#52514e"}
    for ax, m in zip(axes, ["d_target_harm"] + [f"dp_{x}" for x in MODELS]):
        for j, k in enumerate(("T", "Cw", "Ce")):
            v = sub[sub.kind == k][m]
            ax.scatter(np.full(len(v), j) + np.linspace(-0.2, 0.2, len(v)), v, s=8, color=col[k], zorder=3)
            ax.plot([j - 0.3, j + 0.3], [v.median()] * 2, color="#0b0b0b", lw=1)
        ax.axhline(0, color="#52514e", lw=0.5)
        ax.set_xticks([0, 1, 2], ["T", "Cw", "Ce"], fontsize=6)
        ax.set_title(m.replace("dp_", "Δp ").replace("d_target_harm", "Δ evidence [dB]"), fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("PEV on 10 faulty PU-BD5 bearings (bearing means; strength 1.0; T targeted, Cw width control, Ce energy control)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pev.png", dpi=300)
    fig.savefig(FIG / "fig_pev.pdf")
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(P[(P.strength == 1.0) & (P.control == "Ce")].round(3).to_string())
    print(dist.to_string())


if __name__ == "__main__":
    main()
