"""Manuscript figures Fig3-Fig8 generated from result files only (no model training, no Ottawa re-evaluation).

Outputs: FINAL_PLOS_ONE_PACKAGE/figures/FigN.{tif,pdf,png}. PLOS ONE: width <= 7.5 in, TIFF LZW 600 dpi, Arial >= 8 pt (PLOS: 8-12 pt), RGB TIFF without alpha.
Colours: class identity H / IR / OR = categorical slots 1-3 (validated reference palette); magnitude = single-hue ramp.
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

FIGDIR = ROOT / "FINAL_PLOS_ONE_PACKAGE" / "figures"
T3 = ROOT / "results/phase3b/tables"
T4 = ROOT / "results/phase4/tables"
CLS = ["H", "IR", "OR"]
CCOL = {"H": "#2a78d6", "IR": "#eb6834", "OR": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["B0", "B1-V2", "B1-V3-1s", "B1-V3-4s", "B2", "B3", "B4", "A2", "A3", "A4", "A6"]
PHYS = {"B0", "B1-V2", "B1-V3-1s", "B1-V3-4s", "B2", "B4", "A2"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
                     "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.2})


def _rgb_tif(path):
    """PLOS: flattened RGB TIFF, no alpha channel, LZW compression."""
    from PIL import Image
    with Image.open(path) as im:
        rgb = Image.new("RGB", im.size, (255, 255, 255))
        rgb.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        dpi = im.info.get("dpi", (600, 600))
    rgb.save(path, compression="tiff_lzw", dpi=dpi)


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    _rgb_tif(FIGDIR / f"{name}.tif")
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def panel(ax, s):
    from matplotlib.transforms import offset_copy
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=-26, y=3, units="points")
    ax.text(0, 1, s, transform=tr, fontsize=9, fontweight="bold", ha="left", va="bottom", color=INK)


def kin_pu():
    from pave.signal.physics import BearingGeometry, orders
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    return {c: orders(BearingGeometry(c, int(r.geom_n_balls), r.geom_ball_diameter_mm, r.geom_pitch_diameter_mm, 0.0)) for c, r in bear.iterrows()}


def spectrum_panel(ax, f, E, fr, kin, title, fmax=None):
    fmax = fmax or 3.6 * kin["BPFI"] * fr
    k = (f > 2) & (f <= fmax)
    rel = E[k] / np.median(E[k])
    ax.plot(f[k], rel, color=INK2, lw=0.5)
    for fam, col, ls in (("BPFO", CCOL["OR"], "-"), ("BPFI", CCOL["IR"], "--")):
        for h in (1, 2, 3):
            ax.axvline(h * kin[fam] * fr, color=col, lw=0.8, ls=ls, alpha=0.9, label=fam if h == 1 else None)
    for h in range(1, int(fmax / fr) + 1):
        ax.axvline(h * fr, color="#b8b7b0", lw=0.4, ls=":", label="shaft harmonics" if h == 1 else None)
    ax.set_xlim(0, fmax)
    ax.set_title(title, loc="left")
    ax.set_xlabel("envelope frequency (Hz)")
    ax.set_ylabel("amplitude / median")


def zoom_panel(ax, f, E, fr, kin, fam, fmax):
    """Separate zoom panel: +-6 Hz around the fundamental of the family, with the nearest shaft harmonic."""
    f0 = kin[fam] * fr
    k = (f > f0 - 6) & (f < f0 + 6)
    ax.plot(f[k], E[k] / np.median(E[(f > 2) & (f <= fmax)]), color=INK2, lw=0.7)
    ax.axvline(f0, color=CCOL["OR"] if fam == "BPFO" else CCOL["IR"], lw=0.9, ls="-" if fam == "BPFO" else "--")
    n = round(f0 / fr)
    ax.axvline(n * fr, color="#8f8e88", lw=0.7, ls=":")
    ax.set_xlim(f0 - 6, f0 + 6)
    ax.set_title(f"zoom: {fam} fundamental", loc="left")


# ------------------------------------------------------------------ Fig 3: evidence extraction
def fig3():
    meas = pd.read_csv(ROOT / "results/phase3b/env4s/meas.csv")
    env = np.load(ROOT / "results/phase3b/env4s/env4.npy", mmap_mode="r")
    f = np.arange(env.shape[1]) * 0.25
    kin = kin_pu()
    fin = json.loads((ROOT / "results/phase3b/folds/FINAL.json").read_text())
    z = np.load(ROOT / "results/phase3b/folds/FINAL.npz")
    fig = plt.figure(figsize=(7.5, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.38)
    for j, (b, lab) in enumerate((("K001", "healthy (K001)"), ("KI04", "inner race (KI04)"), ("KA04", "outer race (KA04)"))):
        r = meas[(meas.bearing == b) & (meas.setting == "N15_M07_F10")].sort_values("measurement").iloc[0]
        ax = fig.add_subplot(gs[0, j])
        spectrum_panel(ax, f, np.asarray(env[r.name]), r.fr_hz, kin[b], lab)
        if j:
            ax.set_ylabel("")
        if j == 0:
            panel(ax, "A")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    bd5 = set(fin["train_bearings"])
    sel = meas.bearing.isin(bd5).to_numpy()
    C, M = z["meas_C4"], z["meas_M4"]
    ax = fig.add_subplot(gs[1, 0:2])
    for c in CLS:
        k = sel & (meas.label3 == c).to_numpy() & (M[:, 1] > 0) & (M[:, 4] > 0)
        ax.scatter(C[k, 1], C[k, 4], s=5, alpha=0.55, color=CCOL[c], edgecolors="none", label=f"{c} (n = {k.sum()})")
    ax.axvline(fin["b0_tau_dB"], color=INK2, lw=0.7, ls="--")
    ax.axhline(fin["b0_tau_dB"], color=INK2, lw=0.7, ls="--")
    ax.plot([], [], color=INK2, lw=0.7, ls="--", label="B0 threshold")
    ax.set_xlabel("BPFO harmonic evidence (dB)")
    ax.set_ylabel("BPFI harmonic evidence (dB)")
    ax.set_title("Harmonic evidence, 4-s records of the 15 PU-BD5 bearings", loc="left")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, loc="upper right", markerscale=2)
    panel(ax, "B")
    # validity (identifiable fraction) by context, from the stored masks
    ax = fig.add_subplot(gs[1, 2])
    seg = pd.read_csv(ROOT / "results/phase4/ottawa_segments4s.csv")
    vals = {"PU 4 s": (M[sel, 1].mean(), M[sel, 4].mean()), "Ottawa 4 s": (seg.valid_BPFO.mean(), seg.valid_BPFI.mean())}
    x = np.arange(2)
    for i, (fam, col) in enumerate((("BPFO", CCOL["OR"]), ("BPFI", CCOL["IR"]))):
        v = [vals[k][i] for k in vals]
        ax.bar(x + (i - 0.5) * 0.36, v, 0.34, color=col, label=fam)
        for xx, vv in zip(x + (i - 0.5) * 0.36, v):
            ax.text(xx, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, list(vals))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of 4-s records")
    ax.set_title("Identifiable records", loc="left")
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    panel(ax, "C")
    save(fig, "Fig3")


# ------------------------------------------------------------------ Fig 4: PU bearing-level classification
def fig4():
    mr = pd.read_csv(T3 / "main_results.csv")
    mr = mr.set_index("model").loc[ORDER]
    ci = np.array([json.loads(s) for s in mr.ci95])
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.9), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.22})
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.30, top=0.91)
    ax = axs[0]
    yy = np.arange(len(ORDER))[::-1]
    for i, m in enumerate(ORDER):
        col = CCOL["H"] if m in PHYS else CCOL["IR"]
        ax.plot(ci[i], [yy[i]] * 2, color=col, lw=1.2)
        ax.plot(mr.bearing_macro_f1.iloc[i], yy[i], "o", ms=4.5, color=col, mec="white", mew=0.6)
        ax.text(1.02, yy[i], f"{mr.bearing_macro_f1.iloc[i]:.3f}", va="center", fontsize=8, color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(yy, ORDER)
    ax.set_xlim(0, 1)
    ax.set_xlabel("bearing-level macro-F1 (95% CI)")
    ax.set_title("PU-BD5, 15 held-out bearings", loc="left")
    ax.plot([], [], "o", color=CCOL["H"], label="concept-only")
    ax.plot([], [], "o", color=CCOL["IR"], label="uses raw-signal CNN")
    ax.legend(frameon=False, loc="upper left", fontsize=8, bbox_to_anchor=(-0.05, -0.17), ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax.set_ylim(-0.5, len(ORDER) - 0.5)
    ax.grid(axis="y", visible=False)
    panel(ax, "A")
    pb = pd.read_csv(T3 / "per_bearing_probs.csv")
    pb["ptrue"] = pb[["pH", "pIR", "pOR"]].to_numpy()[np.arange(len(pb)), pb.y.to_numpy()]
    mat = pb.pivot(index="model", columns="bearing", values="ptrue").loc[ORDER]
    cor = pb.pivot(index="model", columns="bearing", values="correct").loc[ORDER]
    cols = sorted(mat.columns, key=lambda b: (b[:2] != "K0", b[:2] != "KI", b))
    mat, cor = mat[cols], cor[cols]
    ax = axs[1]
    im = ax.imshow(mat.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not bool(cor.iloc[i, j]):
                ax.text(j, i, "×", ha="center", va="center", fontsize=8, color="#e34948", fontweight="bold")
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(ORDER)), [""] * len(ORDER))  # rows aligned with panel A
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for x0 in (4.5, 9.5):
        ax.axvline(x0, color="white", lw=2)
    ax.set_title("Probability of the true class (× misclassified)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("p(true class)")
    cb.outline.set_visible(False)
    panel(ax, "B")
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig 5: PEV
def fig5():
    bl = pd.read_csv(T3 / "pev_bearing_level.csv")
    own = bl[(bl.relation == "own_family") & (bl.strength == 1.0)]
    tm = pd.read_csv(T3 / "pev_targeted_minus_control.csv")
    tm = tm[(tm.strength == 1.0) & (tm.control == "Ce")]
    ot = pd.read_csv(ROOT / "results/phase4/pev/pev_ottawa_targeted_minus_control.csv").set_index("quantity")
    om = pd.read_csv(ROOT / "results/phase4/pev/pev_ottawa_measurements.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 3.3), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1, 1.3]})
    fig.subplots_adjust(bottom=0.27, top=0.9, left=0.08, right=0.98)
    # A: evidence change per bearing (PU) for T, Cw, Ce
    ax = axs[0]
    kinds = [("T", "targeted"), ("Cw", "width\ncontrol"), ("Ce", "energy\ncontrol")]
    for i, (k, lab) in enumerate(kinds):
        v = own[own.kind == k].d_target_harm.to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=9, color=CCOL["IR"] if k == "T" else INK2, edgecolors="none")
        ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color=INK, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(3), [l for _, l in kinds])
    ax.set_ylabel("change in own-family evidence (dB)")
    ax.set_title("PU: evidence removed", loc="left")
    ax.grid(axis="x", visible=False)
    panel(ax, "A")
    # B: same for Ottawa (recording level)
    ax = axs[1]
    om["own"] = np.where(om.label3 == "OR", "BPFO", "BPFI")
    om["evidence"] = np.where(om.own == "BPFO", om.c_BPFO_harm, om.c_BPFI_harm)
    base = om[om.variant == "original"].groupby("stem").evidence.mean()
    for i, (k, lab) in enumerate(kinds):
        s = om[(om.strength == 1.0) & (om.variant == om.own.radd(k + "_"))].groupby("stem").evidence.mean()
        v = (s - base.loc[s.index]).to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=9, color=CCOL["IR"] if k == "T" else INK2, edgecolors="none")
        ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color=INK, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(3), [l for _, l in kinds])
    ax.set_title("Ottawa: evidence removed", loc="left")
    ax.grid(axis="x", visible=False)
    panel(ax, "B")
    # C: decision effect T - Ce on p(true)
    ax = axs[2]
    models = ["B1-V3-4s", "A2", "A4", "A6", "B3"]
    y0 = np.arange(len(models))[::-1]
    for i, m in enumerate(models):
        r = tm[tm.metric == f"dp_{m}"].iloc[0]
        ax.plot([r.ci95_lo, r.ci95_hi], [y0[i] + 0.15] * 2, color=CCOL["H"], lw=1.2)
        ax.plot(r.mean_T_minus_C, y0[i] + 0.15, "o", ms=4.5, color=CCOL["H"], mec="white", mew=0.6, label="PU (mean, 10 bearings)" if i == 0 else None)
        if m in ot.index:
            q = ot.loc[m]
            ax.plot([q.ci95_lo, q.ci95_hi], [y0[i] - 0.15] * 2, color=CCOL["OR"], lw=1.2)
            ax.plot(q.median_T_minus_Ce, y0[i] - 0.15, "s", ms=4.2, color=CCOL["OR"], mec="white", mew=0.6, label="Ottawa (median, 10 recordings)" if m == "B1-V3-4s" else None)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(y0, models)
    ax.set_xlabel("Δp(true class), targeted − energy control")
    ax.set_title("Change in p(true class)", loc="left")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="lower right", fontsize=8, ncol=2, bbox_to_anchor=(0.99, 0.0))
    ax.grid(axis="y", visible=False)
    panel(ax, "C")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.9)
    save(fig, "Fig5")


# ------------------------------------------------------------------ Fig 6: risk-coverage (PU, recomputed as in report_phase3b)
def risk_curves():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from pave.calibration.temperature import apply_temperature, fit_temperature
    from pave.experiments import phase3a as P3A
    from pave.experiments import phase3b as P3B
    meta = pd.read_csv(ROOT / "results/phase2/windows/pu/meta.csv")
    w2m = np.load(ROOT / "results/phase3b/env4s/win2meas.npy")
    folds = [(json.loads((ROOT / f"results/phase3b/folds/O{k}.json").read_text()), np.load(ROOT / f"results/phase3b/folds/O{k}.npz")) for k in range(1, 6)]
    curves = {}
    for base in ("B1-V3-4s", "A4"):
        sc = {k: [] for k in ("S3", "S4")}
        Pall, y, b = [], [], []
        for js, z in folds:
            t = fit_temperature(z[f"oof_{base}"], z["oof_y"])
            Pt = apply_temperature(z[f"test_{base}"], t)
            Po = apply_temperature(z[f"oof_{base}"], t)
            mt = w2m[z["test_idx"]]
            kt = P3B.compatibility(z["meas_C4"][mt], z["meas_M4"][mt], Pt.argmax(1), js["b0_tau_dB"])
            ko = P3B.compatibility(z["oof_C4"], z["oof_M4"], Po.argmax(1), z["oof_tau"])
            s4 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(
                np.c_[Po.max(1), ko, z["oof_Q4"][:, 2]], Po.argmax(1) == z["oof_y"])
            sc["S4"].append(s4.predict_proba(np.c_[Pt.max(1), kt, z["meas_Q4"][mt, 2]])[:, 1])
            sc["S3"].append(z["meas_Q4"][mt, 2] * np.abs(z["meas_Q4"][mt, 3] - js["b0_tau_dB"]))
            Pall.append(Pt)
            y.append(z["test_y"])
            b.append(meta.bearing.to_numpy()[z["test_idx"]])
        P, y, b = np.concatenate(Pall), np.concatenate(y), np.concatenate(b)
        bt = P3A.bearing_table(P, y, b).set_index("bearing")
        pp = bt[["pH", "pIR", "pOR"]].to_numpy()
        scores = {"S1": pd.Series(pp.max(1), index=bt.index), "S2": pd.Series(np.sum(pp * np.log(np.clip(pp, 1e-12, 1)), 1), index=bt.index),
                  "S3": pd.Series(np.concatenate(sc["S3"])).groupby(b).mean(), "S4": pd.Series(np.concatenate(sc["S4"])).groupby(b).mean()}
        for s, v in scores.items():
            d = bt.join(v.rename("score")).sort_values("score", ascending=False)
            err = (~d.correct).to_numpy().astype(float)
            curves[(base, s)] = np.cumsum(err) / np.arange(1, len(err) + 1)
    return curves


def fig6():
    curves = risk_curves()
    sd = pd.read_csv(T3 / "selective_diagnosis.csv").set_index(["base", "score"])
    # consistency with the frozen table
    for k, r in curves.items():
        assert abs(r.mean() - sd.loc[k, "AURC"]) < 1e-3, (k, r.mean(), sd.loc[k, "AURC"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    cols = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#6250d6"}
    names = {"S1": "R1 max prob.", "S2": "R2 −entropy", "S3": "R3 evidence clarity", "S4": "R4 learned"}
    mk = {"S1": "o", "S2": "s", "S3": "^", "S4": "D"}
    for ax, base, lab in zip(axs, ("B1-V3-4s", "A4"), ("A", "B")):
        for s in ("S1", "S2", "S3", "S4"):
            ax.plot(np.arange(1, 16) / 15, curves[(base, s)], marker=mk[s], ms=3, color=cols[s],
                    label=f"{s.replace('S', 'R')} ({sd.loc[(base, s), 'AURC']:.3f})")
        for c in (0.6, 0.8):
            ax.axvline(c, color=INK2, lw=0.5, ls=":")
        ax.set_title(f"{base} ({int(sd.loc[(base, 'S1'), 'n_errors_total'])} errors / 15 bearings)", loc="left")
        ax.set_xlabel("coverage (fraction of bearings accepted)")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1, fontsize=8, loc="upper left", ncol=2,
                  title="score (AURC)", title_fontsize=8, alignment="left")
        panel(ax, lab)
    axs[0].set_ylabel("selective risk")
    axs[0].set_ylim(-0.02, 1.0)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.88)
    save(fig, "Fig6")


# ------------------------------------------------------------------ Fig 7: external validation
def fig7():
    mr = pd.read_csv(T3 / "main_results.csv").set_index("model")
    ex = pd.read_json(T4 / "external_primary.json").set_index("model")
    fig = plt.figure(figsize=(7.5, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.85, 0.9], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    yy = np.arange(len(ORDER))[::-1]
    for i, m in enumerate(ORDER):
        a, e = mr.loc[m, "bearing_macro_f1"], ex.loc[m, "macro_f1"]
        lo, hi = ex.loc[m, "ci95"]
        ax.plot([e, a], [yy[i]] * 2, color=GRID, lw=2.2, zorder=1)
        ax.plot([lo, hi], [yy[i] - 0.18] * 2, color=CCOL["OR"], lw=0.9)
        ax.plot(a, yy[i], "o", ms=4.5, color=CCOL["H"], mec="white", mew=0.6, zorder=3, label="PU-BD5 (internal)" if i == 0 else None)
        ax.plot(e, yy[i], "s", ms=4.5, color=CCOL["OR"], mec="white", mew=0.6, zorder=3, label="Ottawa (external, 95% CI)" if i == 0 else None)
    ax.set_yticks(yy, [m + (" *" if m == "B1-V3-4s" else "") for m in ORDER])
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1 (bearing / recording level)")
    ax.set_title("Internal vs. single external evaluation", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8, bbox_to_anchor=(-0.02, -0.19), ncol=2)
    ax.grid(axis="y", visible=False)
    panel(ax, "A")
    ax = fig.add_subplot(gs[1])
    cm = np.array(ex.loc["B1-V3-4s", "confusion"])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_xticks(range(3), CLS)
    ax.set_yticks(range(3), CLS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title("B1-V3-4s on Ottawa", loc="left")
    panel(ax, "B")
    ax = fig.add_subplot(gs[2])
    fa = ex.loc[ORDER, "healthy_false_alarm"].to_numpy()
    sens = ex.loc[ORDER, "fault_sensitivity"].to_numpy()
    ax.barh(yy, fa, 0.6, color=[CCOL["H"] if m in PHYS else CCOL["IR"] for m in ORDER])
    for yv, v, sv in zip(yy, fa, sens):
        ax.text(v + 0.02, yv, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(yy, ORDER)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("healthy false-alarm rate")
    ax.set_title("Ottawa false alarms", loc="left")
    ax.grid(axis="y", visible=False)
    panel(ax, "C")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.9)
    save(fig, "Fig7")


# ------------------------------------------------------------------ Fig 8: success and failure cases
def fig8():
    from pave.data import ottawa as OT
    from scripts.external_ottawa import KIN_OT, env_spec
    meas = pd.read_csv(ROOT / "results/phase3b/env4s/meas.csv")
    env = np.load(ROOT / "results/phase3b/env4s/env4.npy", mmap_mode="r")
    f = np.arange(env.shape[1]) * 0.25
    kin = kin_pu()
    pb = pd.read_csv(T3 / "per_bearing_probs.csv").query("model == 'B1-V3-4s'").set_index("bearing")
    rp = pd.read_csv(T4 / "external_recording_predictions.csv").query("model == 'B1-V3-4s'").set_index("stem")
    fig, axs = plt.subplots(4, 2, figsize=(7.5, 7.6), gridspec_kw={"hspace": 0.75, "wspace": 0.18, "width_ratios": [3.2, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.92)

    def ptxt(p):
        return f"p(H, IR, OR) = {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}"

    cases = []
    for b, lab, fam in (("KA16", "PU KA16, outer race, correct", "BPFO"), ("KA30", "PU KA30, outer race, missed", "BPFO")):
        r = meas[(meas.bearing == b) & (meas.setting == "N15_M07_F10")].sort_values("measurement").iloc[0]
        cases.append((f, np.asarray(env[r.name]), r.fr_hz, kin[b], fam, lab + "; bearing mean " + ptxt(pb.loc[b, ["pH", "pIR", "pOR"]].to_numpy())))
    raw = ROOT / "01_RAW_DATA" / "Ottawa_data"
    files = {p.stem: p for p in OT.find_files(raw)}
    for st, lab in (("I_1_2", "Ottawa I_1_2, inner race, correct"), ("H_3_0", "Ottawa H_3_0, healthy, false alarm")):
        a = OT.read_ottawa(files[st])
        fr = float(a[0, 2]) / 60.0
        fo, Eo = env_spec(a[: int(4 * OT.FS), 0].astype(np.float64), OT.FS, int(4 * OT.FS))
        cases.append((fo, Eo, fr, KIN_OT, "BPFI", lab + "; " + ptxt(rp.loc[st, ["pH", "pIR", "pOR"]].to_numpy())))
    for i, ((ff, E, fr, kk, fam, title), tag) in enumerate(zip(cases, "ABCD")):
        fmax = 3.6 * kk["BPFI"] * fr
        spectrum_panel(axs[i, 0], ff, E, fr, kk, title, fmax=fmax)
        zoom_panel(axs[i, 1], ff, E, fr, kk, fam, fmax)
        axs[i, 0].set_ylabel("amplitude / median")
        if i < 3:
            axs[i, 0].set_xlabel("")
        else:
            axs[i, 1].set_xlabel("frequency (Hz)")
        panel(axs[i, 0], tag)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    save(fig, "Fig8")


if __name__ == "__main__":
    which = sys.argv[1:] or ["3", "4", "5", "6", "7", "8"]
    for w in which:
        globals()[f"fig{w}"]()
