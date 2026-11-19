"""Audit figures from genuine data (PNG 300 dpi). All panels are computed from cached audit outputs."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pave.data.conditions import parse_filename  # noqa: E402
from pave.signal.physics import all_frequencies  # noqa: E402

# validated reference palette (categorical slots 1-3), neutral for out-of-scope
GROUP_COLOR = {"Healthy": "#2a78d6", "IR": "#eb6834", "OR": "#1baf7a", "Misalign": "#8a8984", "Unbalance": "#b5b4ad"}
LOAD_COLOR = {0: "#86b6ef", 2: "#2a78d6", 4: "#104281"}  # sequential blue 250/450/650
SEV_MARKER = {None: "o", 0.3: "o", 1.0: "s", 3.0: "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CLASSES7 = ["H", "IR_0p3", "IR_1p0", "IR_3p0", "OR_0p3", "OR_1p0", "OR_3p0"]
CLASS_GROUP = {"H": "Healthy", **{c: c[:2] for c in CLASSES7[1:]}}
CLASS_TXT = {"H": "Healthy", "IR_0p3": "IR 0.3 mm", "IR_1p0": "IR 1.0 mm", "IR_3p0": "IR 3.0 mm",
             "OR_0p3": "OR 0.3 mm", "OR_1p0": "OR 1.0 mm", "OR_3p0": "OR 3.0 mm"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 8, "axes.titlesize": 9,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.0,
})


def _save(fig, d: Path, name: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_distribution(rec: pd.DataFrame, d: Path) -> None:
    order = sorted(rec.condition_id.unique(), key=lambda c: (not rec[rec.condition_id == c].in_bearing_scope.iloc[0], c))
    M = rec.pivot_table(index="condition_id", columns="load_Nm", values="vib_duration_s").loc[order]
    fig, ax = plt.subplots(figsize=(4.2, 5.2))
    ax.imshow(M.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#256abf"]),
              aspect="auto", vmin=0, vmax=320)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.0f} s", ha="center", va="center", fontsize=7,
                    color="white" if M.values[i, j] > 200 else INK)
    ax.set_xticks(range(3), [f"{c} Nm" for c in M.columns])
    ax.set_yticks(range(len(order)), order)
    n_in = int(rec[rec.in_bearing_scope].condition_id.nunique())
    ax.axhline(n_in - 0.5, color=INK, lw=1.2)
    ax.text(2.6, n_in / 2 - 0.5, "bearing task\n(21 recordings)", va="center", fontsize=7, color=INK2)
    ax.text(2.6, n_in + (len(order) - n_in) / 2 - 0.5, "out of scope\n(24 recordings)", va="center", fontsize=7, color=INK2)
    ax.grid(False)
    ax.set_title("Recording duration per condition and load\n(one recording per cell)")
    _save(fig, d, "fig01_dataset_distribution")


def fig_timeline(rec: pd.DataFrame, pairs: pd.DataFrame, d: Path) -> None:
    P = pairs.set_index("recording_id")
    sessions = sorted(rec.session.unique())
    fig, axes = plt.subplots(len(sessions), 1, figsize=(7.2, 9.0), gridspec_kw={"height_ratios": [30, 9, 9]})
    for ax, s in zip(axes, sessions):
        g = rec[rec.session == s].sort_values("vib_time_start_utc")
        off = P.loc[g.recording_id.iloc[0]].session_clock_offset_s
        t0 = min(dt.datetime.fromisoformat(x) for x in g.temp_time_start_utc_tdms_clock)
        for i, r in enumerate(g.itertuples()):
            p = P.loc[r.recording_id]
            vs = (dt.datetime.fromisoformat(r.vib_time_start_utc) - dt.timedelta(seconds=off) - t0).total_seconds() / 60
            ts = (dt.datetime.fromisoformat(r.temp_time_start_utc_tdms_clock) - t0).total_seconds() / 60
            col = GROUP_COLOR[r.fault_group]
            ax.barh(i + 0.18, r.vib_duration_s / 60, left=vs, height=0.34, color=col, edgecolor="white", lw=0.5)
            ax.barh(i - 0.18, r.temp_duration_s / 60, left=ts, height=0.34, color="white", edgecolor=col, lw=1.0)
            if p.pairing_status != "CONCURRENT_BY_NAME":
                tag = "TDMS filename swap (resolved)" if p.pairing_status == "FILENAME_SWAP_RESOLVED" else \
                    f"temperature NOT concurrent ({p.resolved_residual_s/60:+.1f} min)"
                ax.text(max(vs, ts) + max(r.vib_duration_s, r.temp_duration_s) / 60 + 0.5, i, tag, va="center",
                        fontsize=6.5, color=INK)
        ax.set_yticks(range(len(g)), g.recording_id, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f"Session {s}: vibration (filled, clock-offset corrected by {off:.1f} s) vs TDMS log (outline)",
                     fontsize=8, loc="left")
        ax.set_xlabel("minutes since first TDMS log of the session (NI cDAQ clock)")
    handles = [matplotlib.patches.Patch(color=c, label=k) for k, c in GROUP_COLOR.items()]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7, ncol=5, bbox_to_anchor=(1, 1.04))
    fig.tight_layout()
    _save(fig, d, "fig02_acquisition_timeline")


def fig_offsets(pairs: pd.DataFrame, d: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, col, title in ((axes[0], "raw_offset_by_filename_s", "Raw start offset, files paired by filename"),
                           (axes[1], "resolved_residual_s", "Residual after per-session clock offset (resolved pairing)")):
        for k, (s, g) in enumerate(pairs.groupby("session")):
            x = k + np.linspace(-0.25, 0.25, len(g))
            cols = [GROUP_COLOR[parse_filename_safe(r)] for r in g.recording_id]
            ax.scatter(x, g[col], s=14, c=cols, edgecolors="white", linewidths=0.6, zorder=3)
            for xx, (_, r) in zip(x, g.iterrows()):
                if abs(r[col] - (r.session_clock_offset_s if col.startswith("raw") else 0)) > 5:
                    ax.annotate(r.recording_id, (xx, r[col]), fontsize=6, xytext=(4, 0), textcoords="offset points",
                                color=INK)
        ax.set_xticks(range(pairs.session.nunique()), sorted(pairs.session.unique()))
        ax.set_ylabel("vibration start - TDMS start [s]" if col.startswith("raw") else "residual [s]")
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    _save(fig, d, "fig03_clock_offsets")


def parse_filename_safe(rid: str) -> str:
    # recording_id like L0_IR_0p3 / L0_H / L2_Unbalance_583mg / L4_Misalign_0p1
    grp = rid.split("_")[1]
    return {"H": "Healthy"}.get(grp, grp)


def _rid(label: str, load: int) -> str:
    return f"L{load}_{label}"


def fig_waveforms(cache: Path, rec: pd.DataFrame, d: Path, load: int = 0) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(7.2, 7.0), sharex=True)
    R = rec.set_index("recording_id")
    for ax, c in zip(axes, CLASSES7):
        stem = Path(R.loc[_rid(c, load)].vib_relative_path).stem
        sn = np.load(cache / f"vib_{stem}.npz")["snippet"]
        t = np.arange(sn.shape[0]) / 25600 * 1000
        ax.plot(t, sn[:, 0], color=GROUP_COLOR[CLASS_GROUP[c]], lw=0.5)
        ax.set_ylabel(CLASS_TXT[c], rotation=0, ha="right", va="center", fontsize=7)
    axes[-1].set_xlabel("time [ms] (200 ms excerpt from mid-recording; housing A, x; stored units, see report)")
    axes[0].set_title(f"Raw vibration waveforms, {load} Nm")
    fig.tight_layout()
    _save(fig, d, f"fig04_waveforms_{load}Nm")


def fig_psd(cache: Path, rec: pd.DataFrame, d: Path, load: int = 0) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True, sharey=True)
    R = rec.set_index("recording_id")
    for ax, c in zip(axes, ["H", "IR_1p0", "OR_1p0"]):
        z = np.load(cache / f"vib_{Path(R.loc[_rid(c, load)].vib_relative_path).stem}.npz")
        ax.semilogy(z["psd_f"], z["psd"][:, 0], color=GROUP_COLOR[CLASS_GROUP[c]], lw=0.6)
        ax.set_ylabel("PSD [unit²/Hz]")
        ax.text(0.99, 0.9, CLASS_TXT[c], transform=ax.transAxes, ha="right", fontsize=8)
    axes[-1].set_xlabel("frequency [Hz] (Welch, 1 Hz resolution, housing A x)")
    axes[0].set_title(f"Power spectral density, {load} Nm")
    fig.tight_layout()
    _save(fig, d, f"fig05_psd_{load}Nm")


def fig_envelope_grid(cache: Path, rec: pd.DataFrame, d: Path, channel: str = "x_housing_A", band: str = "FIXED") -> None:
    fig, axes = plt.subplots(7, 3, figsize=(7.2, 9.5), sharex=True)
    R = rec.set_index("recording_id")
    for i, c in enumerate(CLASSES7):
        for j, load in enumerate((0, 2, 4)):
            ax = axes[i, j]
            r = R.loc[_rid(c, load)]
            z = np.load(cache / f"vib_{Path(r.vib_relative_path).stem}.npz")
            f, E = z["env_f"], z[f"env|{channel}|{band}"]
            m = f <= 600
            ax.plot(f[m], E[m], color=GROUP_COLOR[CLASS_GROUP[c]], lw=0.5)
            fq = all_frequencies(r.shaft_fr_measured_hz)
            for h in (1, 2, 3):
                ax.axvline(h * fq["BPFO"], color=GROUP_COLOR["OR"], lw=0.7, ls=(0, (3, 2)), alpha=0.9)
                ax.axvline(h * fq["BPFI"], color=GROUP_COLOR["IR"], lw=0.7, ls=(0, (1, 1.5)), alpha=0.9)
            ax.set_yticks([])
            ax.set_xlim(0, 600)
            if j == 0:
                ax.set_ylabel(CLASS_TXT[c], rotation=0, ha="right", va="center", fontsize=7)
            if i == 0:
                ax.set_title(f"{load} Nm")
    for ax in axes[-1]:
        ax.set_xlabel("envelope frequency [Hz]")
    fig.suptitle(f"Envelope spectra ({channel}, band {band}); dashed = k×BPFO, dotted = k×BPFI at measured speed",
                 fontsize=8, y=1.0)
    fig.tight_layout()
    _save(fig, d, f"fig06_envelope_spectra_{channel}_{band}")


def fig_evidence_heatmap(ev: pd.DataFrame, d: Path, band: str = "FIXED") -> None:
    e = ev[(ev.band == band) & ev.channel.isin(["x_housing_A", "y_housing_A"]) & ev.in_bearing_scope]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f7f7f5", "#86b6ef", "#1c5cab", "#0d366b"])
    for ax, tgt in zip(axes, ("BPFO_1", "BPFI_1")):
        g = e[e.target == tgt].groupby(["label7", "load_Nm"]).agg(x=("excess_db", "max"),
                                                                  p=("ctrl_exceed_frac", "min")).reset_index()
        M = g.pivot(index="label7", columns="load_Nm", values="x").loc[CLASSES7]
        Pm = g.pivot(index="label7", columns="load_Nm", values="p").loc[CLASSES7]
        im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=30, aspect="auto")
        for a in range(7):
            for b in range(3):
                det = M.values[a, b] >= 6 and Pm.values[a, b] <= 0
                ax.text(b, a, f"{M.values[a, b]:.1f}" + ("*" if det else ""), ha="center", va="center", fontsize=7,
                        color="white" if M.values[a, b] > 17 else INK, fontweight="bold" if det else "normal")
        ax.set_xticks(range(3), ["0 Nm", "2 Nm", "4 Nm"])
        ax.set_yticks(range(7), [CLASS_TXT[c] for c in CLASSES7])
        ax.grid(False)
        ax.set_title(f"{tgt.replace('_1', '')} excess over control median [dB]", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="dB (max of housing-A x/y)")
    fig.text(0.01, -0.02, "* detected: excess ≥ 6 dB and target exceeds all 24 control frequencies (record level)",
             fontsize=6.5, color=INK2)
    _save(fig, d, f"fig07_evidence_heatmap_{band}")


def fig_window_detection(wev: pd.DataFrame, d: Path, band: str = "FIXED") -> None:
    w = wev[(wev.band == band) & wev.channel.isin(["x_housing_A", "y_housing_A"]) & wev.in_bearing_scope]
    g = w.groupby(["label7", "load_Nm", "target"]).frac_detected.max().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, tgt in zip(axes, ("BPFO_1", "BPFI_1")):
        s = g[g.target == tgt]
        for j, load in enumerate((0, 2, 4)):
            vals = [s[(s.label7 == c) & (s.load_Nm == load)].frac_detected.iloc[0] for c in CLASSES7]
            ax.bar(np.arange(7) + (j - 1) * 0.27, vals, width=0.25, color=LOAD_COLOR[load], label=f"{load} Nm",
                   edgecolor="white", lw=0.5)
        ax.set_xticks(range(7), [CLASS_TXT[c].replace(" mm", "") for c in CLASSES7], rotation=30, ha="right")
        ax.set_title(f"1-s windows with {tgt.replace('_1', '')} evidence", fontsize=8)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("fraction of windows")
    axes[1].legend(title="load", fontsize=7, title_fontsize=7)
    fig.tight_layout()
    _save(fig, d, f"fig08_window_detection_{band}")


def fig_temperature_traces(cache: Path, rec: pd.DataFrame, d: Path, channel: str = "T_housing_A") -> None:
    sessions = sorted(rec.session.unique())
    fig, axes = plt.subplots(1, len(sessions), figsize=(7.2, 2.8), sharey=True)
    for ax, s in zip(axes, sessions):
        g = rec[rec.session == s]
        t0 = min(dt.datetime.fromisoformat(x) for x in g.temp_time_start_utc_tdms_clock)
        for r in g.itertuples():
            z = np.load(cache / f"tdms_{Path(r.tdms_relative_path).stem}.npz")
            st = (dt.datetime.fromisoformat(r.temp_time_start_utc_tdms_clock) - t0).total_seconds() / 60
            ax.plot(st + z[f"T|{channel}|t"] / 60, z[f"T|{channel}|v"], color=GROUP_COLOR[r.fault_group], lw=1.2)
        ax.set_title(s, fontsize=8)
        ax.set_xlabel("minutes since first log")
    axes[0].set_ylabel(f"{channel.replace('T_housing_', 'housing ')} temperature [°C]")
    handles = [matplotlib.lines.Line2D([], [], color=c, label=k) for k, c in GROUP_COLOR.items()]
    axes[-1].legend(handles=handles, fontsize=6.5, loc="lower right")
    fig.suptitle("Thermocouple traces of every TDMS log, in acquisition order (warm-up visible)", fontsize=8)
    fig.tight_layout()
    _save(fig, d, f"fig09_temperature_traces_{channel}")


def fig_temperature_by_class(trec: pd.DataFrame, d: Path) -> None:
    t = trec[trec.in_bearing_scope]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for ax, col, lab in ((axes[0], "TA_mean", "housing A mean [°C]"), (axes[1], "TB_mean", "housing B mean [°C]"),
                         (axes[2], "A_minus_B", "A − B [°C]")):
        for j, load in enumerate((0, 2, 4)):
            s = t[t.load_Nm == load].set_index("label7").loc[CLASSES7]
            ax.scatter(np.arange(7) + (j - 1) * 0.18, s[col], s=18, color=LOAD_COLOR[load], label=f"{load} Nm",
                       edgecolors="white", linewidths=0.6, zorder=3,
                       marker="o")
        nc = t[~t.temp_concurrent]
        for r in nc.itertuples():
            k = CLASSES7.index(r.label7)
            j = (0, 2, 4).index(r.load_Nm)
            ax.scatter(k + (j - 1) * 0.18, getattr(r, col), s=60, facecolors="none", edgecolors=INK, lw=0.8, zorder=4)
        ax.set_xticks(range(7), [CLASS_TXT[c].replace(" mm", "") for c in CLASSES7], rotation=35, ha="right")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=6.5, title="load", title_fontsize=6.5)
    fig.suptitle("Record-mean temperature by class and load (ringed = TDMS not concurrent with vibration)", fontsize=8)
    fig.tight_layout()
    _save(fig, d, "fig10_temperature_by_class_load")


def fig_signal_quality(sq: pd.DataFrame, d: Path, channel: str = "x_housing_A") -> None:
    s = sq[(sq.channel == channel) & sq.label7.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, col, lab in ((axes[0], "rms", "RMS [stored unit]"), (axes[1], "kurtosis", "kurtosis (Gaussian = 3)")):
        for j, load in enumerate((0, 2, 4)):
            v = s[s.load_Nm == load].set_index("label7").loc[CLASSES7]
            ax.scatter(np.arange(7) + (j - 1) * 0.18, v[col], s=18, color=LOAD_COLOR[load], label=f"{load} Nm",
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xticks(range(7), [CLASS_TXT[c].replace(" mm", "") for c in CLASSES7], rotation=35, ha="right")
        ax.set_ylabel(lab)
        ax.set_yscale("log" if col == "rms" else "linear")
    axes[0].legend(fontsize=6.5, title="load", title_fontsize=6.5)
    fig.suptitle(f"Whole-recording statistics, {channel}", fontsize=8)
    fig.tight_layout()
    _save(fig, d, f"fig11_signal_quality_{channel}")


def fig_speed(rec: pd.DataFrame, d: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for r in rec.itertuples():
        ax.scatter(r.load_Nm + (hash(r.condition_id) % 7 - 3) * 0.05, r.shaft_fr_measured_hz, s=14,
                   color=GROUP_COLOR[r.fault_group], edgecolors="white", linewidths=0.5, zorder=3)
    ax.axhline(3010 / 60, color=INK2, lw=0.8, ls="--")
    ax.text(4.3, 3010 / 60, "paper: 3010 rpm", fontsize=6.5, va="bottom", ha="right", color=INK2)
    ax.set_xticks([0, 2, 4], ["0 Nm", "2 Nm", "4 Nm"])
    ax.set_ylabel("measured shaft frequency [Hz]")
    ax.set_title("Shaft speed from vibration spectra", fontsize=8)
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in GROUP_COLOR.items()]
    ax.legend(handles=handles, fontsize=6, loc="lower left")
    fig.tight_layout()
    _save(fig, d, "fig12_shaft_speed")


def make_all(cache, d, rec, pairs, trec, ev, wev, dec, sq, tdms, vib) -> None:
    d = Path(d)
    fig_distribution(rec, d)
    fig_timeline(rec, pairs, d)
    fig_offsets(pairs, d)
    for load in (0, 4):
        fig_waveforms(cache, rec, d, load)
    fig_psd(cache, rec, d, 0)
    fig_envelope_grid(cache, rec, d, "x_housing_A", "FIXED")
    fig_envelope_grid(cache, rec, d, "x_housing_A", "SK")
    fig_evidence_heatmap(ev, d, "FIXED")
    fig_evidence_heatmap(ev, d, "SK")
    fig_window_detection(wev, d, "FIXED")
    fig_window_detection(wev, d, "SK")
    fig_temperature_traces(cache, rec, d, "T_housing_A")
    fig_temperature_traces(cache, rec, d, "T_housing_B")
    fig_temperature_by_class(trec, d)
    fig_signal_quality(sq, d, "x_housing_A")
    fig_signal_quality(sq, d, "x_housing_B")
    fig_speed(rec, d)
    fig_family(d, "FIXED")
    fig_family(d, "SK")


CALL_COLOR = {"none": "#b5b4ad", "IR-like": GROUP_COLOR["IR"], "OR-like": GROUP_COLOR["OR"], "ambiguous": "#6250d6"}


def fig_family(d: Path, band: str = "FIXED") -> None:
    from pave.signal.physics import orders
    tab = d.parent / "tables"
    W = pd.read_csv(tab / f"physics_family_windows_{band}.csv")
    P = pd.read_csv(tab / f"physics_family_per_recording_{band}.csv")
    o = orders()
    # --- fig13: observed family order per recording (whole-recording median over windows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, grp, col, key in ((axes[0], "IR", "med_order_I", "BPFI"), (axes[1], "OR", "med_order_O", "BPFO")):
        sub = P[P.fault_group == grp].set_index(["label7", "load_Nm"])
        labs = [c for c in CLASSES7 if c.startswith(grp)]
        th = o[key]
        ax.axhspan(th * 0.99, th * 1.01, color="#e6e5e0", zorder=0)
        ax.axhspan(th * (1 - 0.035), th * (1 + 0.035), color="#f4f4f1", zorder=-1)
        ax.axhline(th, color=INK2, lw=0.8, ls="--")
        ax.text(2.45, th, "kinematic\n(paper geometry)", fontsize=6, va="center", ha="right", color=INK2)
        for j, load in enumerate((0, 2, 4)):
            ys = [sub.loc[(l, load), col] for l in labs]
            ax.scatter(np.arange(3) + (j - 1) * 0.15, ys, s=22, color=LOAD_COLOR[load], label=f"{load} Nm",
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xticks(range(3), [CLASS_TXT[l] for l in labs])
        ax.set_ylabel(f"observed {key} family order (× fr)")
        ax.set_title(f"{grp}: strongest line in ±3.5 % band (dark band = ±1 %)", fontsize=8)
    axes[1].legend(fontsize=6.5, title="load", title_fontsize=6.5, loc="lower right")
    fig.tight_layout()
    _save(fig, d, f"fig13_observed_fault_orders_{band}")
    # --- fig14: window-level family scores
    t = json_thr(d.parent, band)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True, sharey=True)
    for ax, load in zip(axes, (0, 2, 4)):
        s = W[W.load_Nm == load]
        oos = s[~s.in_bearing_scope]
        ax.scatter(oos.score_O, oos.score_I, s=3, color=GROUP_COLOR["Misalign"], alpha=0.35, lw=0, label="misalign/unbalance")
        for g in ("Healthy", "IR", "OR"):
            q = s[s.fault_group == g]
            ax.scatter(q.score_O, q.score_I, s=3, color=GROUP_COLOR[g], alpha=0.6, lw=0, label=g)
        ax.axvline(t[0], color=INK2, lw=0.6, ls=":")
        ax.axhline(t[1], color=INK2, lw=0.6, ls=":")
        ax.set_title(f"{load} Nm", fontsize=8)
        ax.set_xlabel("BPFO-family score [dB]")
    axes[0].set_ylabel("BPFI-family score [dB]")
    axes[-1].legend(fontsize=6, markerscale=3, loc="upper right")
    fig.suptitle(f"1-s window evidence (band {band}); dotted = 99th pct of healthy windows", fontsize=8)
    fig.tight_layout()
    _save(fig, d, f"fig14_window_family_scores_{band}")
    # --- fig15: per-recording call shares
    ins = P[P.in_bearing_scope].copy()
    ins["k"] = [CLASSES7.index(l) * 3 + (0, 2, 4).index(ld) for l, ld in zip(ins.label7, ins.load_Nm)]
    ins = ins.sort_values("k")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bottom = np.zeros(len(ins))
    for cl in ("IR-like", "OR-like", "ambiguous", "none"):
        ax.bar(np.arange(len(ins)), ins[cl], bottom=bottom, color=CALL_COLOR[cl], label=cl, width=0.8,
               edgecolor="white", lw=0.5)
        bottom += ins[cl].values
    ax.set_xticks(np.arange(len(ins)), [f"{CLASS_TXT[l].replace(' mm', '')} {ld}Nm" for l, ld in zip(ins.label7, ins.load_Nm)],
                  rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("fraction of 1-s windows")
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, d, f"fig15_window_calls_{band}")


def json_thr(out: Path, band: str) -> tuple[float, float]:
    import json
    s = json.loads((out / "audit_summary.json").read_text())["family_search"][band]
    return s["thr_O_healthy_q99"], s["thr_I_healthy_q99"]
