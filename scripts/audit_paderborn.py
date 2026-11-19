"""Paderborn metadata audit (Phase 2, part C). Reads only manifests produced by build_paderborn_cache.py."""
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

from pave.data.paderborn import SETTINGS  # noqa: E402
from pave.signal.physics import BearingGeometry, orders  # noqa: E402

MAN = ROOT / "03_MANIFESTS" / "paderborn"
OUT = ROOT / "results" / "phase2" / "pu_audit"
GCOL = {"healthy": "#2a78d6", "IR_artificial": "#f4a582", "IR_real": "#eb6834", "OR_artificial": "#8fd9bd",
        "OR_real": "#1baf7a", "multiple_real": "#6250d6"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(MAN / "pu_files.csv")
    b = pd.read_csv(MAN / "pu_bearings.csv")
    f["dt"] = pd.to_datetime(f.datetime)
    s = {}
    # 1. inventory
    per = f.groupby(["bearing", "setting"]).size().unstack(fill_value=0)
    s["n_bearings"] = int(b.shape[0])
    s["n_files"] = int(len(f))
    s["files_per_bearing_setting_unique"] = sorted(set(per.values.ravel().tolist()))
    s["bearings_not_80_files"] = b[b.n_mat != 80].bearing.tolist()
    s["mat_sha_duplicates"] = int(f.mat_sha256.duplicated().sum())
    s["nan_inf_total"] = int(f.vib_nan.sum() + f.vib_inf.sum())
    s["n_vib_min_max"] = [int(f.n_vib.min()), int(f.n_vib.max())]
    s["files_longer_than_4s_frac"] = float(np.mean(f.n_vib > 256001))
    s["longest_constant_run_max"] = int(f.vib_longest_const.max())
    # 2. operating conditions vs nominal
    oc = f.groupby("setting").agg(speed_mean=("speed_rpm_mean", "mean"), speed_min=("speed_rpm_mean", "min"),
                                  speed_max=("speed_rpm_mean", "max"), torque_mean=("torque_mean", "mean"),
                                  torque_min=("torque_mean", "min"), torque_max=("torque_mean", "max"),
                                  force_mean=("force_mean", "mean"), force_min=("force_mean", "min"),
                                  force_max=("force_mean", "max"), temp_min=("temp_mean", "min"), temp_max=("temp_mean", "max"))
    oc["nominal_rpm"] = [SETTINGS[k]["rpm"] for k in oc.index]
    oc["nominal_torque"] = [SETTINGS[k]["torque_Nm"] for k in oc.index]
    oc["nominal_force"] = [SETTINGS[k]["force_N"] for k in oc.index]
    oc.round(3).to_csv(OUT / "operating_conditions_vs_nominal.csv")
    # speed by class within setting (shortcut check)
    f = f.merge(b[["bearing", "group", "label3"]], on="bearing")
    sp = f.groupby(["setting", "group"]).speed_rpm_mean.agg(["mean", "min", "max"]).round(3)
    sp.to_csv(OUT / "speed_by_setting_group.csv")
    s["speed_range_by_setting_rpm"] = {k: [float(v.min()), float(v.max())] for k, v in f.groupby("setting").speed_rpm_mean}
    # 3. geometry
    geo = b[["bearing", "group", "fs_manufacturer", "geom_pitch_diameter_mm", "geom_ball_diameter_mm", "geom_n_balls",
             "geom_contact_angle_deg", "geom_defaulted"]].copy()
    ko = [orders(BearingGeometry(r.bearing, int(r.geom_n_balls), r.geom_ball_diameter_mm, r.geom_pitch_diameter_mm,
                                 r.geom_contact_angle_deg)) for r in geo.itertuples()]
    geo["BPFO_order"] = [o["BPFO"] for o in ko]
    geo["BPFI_order"] = [o["BPFI"] for o in ko]
    geo.round(4).to_csv(OUT / "geometry_per_bearing.csv", index=False)
    s["manufacturers"] = geo.fs_manufacturer.value_counts(dropna=False).to_dict()
    s["pitch_diameters_mm"] = sorted(geo.geom_pitch_diameter_mm.unique().tolist())
    s["geometry_defaulted"] = geo[geo.geom_defaulted].bearing.tolist()
    s["BPFO_order_range"] = [float(geo.BPFO_order.min()), float(geo.BPFO_order.max())]
    s["BPFI_order_range"] = [float(geo.BPFI_order.min()), float(geo.BPFI_order.max())]
    s["manufacturer_by_group"] = pd.crosstab(geo.group, geo.fs_manufacturer.fillna("n/a")).to_dict()
    # 4. dates / sessions
    dts = f.groupby("bearing").dt.agg(["min", "max"]).merge(b[["bearing", "group"]].set_index("bearing"), left_index=True, right_index=True)
    dts["date"] = dts["min"].dt.date.astype(str)
    dts["duration_min"] = (dts["max"] - dts["min"]).dt.total_seconds() / 60
    dts.sort_values("min").to_csv(OUT / "measurement_dates_per_bearing.csv")
    s["n_distinct_dates"] = int(dts.date.nunique())
    s["dates_by_group"] = {g: sorted(v.date.unique().tolist()) for g, v in dts.groupby("group")}
    s["bearings_sharing_date_with_other_group"] = int(sum(dts[dts.date == d].group.nunique() > 1 for d in dts.date.unique()))
    s["year_by_group"] = pd.crosstab(dts.group, dts["min"].dt.year).to_dict()
    # 5. RMS by group and setting
    rms = f.groupby(["group", "setting"]).vib_rms.agg(["median", "min", "max"]).round(4)
    rms.to_csv(OUT / "rms_by_group_setting.csv")
    rb = f.groupby("bearing").vib_rms.median()
    s["rms_median_by_group"] = f.groupby("group").vib_rms.median().round(4).to_dict()
    s["temp_range_C"] = [float(f.temp_mean.min()), float(f.temp_mean.max())]
    (OUT / "pu_audit_summary.json").write_text(json.dumps(s, indent=2, default=str))
    # figures
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    order_b = dts.sort_values("min").index.tolist()
    for i, code in enumerate(order_b):
        sub = f[f.bearing == code]
        ax.scatter(sub.dt, np.full(len(sub), i), s=4, color=GCOL[dts.loc[code, "group"]])
    ax.set_yticks(range(len(order_b)), order_b, fontsize=6)
    ax.set_title("Paderborn: measurement timestamps per bearing (colour = group)", fontsize=8)
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in GCOL.items()]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig_pu_timeline.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    codes = sorted(rb.index, key=lambda c: (list(GCOL).index(b.set_index("bearing").group[c]), c))
    ax.bar(range(len(codes)), rb[codes], color=[GCOL[b.set_index("bearing").group[c]] for c in codes])
    ax.set_xticks(range(len(codes)), codes, rotation=90, fontsize=6)
    ax.set_ylabel("median RMS (stored units)")
    ax.set_title("Paderborn: vibration RMS per bearing", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pu_rms_per_bearing.png", dpi=300)
    plt.close(fig)
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
