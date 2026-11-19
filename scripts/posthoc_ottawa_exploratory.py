"""POST-HOC EXPLORATORY analysis of the Ottawa external result (after EV-V1 evaluation; does not replace it).

Questions: (1) are healthy-state false alarms concentrated in bearings that later developed inner-race faults, and is
BPFI evidence already present in their healthy recordings? (2) how do Ottawa healthy concept levels compare with PU
healthy levels (distribution shift of the evidence baseline)? (3) speed-reading vs dataset-wide distribution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pave.data import ottawa as OT  # noqa: E402
from pave.features import physics_v3 as V3  # noqa: E402

OUT = ROOT / "results/phase4/posthoc"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = np.load(ROOT / "results/phase4/ottawa_features.npz")
    seg = pd.read_csv(ROOT / "results/phase4/ottawa_segments4s.csv")
    C, M = F["C4"], F["M4"]
    for j, k in enumerate(V3.CONCEPTS_V3):
        seg[k] = np.where(M[:, j] > 0, C[:, j], np.nan)
    info = pd.DataFrame([OT.parse_stem(s) for s in seg.stem])
    seg["bearing_fault_type"] = info.bearing_fault_type
    seg["state_name"] = seg.state.map({0: "healthy", 1: "developing", 2: "faulty"})
    rec = seg.groupby(["stem", "bearing", "bearing_fault_type", "state_name"])[["BPFO_harm", "BPFI_harm", "contrast", "BPFI_sb", "rpm_sample0"]].mean().reset_index()
    rec.round(3).to_csv(OUT / "ottawa_recording_concepts.csv", index=False)
    tab = rec.groupby(["bearing_fault_type", "state_name"])[["BPFO_harm", "BPFI_harm", "contrast", "rpm_sample0"]].median().round(2)
    tab.to_csv(OUT / "concepts_by_group_state.csv")
    # PU healthy reference (FINAL calibration, training data)
    z = np.load(ROOT / "results/phase3b/folds/FINAL.npz")
    meas = pd.read_csv(ROOT / "results/phase3b/env4s/meas.csv")
    pu = meas.assign(**{k: np.where(z["meas_M4"][:, j] > 0, z["meas_C4"][:, j], np.nan) for j, k in enumerate(V3.CONCEPTS_V3)})
    puh = pu[pu.bearing.isin(["K001", "K002", "K003", "K004", "K005"])][["BPFO_harm", "BPFI_harm"]].describe().round(2)
    puh.to_csv(OUT / "pu_healthy_reference.csv")
    fin = json.loads((ROOT / "results/phase3b/folds/FINAL.json").read_text())
    pred = pd.read_csv(ROOT / "results/phase4/tables/external_recording_predictions.csv")
    hp = pred[(pred.model == "B1-V3-4s") & (pred.y == 0)].copy()
    hp["group_of_bearing"] = hp.bearing.map(lambda b: "IR-bearing (1-5)" if b <= 5 else "OR-bearing (6-10)" if b <= 10 else "ball/cage (11-20)")
    fa = hp.groupby("group_of_bearing").apply(lambda d: f"{int((d.pred != 0).sum())}/{len(d)}")
    out = {"healthy_false_alarms_by_later_fault_group_B1-V3-4s": fa.to_dict(), "B0_tau_dB_final": fin["b0_tau_dB"],
           "ottawa_concept_medians_by_group_state": tab.reset_index().to_dict("records"),
           "pu_healthy_BPFO_BPFI_harm": puh.to_dict()}
    (OUT / "posthoc_summary.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 200)
    print("POST-HOC EXPLORATORY")
    print(fa.to_string())
    print(tab.to_string())
    print(puh.to_string())
    print("B0 tau", fin["b0_tau_dB"])


if __name__ == "__main__":
    main()
