"""Phase 3A-B: Paderborn metadata verification vs Lessmeier et al. 2016 (Tables 4, 5, 7, 8, 10) and the
implemented protocols. Reads only existing manifests/caches; writes results/phase3a/pu_verification/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from pave.data.paderborn import BD5_ROWS, BEARINGS  # noqa: E402
from pave.experiments.phase2 import PU_A2R, PU_CHALLENGE, pu_bd5_sets  # noqa: E402
from pave.experiments.phase3a import inner_folds, outer_folds  # noqa: E402

OUT = ROOT / "results" / "phase3a" / "pu_verification"

# Transcribed from Lessmeier et al. 2016 (local PDF, p. 11), Table 8 (artificial -> real) and Table 10 (real 5x3)
LESSMEIER_T8 = {"train": {"K002", "KA01", "KA05", "KA07", "KI01", "KI05", "KI07"},
                "test": {"K001", "KA22", "KA04", "KA15", "KA30", "KA16", "KI14", "KI21", "KI17", "KI18", "KI16"}}
LESSMEIER_T10 = {"H": ["K001", "K002", "K003", "K004", "K005"], "OR": ["KA04", "KA15", "KA16", "KA22", "KA30"],
                 "IR": ["KI04", "KI14", "KI16", "KI18", "KI21"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bear = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_bearings.csv").set_index("bearing")
    files = pd.read_csv(ROOT / "03_MANIFESTS/paderborn/pu_files.csv")
    meta = pd.read_csv(ROOT / "results/phase2/windows/pu/meta.csv")
    ometa = pd.read_csv(ROOT / "results/phase2/windows/pu/order_meta.csv")
    dates = pd.read_csv(ROOT / "results/phase2/pu_audit/measurement_dates_per_bearing.csv", index_col=0)
    bd5 = pu_bd5_sets()
    rows = []
    for code in sorted(BEARINGS):
        b = BEARINGS[code]
        f = files[files.bearing == code]
        origin = "healthy" if b["group"] == "healthy" else ("artificial" if "artificial" in b["group"] else "real")
        r = {"bearing": code, "group": b["group"], "label3": b["label3"] or "(multiple: IR+OR)", "origin": origin,
             "component": b["component"], "method": b["method"], "extent_level": b["extent"],
             "combination": b["combination"], "characteristic": b["characteristic"],
             "manufacturer": bear.fs_manufacturer[code], "pitch_diameter_mm": bear.geom_pitch_diameter_mm[code],
             "first_measurement": dates.loc[code, "min"], "year": str(dates.loc[code, "min"])[:4],
             "n_files": len(f), "n_unreadable": int(f.read_error.notna().sum()) if "read_error" in f else 0,
             "n_windows_1s": int((meta.bearing == code).sum()), "n_order_windows": int((ometa.bearing == code).sum()),
             "in_lessmeier_T10": any(code in v for v in LESSMEIER_T10.values()),
             "lessmeier_T8_role": "train" if code in LESSMEIER_T8["train"] else ("test" if code in LESSMEIER_T8["test"] else "-"),
             "PU_A2R_role": next((k for k in ("train", "val", "test") if code in PU_A2R[k]), "challenge" if code in PU_CHALLENGE else "-")}
        for s in bd5:
            r[s["fold"]] = next((k for k in ("train", "val", "test") if code in s[k]), "-")
        r["BD5_times_tested"] = sum(r[s["fold"]] == "test" for s in bd5)
        excl = []
        if code in PU_CHALLENGE:
            excl.append("multiple damage IR+OR: no single 3-class label (challenge only)")
        if code in ("K006", "KI17"):
            excl.append("not in Lessmeier Table 10 -> not in PU-BD5")
        if b["group"].endswith("artificial"):
            excl.append("artificial damage: not in PU-BD5 (used in PU-A2R)")
        if r["n_unreadable"]:
            excl.append(f"{r['n_unreadable']} unreadable file(s) excluded")
        r["exclusions_notes"] = "; ".join(excl)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "pu_32_bearings.csv", index=False)
    # counts
    counts = {"by_group": T.group.value_counts().to_dict(), "by_origin": T.origin.value_counts().to_dict(),
              "by_label3": T.label3.value_counts().to_dict(),
              "manufacturer_x_group": pd.crosstab(T.group, T.manufacturer).to_dict(),
              "year_x_group": pd.crosstab(T.group, T.year).to_dict()}
    # per fold tables (V2 and V2.1 nested)
    fold_rows = []
    for o in outer_folds():
        fold_rows.append({"protocol": "V2.1 outer", "fold": o["fold"], "role": "test", "bearings": " ".join(sorted(o["test"]))})
        fold_rows.append({"protocol": "V2.1 outer", "fold": o["fold"], "role": "train(4 rows)", "bearings": " ".join(sorted(o["train"]))})
        for i in inner_folds(o):
            fold_rows.append({"protocol": "V2.1 inner", "fold": i["fold"], "role": "inner-val", "bearings": " ".join(sorted(i["val"]))})
            fold_rows.append({"protocol": "V2.1 inner", "fold": i["fold"], "role": "inner-train", "bearings": " ".join(sorted(i["train"]))})
    for s in bd5:
        for k in ("train", "val", "test"):
            fold_rows.append({"protocol": "V2 PU-BD5", "fold": s["fold"], "role": k, "bearings": " ".join(sorted(s[k]))})
    for k in ("train", "val", "test"):
        fold_rows.append({"protocol": "V2 PU-A2R", "fold": "A2R", "role": k, "bearings": " ".join(sorted(PU_A2R[k]))})
    F = pd.DataFrame(fold_rows)
    F["n_bearings"] = F.bearings.str.split().str.len()
    F["n_H"] = F.bearings.apply(lambda s: sum(BEARINGS[b]["label3"] == "H" for b in s.split()))
    F["n_IR"] = F.bearings.apply(lambda s: sum(BEARINGS[b]["label3"] == "IR" for b in s.split()))
    F["n_OR"] = F.bearings.apply(lambda s: sum(BEARINGS[b]["label3"] == "OR" for b in s.split()))
    F["n_windows_1s"] = F.bearings.apply(lambda s: int(meta.bearing.isin(s.split()).sum()))
    F.to_csv(OUT / "fold_composition.csv", index=False)
    # comparison with the paper
    cmp = {
        "table10_members_equal_BD5_members": sorted(sum(LESSMEIER_T10.values(), [])) == sorted(set().union(*[set(r) for r in BD5_ROWS])),
        "table10_rows_equal_BD5_rows": [list(r) for r in BD5_ROWS] == [[LESSMEIER_T10["H"][i], LESSMEIER_T10["OR"][i], LESSMEIER_T10["IR"][i]] for i in range(5)],
        "paper_table10_protocol": "5-fold-style: 'three data sets from each class were used for training and the other two for testing, resulting in 10 combinations' (C(5,3)=10); no validation set described",
        "implemented_PU_BD5": "5 folds; test = 1 row, validation = next row, train = 3 rows; each bearing tested once",
        "implemented_V2.1": "outer 5 folds (test = 1 row, train = 4 rows); inner 4 folds leave-one-row-out inside the 4 training rows",
        "A2R_vs_table8": {"train_in_A2R_not_T8": sorted(PU_A2R["train"] - LESSMEIER_T8["train"]),
                          "T8_train_not_in_A2R_train": sorted(LESSMEIER_T8["train"] - PU_A2R["train"]),
                          "test_in_A2R_not_T8": sorted(PU_A2R["test"] - LESSMEIER_T8["test"]),
                          "T8_test_not_in_A2R": sorted(LESSMEIER_T8["test"] - PU_A2R["test"])},
        "row_pairing_note": "BD5 rows pair the i-th listed bearing of each class in Table 10; the paper does not state that rows are meaningful units. The 10-combination count implies the paper also moved bearings in row-coupled sets (otherwise 10^3 combinations).",
    }
    (OUT / "verification_summary.json").write_text(json.dumps({"counts": counts, "comparison": cmp}, indent=2, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(T[["bearing", "group", "origin", "manufacturer", "year", "n_files", "n_windows_1s", "lessmeier_T8_role", "PU_A2R_role",
             "BD5_f1", "BD5_f2", "BD5_f3", "BD5_f4", "BD5_f5", "BD5_times_tested"]].to_string())
    print(json.dumps(cmp, indent=1))
    print(F[F.protocol != "V2 PU-BD5"].head(12).to_string())


if __name__ == "__main__":
    main()
