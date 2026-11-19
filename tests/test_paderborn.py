"""Paderborn label table, filename parsing and bearing-disjoint split tests."""
import json
from pathlib import Path

import pytest

from pave.data.paderborn import BD5_ROWS, BEARINGS, parse_fact_sheet_text, parse_pu_filename
from pave.experiments.phase2 import PU_A2R, PU_CHALLENGE, pu_bd5_sets

SPLITS = Path(__file__).resolve().parents[1] / "03_MANIFESTS" / "paderborn" / "splits"


def test_label_table_matches_paper_counts():
    groups = {}
    for b in BEARINGS.values():
        groups[b["group"]] = groups.get(b["group"], 0) + 1
    assert len(BEARINGS) == 32
    assert groups == {"healthy": 6, "OR_artificial": 7, "IR_artificial": 5, "OR_real": 5, "IR_real": 6, "multiple_real": 3}
    assert all(BEARINGS[c]["label3"] is None for c in PU_CHALLENGE)


def test_bd5_rows_are_paper_table10():
    for h, o, i in BD5_ROWS:
        assert BEARINGS[h]["label3"] == "H" and BEARINGS[o]["group"] == "OR_real" and BEARINGS[i]["group"] == "IR_real"


def test_bd5_folds_bearing_disjoint_and_each_row_tested_once():
    folds = pu_bd5_sets()
    tested = []
    for s in folds:
        assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"])
        assert len(s["train"]) == 9 and len(s["val"]) == 3 and len(s["test"]) == 3
        for role in ("val", "test"):
            assert sorted(BEARINGS[b]["label3"] for b in s[role]) == ["H", "IR", "OR"]
        tested += sorted(s["test"])
    assert len(tested) == 15 and len(set(tested)) == 15


def test_a2r_artificial_train_real_test():
    assert not (PU_A2R["train"] & PU_A2R["test"] or PU_A2R["val"] & PU_A2R["test"] or PU_A2R["train"] & PU_A2R["val"])
    assert all("artificial" in BEARINGS[b]["group"] or BEARINGS[b]["group"] == "healthy" for b in PU_A2R["train"] | PU_A2R["val"])
    assert all(BEARINGS[b]["group"] in ("OR_real", "IR_real") or b == "K001" for b in PU_A2R["test"])
    assert not (PU_CHALLENGE & (PU_A2R["train"] | PU_A2R["val"] | PU_A2R["test"]))


def test_filename_parse():
    assert parse_pu_filename("N09_M07_F10_KI18_20") == ("N09_M07_F10", "KI18", 20)
    with pytest.raises(ValueError):
        parse_pu_filename("N11_M07_F10_K001_1")
    with pytest.raises(ValueError):
        parse_pu_filename("N15_M07_F10_KX99_1")


def test_fact_sheet_text_parser():
    txt = "Pitch circle diameter mm 29.05\nNumber of rolling elements pc. 8\nRolling element diameter mm 6.75\n" \
          "Nominal pressure angle ° 0\nManufacturer - IBU\n"
    d = parse_fact_sheet_text(txt)
    assert d == {"pitch_diameter_mm": 29.05, "n_balls": 8.0, "ball_diameter_mm": 6.75, "contact_angle_deg": 0.0, "manufacturer": "IBU"}


@pytest.mark.skipif(not SPLITS.exists(), reason="run scripts/build_paderborn_splits.py")
def test_written_split_manifests():
    import hashlib
    files = sorted(SPLITS.glob("split_PU_*.json"))
    assert len(files) == 6
    for f in files:
        d = json.loads(f.read_text())
        h = hashlib.sha256(json.dumps(d["assignments"], sort_keys=True).encode()).hexdigest()
        assert h == d["content_sha256"]
        roles = {}
        for a in d["assignments"]:
            assert a["bearing"] not in roles, "bearing in two roles"
            roles[a["bearing"]] = a["role"]


def test_ottawa_mapping():
    from pave.data.ottawa import label_mapping, parse_stem
    assert label_mapping(parse_stem("H_3_0"))["primary"] == "H"
    assert label_mapping(parse_stem("I_2_2"))["primary"] == "IR"
    assert label_mapping(parse_stem("O_7_2"))["primary"] == "OR"
    d = label_mapping(parse_stem("I_2_1"))
    assert d["primary"] is None and d["role"] == "developing" and d["developing_label"] == "IR"
    assert label_mapping(parse_stem("B_12_2"))["role"] == "ood_faulty"
    assert label_mapping(parse_stem("C_17_1"))["role"] == "ood_developing"
    assert parse_stem("H_14_0")["manufacturer"] == "NSK 6203ZZ"
    with pytest.raises(ValueError):
        parse_stem("I_7_2")   # bearing 7 is an outer-race bearing
    with pytest.raises(ValueError):
        parse_stem("H_3_2")
