"""Integrity tests on the REAL generated manifests (skipped if the audit has not been run)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from pave.data.split import Assignment, Interval, Split, check_split

MAN = Path(__file__).resolve().parents[1] / "03_MANIFESTS"
pytestmark = pytest.mark.skipif(not (MAN / "recordings.csv").exists(), reason="run scripts/audit_data.py first")


def _load_split(p: Path) -> Split:
    d = json.loads(p.read_text())
    sp = Split(d["protocol"], d["fold_id"], d["description"])
    sp.assignments = [Assignment(a["recording_id"], a["role"], Interval(a["start_s"], a["end_s"])) for a in d["assignments"]]
    assert sp.content_hash() == d["content_sha256"], f"{p.name}: content hash mismatch (file edited?)"
    return sp


def test_recordings_manifest():
    r = pd.read_csv(MAN / "recordings.csv")
    assert len(r) == 45 and r.recording_id.is_unique
    ins = r[r.in_bearing_scope]
    assert len(ins) == 21
    assert sorted(ins.label7.value_counts().to_dict().values()) == [3] * 7
    inv = pd.read_csv(MAN / "file_inventory.csv")
    sha = dict(zip(inv.relative_path, inv.sha256))
    assert all(sha[p] == h for p, h in zip(r.vib_relative_path, r.vib_sha256))
    assert all(sha[p] == h for p, h in zip(r.tdms_relative_path, r.tdms_sha256))
    assert inv.crc_match_archive.dropna().all()


def test_pairing_resolution():
    p = pd.read_csv(MAN / "pairs.csv").set_index("recording_id")
    assert p.loc["L0_IR_0p3"].resolved_tdms == "0Nm_BPFI_10.tdms"
    assert p.loc["L0_IR_1p0"].resolved_tdms == "0Nm_BPFI_03.tdms"
    assert set(p[p.pairing_status == "NON_CONCURRENT"].index) == {"L0_OR_0p3", "L2_OR_0p3", "L4_OR_0p3"}
    conc = p[p.pairing_status != "NON_CONCURRENT"]
    assert conc.resolved_residual_s.abs().max() < 5.0
    assert conc.speed_mismatch_hz.abs().max() < 0.002


def test_real_splits_leakage_free():
    files = sorted((MAN / "splits").glob("split_*.json"))
    assert len(files) == 10
    tested = []
    for f in files:
        sp = _load_split(f)
        check_split(sp, max_window_s=2.0)
        if sp.protocol == "A_LOLO":
            load = int(sp.fold_id[len("A_test")])
            assert all(rid.startswith(f"L{load}_") for rid in sp.recordings("test"))
            assert not any(rid.startswith(f"L{load}_") for rid in sp.recordings("train") | sp.recordings("val"))
            tested += sorted(sp.recordings("test"))
    assert len(tested) == 21 and len(set(tested)) == 21


def test_challenge_set_disjoint_from_task():
    c = pd.read_csv(MAN / "splits" / "challenge_set_out_of_scope.csv")
    r = pd.read_csv(MAN / "recordings.csv")
    assert set(c.recording_id).isdisjoint(set(r[r.in_bearing_scope].recording_id))
    assert len(c) == 24
