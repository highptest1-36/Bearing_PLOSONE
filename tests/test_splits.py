import pytest

from pave.data.split import (Assignment, Interval, LeakageError, Rec, check_split, lolo_splits,
                             permutation_splits, source_only_stats, windows_in_interval,
                             within_recording_split)

CLASSES = ["H", "IR_0p3", "IR_1p0", "IR_3p0", "OR_0p3", "OR_1p0", "OR_3p0"]
RECS = [Rec(f"L{l}_{c}", l, 300.0 if (l == 0 and c == "H") else (120.0 if c == "H" else 60.0))
        for l in (0, 2, 4) for c in CLASSES]


def test_lolo_three_folds_disjoint():
    sps = lolo_splits(RECS)
    assert [s.fold_id for s in sps] == ["A_test0Nm", "A_test2Nm", "A_test4Nm"]
    for sp in sps:
        check_split(sp, max_window_s=2.0)
        test = sp.recordings("test")
        assert len(test) == 7
        assert not (test & sp.recordings("train")) and not (test & sp.recordings("val"))
        load = int(sp.fold_id[len("A_test")])
        assert all(r.startswith(f"L{load}_") for r in test)
    tests = [r for sp in sps for r in sp.recordings("test")]
    assert sorted(tests) == sorted(r.recording_id for r in RECS)  # each recording tested exactly once


def test_lolo_purge_and_val_fraction():
    sp = lolo_splits(RECS, val_frac=0.2, purge_s=2.0)[0]
    a = {(x.recording_id, x.role): x.interval for x in sp.assignments}
    tr, va = a[("L2_IR_0p3", "train")], a[("L2_IR_0p3", "val")]
    assert va.start_s == pytest.approx(48.0) and va.end_s == pytest.approx(60.0)
    assert tr.end_s == pytest.approx(46.0)
    assert tr.gap_to(va) == pytest.approx(2.0)


def test_insufficient_purge_raises():
    with pytest.raises(LeakageError):
        check_split(lolo_splits(RECS, purge_s=1.0)[0], max_window_s=2.0)


def test_recording_leak_raises():
    sp = lolo_splits(RECS)[0]
    sp.assignments.append(Assignment("L0_H", "train", Interval(0, 10)))
    with pytest.raises(LeakageError):
        check_split(sp)


def test_permutations():
    sps = permutation_splits(RECS)
    assert len(sps) == 6
    for sp in sps:
        check_split(sp)
        assert len(sp.recordings("train")) == len(sp.recordings("val")) == len(sp.recordings("test")) == 7


def test_within_recording_split_is_in_domain_only():
    sp = within_recording_split(RECS)
    check_split(sp)
    assert sp.recordings("test") == sp.recordings("train")


def test_windows_inside_interval():
    w = windows_in_interval(Interval(0, 46.0), 2.0)
    assert len(w) == 23 and w[-1].end_s <= 46.0
    assert all(a.end_s <= b.start_s for a, b in zip(w, w[1:]))


def test_source_only_stats_never_touch_test():
    sp = lolo_splits(RECS)[1]
    seen = []

    def fn(rid, iv):
        seen.append(rid)
        return 1.0, 1.0, 1.0

    st = source_only_stats(sp, fn)
    assert not (set(seen) & sp.recordings("test"))
    assert set(st["recordings_used"]) == sp.recordings("train")


def test_split_hash_stable_and_sensitive():
    a, b = lolo_splits(RECS)[0], lolo_splits(RECS)[0]
    assert a.content_hash() == b.content_hash()
    b.assignments[0] = Assignment(b.assignments[0].recording_id, "val", b.assignments[0].interval)
    assert a.content_hash() != b.content_hash()
