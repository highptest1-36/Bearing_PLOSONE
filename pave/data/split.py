"""Leakage-safe split construction at the RECORDING level, before windowing.

Protocol A (primary): leave-one-load-out (LOLO). For held-out load L, every
in-scope recording with load L is TEST; recordings at the other two loads are
SOURCE. Source recordings are divided in time into a TRAIN interval and a
VALIDATION interval (last ``val_frac`` of each recording) separated by a purge
gap >= the longest window length. Validation is therefore NOT independent of
training (same physical recording); this is disclosed in every split file.

Protocol B (secondary): for each ordered (train_load, val_load, test_load)
permutation, whole recordings are assigned to one role each.

Protocol C (supplementary, in-domain only): within-recording temporal holdout.

All intervals are in seconds relative to the recording start. Windowing code
must only cut windows fully inside an interval.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, other: "Interval") -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s

    def gap_to(self, other: "Interval") -> float:
        if self.overlaps(other):
            return -1.0
        return max(other.start_s - self.end_s, self.start_s - other.end_s)


@dataclass
class Assignment:
    recording_id: str
    role: str  # 'train' | 'val' | 'test'
    interval: Interval


@dataclass
class Split:
    protocol: str
    fold_id: str
    description: str
    assignments: list[Assignment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def roles(self, role: str) -> list[Assignment]:
        return [a for a in self.assignments if a.role == role]

    def recordings(self, role: str) -> set[str]:
        return {a.recording_id for a in self.assignments if a.role == role}

    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol, "fold_id": self.fold_id, "description": self.description,
            "notes": self.notes,
            "assignments": [{"recording_id": a.recording_id, "role": a.role,
                             "start_s": a.interval.start_s, "end_s": a.interval.end_s}
                            for a in self.assignments],
        }

    def content_hash(self) -> str:
        d = self.to_dict()
        payload = json.dumps(d["assignments"], sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class Rec:
    recording_id: str
    load_Nm: int
    duration_s: float


def lolo_splits(recs: list[Rec], val_frac: float = 0.2, purge_s: float = 2.0) -> list[Split]:
    loads = sorted({r.load_Nm for r in recs})
    out = []
    for test_load in loads:
        sp = Split("A_LOLO", f"A_test{test_load}Nm",
                   f"Leave-one-load-out: test={test_load} Nm, sources={[l for l in loads if l != test_load]} Nm")
        for r in sorted(recs, key=lambda r: r.recording_id):
            if r.load_Nm == test_load:
                sp.assignments.append(Assignment(r.recording_id, "test", Interval(0.0, r.duration_s)))
            else:
                v0 = r.duration_s * (1.0 - val_frac)
                sp.assignments.append(Assignment(r.recording_id, "train", Interval(0.0, v0 - purge_s)))
                sp.assignments.append(Assignment(r.recording_id, "val", Interval(v0, r.duration_s)))
        sp.notes += [
            f"Validation = last {val_frac:.0%} of each SOURCE recording; purge gap {purge_s} s >= longest window.",
            "Validation windows come from the same physical recordings as training windows: NOT independent.",
            "Test recordings are never used for normalisation, early stopping, HPO, calibration or thresholds.",
            "One recording per condition per load; no independent bearing-level replication exists.",
        ]
        out.append(sp)
    return out


def permutation_splits(recs: list[Rec]) -> list[Split]:
    loads = sorted({r.load_Nm for r in recs})
    out = []
    for tr, va, te in itertools.permutations(loads, 3):
        sp = Split("B_PERM", f"B_tr{tr}_va{va}_te{te}",
                   f"Single-source transfer: train={tr} Nm, validation={va} Nm, test={te} Nm")
        for r in sorted(recs, key=lambda r: r.recording_id):
            role = {tr: "train", va: "val", te: "test"}[r.load_Nm]
            sp.assignments.append(Assignment(r.recording_id, role, Interval(0.0, r.duration_s)))
        sp.notes.append("Whole recordings per role; validation load is a different domain from both train and test.")
        out.append(sp)
    return out


def within_recording_split(recs: list[Rec], train_frac: float = 0.6, val_frac: float = 0.2,
                           purge_s: float = 2.0) -> Split:
    sp = Split("C_WITHIN", "C_temporal",
               "Within-recording contiguous temporal holdout (in-domain diagnostic only)")
    for r in sorted(recs, key=lambda r: r.recording_id):
        t1 = r.duration_s * train_frac
        t2 = r.duration_s * (train_frac + val_frac)
        sp.assignments += [
            Assignment(r.recording_id, "train", Interval(0.0, t1 - purge_s)),
            Assignment(r.recording_id, "val", Interval(t1, t2 - purge_s)),
            Assignment(r.recording_id, "test", Interval(t2, r.duration_s)),
        ]
    sp.notes += ["Test intervals come from the SAME recordings as training: never evidence of cross-recording "
                 "or cross-load generalisation.", f"Purge gap {purge_s} s between consecutive roles."]
    return sp


class LeakageError(AssertionError):
    pass


def check_split(sp: Split, max_window_s: float = 2.0) -> None:
    """Raise ``LeakageError`` on any recording/time leakage."""
    test = sp.recordings("test")
    train = sp.recordings("train")
    val = sp.recordings("val")
    if sp.protocol in ("A_LOLO", "B_PERM"):
        if test & (train | val):
            raise LeakageError(f"{sp.fold_id}: test recordings also in train/val: {sorted(test & (train | val))}")
    if sp.protocol == "B_PERM" and train & val:
        raise LeakageError(f"{sp.fold_id}: recordings shared by train and val")
    by_rec: dict[str, list[Assignment]] = {}
    for a in sp.assignments:
        if a.interval.duration_s <= 0:
            raise LeakageError(f"{sp.fold_id}: empty/negative interval for {a.recording_id}/{a.role}")
        by_rec.setdefault(a.recording_id, []).append(a)
    for rid, items in by_rec.items():
        for a, b in itertools.combinations(items, 2):
            if a.role == b.role:
                continue
            g = a.interval.gap_to(b.interval)
            if g < 0:
                raise LeakageError(f"{sp.fold_id}: {rid} intervals overlap between {a.role} and {b.role}")
            if g < max_window_s - 1e-9:
                raise LeakageError(f"{sp.fold_id}: {rid} purge gap {g:.3f}s < max window {max_window_s}s")


def windows_in_interval(iv: Interval, win_s: float, hop_s: float | None = None) -> list[Interval]:
    """Non-overlapping (default) windows entirely contained in ``iv``."""
    hop = win_s if hop_s is None else hop_s
    out = []
    t = iv.start_s
    while t + win_s <= iv.end_s + 1e-9:
        out.append(Interval(t, t + win_s))
        t += hop
    return out


def source_only_stats(split: Split, stats_by_rec_interval) -> dict:
    """Compute normalisation statistics from TRAIN assignments only.

    ``stats_by_rec_interval(recording_id, interval) -> (sum, sumsq, count)``.
    Raises if called with a split lacking train assignments.
    """
    tr = split.roles("train")
    if not tr:
        raise LeakageError("no train assignments")
    S = SS = N = 0.0
    used = []
    for a in tr:
        s, ss, n = stats_by_rec_interval(a.recording_id, a.interval)
        S, SS, N = S + s, SS + ss, N + n
        used.append(a.recording_id)
    mean = S / N
    var = SS / N - mean ** 2
    return {"mean": mean, "std": var ** 0.5, "n": N, "recordings_used": sorted(set(used))}


def split_to_json(sp: Split) -> str:
    d = sp.to_dict()
    d["content_sha256"] = sp.content_hash()
    return json.dumps(d, indent=2)


__all__ = [n for n in dir() if not n.startswith("_")] + ["asdict"]
