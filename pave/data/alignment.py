"""Vibration <-> temperature/current pairing and synchronisation analysis.

Two independent acquisition systems are involved:
  * Siemens SCADAS / Simcenter Test.Lab (vibration; ``absolute_time`` in .mat)
  * NI cDAQ-9185 / FlexLogger (temperature + current; ``wf_start_time`` in .tdms)

Their clocks are not stated to be synchronised. This module never assumes
pairing from filenames alone. It provides:

1. ``raw_offsets``: start-time difference (vib - tdms) for candidate pairs.
2. ``estimate_clock_offsets``: robust per-session (calendar-day) clock offset
   from label-consistent pairs, with residuals; outliers flagged.
3. ``speed_identity``: compares shaft speed measured from vibration with shaft
   speed inferred from motor-current eccentricity sidebands (independent
   physical channel) to test whether two files come from the same operating run.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np


def offset_seconds(vib_start: dt.datetime, tdms_start: dt.datetime) -> float:
    return (vib_start - tdms_start).total_seconds()


def robust_center(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based sigma."""
    v = np.asarray(values, dtype=float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) * 1.4826)
    return med, mad


@dataclass
class SessionOffset:
    session: str
    offset_s: float
    sigma_s: float
    n_pairs_used: int
    n_pairs_total: int


def estimate_clock_offsets(sessions: list[str], offsets: list[float], label_ok: list[bool],
                           max_dev_s: float = 10.0) -> dict[str, SessionOffset]:
    """Per-session robust offset from pairs whose internal labels agree.

    Pairs deviating by more than ``max_dev_s`` from the session median are
    excluded from the estimate (and reported as outliers by the caller).
    """
    out: dict[str, SessionOffset] = {}
    S = np.asarray(sessions)
    O = np.asarray(offsets, dtype=float)
    L = np.asarray(label_ok, dtype=bool)
    for s in sorted(set(sessions)):
        m = (S == s) & L
        if m.sum() == 0:
            continue
        med, _ = robust_center(O[m])
        keep = m & (np.abs(O - med) <= max_dev_s)
        med2, sig2 = robust_center(O[keep])
        out[s] = SessionOffset(s, med2, sig2, int(keep.sum()), int((S == s).sum()))
    return out


def overlap_fraction(vib_start_s: float, vib_dur_s: float, tdms_start_s: float, tdms_dur_s: float) -> float:
    """Fraction of the vibration interval covered by the TDMS interval (common time base)."""
    a0, a1 = vib_start_s, vib_start_s + vib_dur_s
    b0, b1 = tdms_start_s, tdms_start_s + tdms_dur_s
    ov = max(0.0, min(a1, b1) - max(a0, b0))
    return ov / vib_dur_s if vib_dur_s > 0 else 0.0


def motor_shaft_from_current(supply_hz: float, upper_sideband_hz: float) -> float:
    """Motor-shaft rotation frequency from eccentricity sideband f_s + f_rm."""
    return upper_sideband_hz - supply_hz
