import numpy as np
import pytest

from pave.data.alignment import estimate_clock_offsets, overlap_fraction, robust_center


def test_session_offsets_robust_to_outliers():
    sessions = ["d1"] * 6 + ["d2"] * 4
    offsets = [203.0, 203.1, 202.9, 203.05, 1793.3, -1398.0, 197.3, 197.9, 197.2, 197.8]
    ok = [True, True, True, True, False, False, True, True, True, True]
    est = estimate_clock_offsets(sessions, offsets, ok)
    assert est["d1"].offset_s == pytest.approx(203.0, abs=0.1)
    assert est["d1"].n_pairs_used == 4
    assert est["d2"].offset_s == pytest.approx(197.55, abs=0.1)


def test_label_consistent_outlier_is_excluded():
    est = estimate_clock_offsets(["d"] * 5, [206.7, 206.6, 206.8, 206.7, -1709.0], [True] * 5)
    assert est["d"].n_pairs_used == 4


def test_overlap_fraction():
    assert overlap_fraction(0, 60, 0, 60) == 1.0
    assert overlap_fraction(0, 60, 30, 60) == 0.5
    assert overlap_fraction(0, 60, 120, 60) == 0.0


def test_robust_center():
    m, s = robust_center(np.array([1.0, 1.0, 1.0, 100.0]))
    assert m == 1.0 and s == 0.0
