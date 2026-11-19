import numpy as np

from pave.signal.quality import channel_stats, clipping_fraction, longest_constant_run, segment_stats


def test_gaussian_stats():
    x = np.random.default_rng(0).standard_normal(200000)
    s = channel_stats(x)
    assert abs(s["kurtosis"] - 3.0) < 0.1
    assert abs(s["rms"] - 1.0) < 0.01
    assert not s["clipping_suspect"] and not s["is_constant"]
    assert s["n_nan"] == 0


def test_clipping_detected():
    x = np.random.default_rng(0).standard_normal(100000) * 3
    assert clipping_fraction(np.clip(x, -2, 2))["clipping_suspect"]
    assert not clipping_fraction(x)["clipping_suspect"]


def test_constant_and_runs():
    assert longest_constant_run(np.array([1, 1, 1, 2, 2, 3])) == 3
    assert channel_stats(np.ones(1000))["is_constant"]


def test_nan_counted():
    x = np.random.default_rng(0).standard_normal(1000)
    x[5] = np.nan
    x[7] = np.inf
    s = channel_stats(x)
    assert s["n_nan"] == 1 and s["n_inf"] == 1 and np.isfinite(s["rms"])


def test_segment_stats_shape():
    s = segment_stats(np.random.default_rng(0).standard_normal(25600 * 3 + 10), 25600, 1.0)
    assert s["rms"].shape == (3,)
