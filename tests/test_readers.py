"""Reader tests on SYNTHETIC fixtures that mimic the verified on-disk schemas."""
import datetime as dt

import numpy as np
import pytest
import scipy.io as sio

from pave.data.read_mat import parse_testlab_time, read_vibration_mat
from pave.data.read_tdms import read_tdms


def _cell(items):
    a = np.empty((1, len(items)), dtype=object)
    for i, v in enumerate(items):
        a[0, i] = v
    return a


@pytest.fixture
def fake_mat(tmp_path):
    n, fs = 2560, 25600.0
    vals = np.random.default_rng(0).standard_normal((n, 4))
    q = {"label": "g", "unit_transformation": {"offset": 0.0, "factor": 0.101971621297793, "log_reference": 1.0},
         "info": "Data saved by Test.Lab in MKS"}
    sig = {
        "x_values": {"start_value": 1e-5, "increment": 1 / fs, "number_of_values": n,
                     "quantity": {"label": "s", "info": "x"}},
        "y_values": {"values": vals, "quantity": q},
        "function_record": {
            "name": _cell([f"{i + 3}:Point{i + 1}" for i in range(4)]),
            "creation_time": _cell(["2021/06/28 20:32:38"] * 4),
            "primary_channel": {"label": _cell([f"Point{i + 1}" for i in range(4)])},
            "TL_export_properties_annotation": {
                "run_name": _cell(["210625_0Nm_Inner_0.3mm"] * 4),
                "absolute_time": _cell(["2021-06-28 11:32:36 ms 799.212"] * 4),
                "orig_location": _cell(["C:\\x"] * 4),
                "channel_group": _cell(["Vibration"] * 4),
            },
        },
    }
    p = tmp_path / "0Nm_BPFI_03.mat"
    sio.savemat(p, {"Signal": sig})
    return p, vals


def test_read_mat_schema(fake_mat):
    p, vals = fake_mat
    r = read_vibration_mat(p)
    assert r.values.shape == (2560, 4)
    np.testing.assert_allclose(r.values, vals)
    assert r.fs_hz == pytest.approx(25600.0)
    assert r.channel_labels == ["Point1", "Point2", "Point3", "Point4"]
    assert r.channel_names[0] == "x_housing_A"
    assert r.unit_label == "g" and r.unit_factor == pytest.approx(1 / 9.80665)
    assert r.run_name == "210625_0Nm_Inner_0.3mm"
    assert r.absolute_time == dt.datetime(2021, 6, 28, 11, 32, 36, 799212)
    assert r.duration_s == pytest.approx(0.1)


def test_parse_testlab_time():
    assert parse_testlab_time("2021-06-30 07:19:50 ms 142.499").microsecond == 142499
    assert parse_testlab_time("garbage") is None


@pytest.fixture
def fake_tdms(tmp_path):
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter
    n = 20000
    temp = np.repeat(np.linspace(30, 30.5, n // 5000), 5000).astype(float)
    start = np.datetime64("2021-06-28T11:02:43.489671")
    tp = {"wf_increment": 3.905e-05, "wf_start_time": start, "unit_string": "C",
          "DAC~Device~Name": "NI 9210 (miniTC)", "NI_ThermocoupleType": "K"}
    cp = {"wf_increment": 3.905e-05, "wf_start_time": start, "unit_string": "A", "DAC~Device~Name": "NI 9775"}
    p = tmp_path / "x.tdms"
    with TdmsWriter(str(p)) as w:
        w.write_segment([
            RootObject({"name": "LogFile_0Nm_Inner_1mm", "RunNumber": 43}),
            GroupObject("Log"),
            ChannelObject("Log", "cDAQ9185-1F486B5Mod1/ai0", temp, properties=tp),
            ChannelObject("Log", "cDAQ9185-1F486B5Mod1/ai1", temp + 0.5, properties=tp),
            ChannelObject("Log", "cDAQ9185-1F486B5Mod2/ai0", np.sin(np.arange(n) / 10.0), properties=cp),
        ])
    return p


def test_read_tdms(fake_tdms):
    r = read_tdms(fake_tdms, load=("temperature",))
    assert r.file_name_prop == "LogFile_0Nm_Inner_1mm" and r.run_number == 43
    names = {c.name: c for c in r.channels}
    assert names["T_housing_A"].kind == "temperature" and names["I_U"].kind == "current"
    assert "I_U" not in r.data  # only requested kinds are loaded
    assert r.data["T_housing_B"].mean() - r.data["T_housing_A"].mean() == pytest.approx(0.5)
    assert names["T_housing_A"].start_time_utc == dt.datetime(2021, 6, 28, 11, 2, 43, 489671)


def test_held_values_and_effective_rate():
    from pave.audit.tdms_audit import held_values, temperature_stats
    idx, v = held_values(np.repeat([1.0, 2.0, 3.0], 10))
    assert list(idx) == [0, 10, 20] and list(v) == [1, 2, 3]
    s = temperature_stats(np.repeat(np.arange(100, dtype=float) * 0.01, 50), 0.01)
    assert s["n_updates"] == 100 and s["effective_rate_hz"] == pytest.approx(2.0)
