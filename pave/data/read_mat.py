"""Reader for KAIST varying-load vibration ``.mat`` files.

Verified schema (inspected 2026-11-18, all files are MATLAB 5.0 MAT-files written
by Siemens Simcenter Test.Lab export):

    Signal.x_values.{start_value, increment, number_of_values, quantity}
    Signal.y_values.values            -> float64 array (N, 4)
    Signal.y_values.quantity.label    -> 'g'
    Signal.y_values.quantity.unit_transformation.factor -> 0.10197162 (=1/9.80665)
    Signal.y_values.quantity.info     -> 'Data saved by Test.Lab in MKS'
    Signal.function_record.{name, primary_channel.label ('Point1'..'Point4'),
        TL_export_properties_annotation.{run_name, absolute_time, orig_location}}

Units: Test.Lab stores ``values`` in the MKS base unit (m/s^2) and attaches a
display transformation ``display = factor * value + offset`` to the user unit
label ('g'). The paper states the data are in g. ``unit_audit`` in the audit
pipeline tests which interpretation is consistent with the raw numbers; this
reader returns the raw stored values unchanged and reports both conventions.

Channel mapping: the files label channels ``Point1..Point4``. The paper lists
columns ``x_direction_housing_A, y_direction_housing_A, x_direction_housing_B,
y_direction_housing_B``. The Point->housing mapping in ``ASSUMED_CHANNEL_MAP``
is an ASSUMPTION based on that column order and is not stated in file metadata.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

ASSUMED_CHANNEL_MAP = {
    "Point1": "x_housing_A",
    "Point2": "y_housing_A",
    "Point3": "x_housing_B",
    "Point4": "y_housing_B",
}
STANDARD_GRAVITY = 9.80665


@dataclass
class VibrationRecord:
    path: Path
    values: np.ndarray  # (N, C) raw stored values
    fs_hz: float
    x_start_s: float
    increment_s: float
    n_samples: int
    channel_labels: list[str]
    channel_names: list[str]
    unit_label: str
    unit_factor: float
    unit_offset: float
    unit_info: str
    run_name: str
    absolute_time_raw: str
    absolute_time: dt.datetime | None
    creation_time_raw: str
    orig_location: str
    mat_header: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.n_samples * self.increment_s


def _s(x: Any) -> str:
    """Coerce MATLAB char/cell content into a python string."""
    while isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        if x.dtype.kind in "U" and x.ndim >= 1 and x.size > 1 and all(len(str(c)) <= 1 for c in x.ravel()):
            return "".join(str(c) for c in x.ravel())
        x = x.ravel()[0]
    return str(x)


def _first(x: Any) -> Any:
    """Return the first element of a (possibly nested) per-channel cell array."""
    if isinstance(x, np.ndarray) and x.dtype == object:
        return x.ravel()[0]
    return x


def _per_channel(x: Any) -> list[str]:
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [_s(v) for v in x.ravel()]
    if isinstance(x, (list, tuple)):
        return [_s(v) for v in x]
    return [_s(x)]


_ABS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: ms ([0-9.]+))?")


def parse_testlab_time(raw: str) -> dt.datetime | None:
    """Parse e.g. ``2021-06-28 11:32:36 ms 799.212``. Timezone is NOT stated."""
    m = _ABS_RE.search(raw or "")
    if not m:
        return None
    base = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    if m.group(2):
        base += dt.timedelta(milliseconds=float(m.group(2)))
    return base


def read_vibration_mat(path: str | Path, load_values: bool = True, dtype=np.float64) -> VibrationRecord:
    path = Path(path)
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "Signal" not in m:
        raise KeyError(f"{path.name}: no 'Signal' variable; keys={list(m)}")
    sig = m["Signal"]
    xv, yv, fr = sig.x_values, sig.y_values, sig.function_record
    values = np.asarray(yv.values)
    if values.ndim == 1:
        values = values[:, None]
    q = yv.quantity
    ut = q.unit_transformation
    ann = fr.TL_export_properties_annotation
    prim = fr.primary_channel
    labels = _per_channel(prim.label)
    abs_raw = _s(_first(ann.absolute_time))
    rec = VibrationRecord(
        path=path,
        values=values.astype(dtype, copy=False) if load_values else np.empty((0, values.shape[1])),
        fs_hz=1.0 / float(xv.increment),
        x_start_s=float(xv.start_value),
        increment_s=float(xv.increment),
        n_samples=int(xv.number_of_values),
        channel_labels=labels,
        channel_names=[ASSUMED_CHANNEL_MAP.get(l, l) for l in labels],
        unit_label=_s(q.label),
        unit_factor=float(ut.factor),
        unit_offset=float(ut.offset),
        unit_info=_s(q.info),
        run_name=_s(_first(ann.run_name)),
        absolute_time_raw=abs_raw,
        absolute_time=parse_testlab_time(abs_raw),
        creation_time_raw=_s(_first(fr.creation_time)),
        orig_location=_s(_first(ann.orig_location)),
        mat_header=str(m.get("__header__", b""), "latin-1") if isinstance(m.get("__header__"), bytes) else str(m.get("__header__")),
        extra={
            "x_quantity_label": _s(xv.quantity.label),
            "function_names": _per_channel(fr.name),
            "per_channel_absolute_time": _per_channel(ann.absolute_time),
            "per_channel_run_name": _per_channel(ann.run_name),
            "channel_group": _per_channel(ann.channel_group),
            "values_shape": tuple(values.shape),
            "values_dtype": str(values.dtype),
        },
    )
    del m
    return rec
