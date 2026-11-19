"""Reader for KAIST varying-load temperature/current ``.tdms`` files.

Verified schema (inspected 2026-11-18; NI FlexLogger 2021 R1.1 logs):

    group 'Log':
      cDAQ9185-1F486B5Mod1/ai0, ai1 -> Temperature, NI 9210 (miniTC), K-type, unit degC
      cDAQ9185-1F486B5Mod2/ai0, ai2, ai3 -> Current, NI 9775, unit A
    waveform properties: wf_start_time (absolute, UTC as decoded by npTDMS),
    wf_increment = 3.905e-05 s (=> 25 608.2 Hz nominal, NOT exactly 25 600 Hz).

Channel mapping to the paper's column names ('Temperature_housing_A',
'Temperature_housing_B', 'U-phase', 'V-phase', 'W-phase') is an ASSUMPTION
based on channel order; it is not present in file metadata.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from nptdms import TdmsFile

ASSUMED_TEMP_MAP = {
    "Mod1/ai0": "T_housing_A",
    "Mod1/ai1": "T_housing_B",
}
ASSUMED_CURRENT_MAP = {
    "Mod2/ai0": "I_U",
    "Mod2/ai2": "I_V",
    "Mod2/ai3": "I_W",
}


@dataclass
class TdmsChannelInfo:
    raw_name: str
    name: str
    kind: str  # 'temperature' | 'current' | 'other'
    n_samples: int
    unit: str
    increment_s: float | None
    start_time_utc: dt.datetime | None
    device: str
    properties: dict[str, Any]


@dataclass
class TdmsRecord:
    path: Path
    file_name_prop: str
    run_number: int | None
    date_created_utc: dt.datetime | None
    product: str
    channels: list[TdmsChannelInfo]
    data: dict[str, np.ndarray] = field(default_factory=dict)  # keyed by canonical name

    def channel(self, name: str) -> TdmsChannelInfo:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)


def _to_dt(x: Any) -> dt.datetime | None:
    if x is None:
        return None
    if isinstance(x, np.datetime64):
        return dt.datetime.utcfromtimestamp(0) + dt.timedelta(microseconds=int(x.astype("datetime64[us]").astype(np.int64)))
    if isinstance(x, dt.datetime):
        return x
    return None


def _canonical(raw: str) -> tuple[str, str]:
    key = raw.split("-", 1)[-1] if "-" in raw else raw
    # raw like 'cDAQ9185-1F486B5Mod1/ai0' -> take 'Mod1/ai0'
    idx = raw.find("Mod")
    key = raw[idx:] if idx >= 0 else raw
    if key in ASSUMED_TEMP_MAP:
        return ASSUMED_TEMP_MAP[key], "temperature"
    if key in ASSUMED_CURRENT_MAP:
        return ASSUMED_CURRENT_MAP[key], "current"
    return raw, "other"


def read_tdms(path: str | Path, load: tuple[str, ...] = ("temperature",)) -> TdmsRecord:
    """Read metadata for every channel and data for channel kinds in ``load``."""
    path = Path(path)
    want_data = bool(load)
    tf = TdmsFile.read(str(path)) if want_data else TdmsFile.read_metadata(str(path))
    props = dict(tf.properties)
    chans: list[TdmsChannelInfo] = []
    data: dict[str, np.ndarray] = {}
    for g in tf.groups():
        for c in g.channels():
            p = dict(c.properties)
            name, kind = _canonical(c.name)
            if kind == "other" and g.name != "Log":
                continue
            n = len(c)
            chans.append(TdmsChannelInfo(
                raw_name=f"{g.name}/{c.name}", name=name, kind=kind, n_samples=n,
                unit=str(p.get("unit_string", "")),
                increment_s=float(p["wf_increment"]) if "wf_increment" in p else None,
                start_time_utc=_to_dt(p.get("wf_start_time")),
                device=str(p.get("DAC~Device~Name", "")),
                properties={k: (str(v) if isinstance(v, np.datetime64) else v) for k, v in p.items()},
            ))
            if want_data and kind in load and n > 0:
                data[name] = np.asarray(c[:], dtype=np.float64)
    rn = props.get("RunNumber")
    rec = TdmsRecord(
        path=path,
        file_name_prop=str(props.get("name", "")),
        run_number=int(rn) if rn is not None else None,
        date_created_utc=_to_dt(props.get("Date Created")),
        product=f"{props.get('ProductName', '')} {props.get('ProductVersion', '')}".strip(),
        channels=chans,
        data=data,
    )
    del tf
    return rec
