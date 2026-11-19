"""Paderborn University Bearing DataCenter (KAt) — metadata, labels and readers.

Source: Lessmeier, Kimotho, Zimmer & Sextro (2016), PHM Society European Conference 3(1),
doi:10.36001/phme.2016.v3i1.1577 (local copy: 02_LITERATURE/Lessmeier2016_PHME_Paderborn_benchmark.pdf).
Data licence stated in the paper: CC BY-NC 4.0 (non-commercial academic use, citation required).

Verified file schema (2026-11-18, K001/N15_M07_F10_K001_1.mat): MATLAB 5 struct named like the file with fields
Info, X (3 time axes: Mech_4kHz, HostService (64 kHz), Temp_1Hz), Y (7 signals: force, phase_current_1,
phase_current_2, speed, temp_2_bearing_module, torque, vibration_1) and Description
(General.DateTime 'dd.mm.yyyy HH:MM:SS'). Units are NOT stored in the files. The 64-kHz time axis has quantised
steps but (n-1)/T = 64000 Hz. Sensor (measuring log): PCB 336C04, 100 mV/g, Kistler 5015 charge amplifier,
30 kHz low-pass.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

FS_VIB = 64000.0
FS_MECH = 4000.0

# Operating settings (paper Table 6)
SETTINGS = {
    "N15_M07_F10": {"rpm": 1500, "torque_Nm": 0.7, "force_N": 1000, "setting_no": 0},
    "N09_M07_F10": {"rpm": 900, "torque_Nm": 0.7, "force_N": 1000, "setting_no": 1},
    "N15_M01_F10": {"rpm": 1500, "torque_Nm": 0.1, "force_N": 1000, "setting_no": 2},
    "N15_M07_F04": {"rpm": 1500, "torque_Nm": 0.7, "force_N": 400, "setting_no": 3},
}

# Bearing groups and damage descriptions (paper Tables 4, 5, 7)
# fields: group, label3, component, method/mode, extent level, combination, arrangement, characteristic
BEARINGS: dict[str, dict] = {}
for c, h, rpm in (("K001", ">50", "1500-2000"), ("K002", "19", "2900"), ("K003", "1", "3000"),
                  ("K004", "5", "3000"), ("K005", "10", "3000"), ("K006", "16", "2900")):
    BEARINGS[c] = dict(group="healthy", label3="H", component="-", method="run-in only", extent=0,
                       combination="-", arrangement="-", characteristic="-", run_in_h=h)
for c, comp, ext, meth in (("KA01", "OR", 1, "EDM"), ("KA03", "OR", 2, "electric engraver"),
                           ("KA05", "OR", 1, "electric engraver"), ("KA06", "OR", 2, "electric engraver"),
                           ("KA07", "OR", 1, "drilling"), ("KA08", "OR", 2, "drilling"), ("KA09", "OR", 2, "drilling"),
                           ("KI01", "IR", 1, "EDM"), ("KI03", "IR", 1, "electric engraver"),
                           ("KI05", "IR", 1, "electric engraver"), ("KI07", "IR", 2, "electric engraver"),
                           ("KI08", "IR", 2, "electric engraver")):
    BEARINGS[c] = dict(group=f"{comp}_artificial", label3=comp, component=comp, method=meth, extent=ext,
                       combination="S", arrangement="no repetition", characteristic="single point")
for c, mode, comp, comb, arr, ext, ch in (
        ("KA04", "fatigue: pitting", "OR", "S", "no repetition", 1, "single point"),
        ("KA15", "plastic deform.: indentations", "OR", "S", "no repetition", 1, "single point"),
        ("KA16", "fatigue: pitting", "OR", "R", "random", 2, "single point"),
        ("KA22", "fatigue: pitting", "OR", "S", "no repetition", 1, "single point"),
        ("KA30", "plastic deform.: indentations", "OR", "R", "random", 1, "distributed"),
        ("KB23", "fatigue: pitting", "IR(+OR)", "M", "random", 2, "single point"),
        ("KB24", "fatigue: pitting", "IR(+OR)", "M", "no repetition", 3, "distributed"),
        ("KB27", "plastic deform.: indentations", "OR+IR", "M", "random", 1, "distributed"),
        ("KI04", "fatigue: pitting", "IR", "M", "no repetition", 1, "single point"),
        ("KI14", "fatigue: pitting", "IR", "M", "no repetition", 1, "single point"),
        ("KI16", "fatigue: pitting", "IR", "S", "no repetition", 3, "single point"),
        ("KI17", "fatigue: pitting", "IR", "R", "random", 1, "single point"),
        ("KI18", "fatigue: pitting", "IR", "S", "no repetition", 2, "single point"),
        ("KI21", "fatigue: pitting", "IR", "S", "no repetition", 1, "single point")):
    if c.startswith("KB"):
        BEARINGS[c] = dict(group="multiple_real", label3=None, component=comp, method=f"accelerated lifetime; {mode}",
                           extent=ext, combination=comb, arrangement=arr, characteristic=ch)
    else:
        BEARINGS[c] = dict(group=f"{comp}_real", label3=comp, component=comp, method=f"accelerated lifetime; {mode}",
                           extent=ext, combination=comb, arrangement=arr, characteristic=ch)

# Paper Table 10 rows (real-damage 3-class sets used in the bearing-disjoint protocol PU-BD5)
BD5_ROWS = [("K001", "KA04", "KI04"), ("K002", "KA15", "KI14"), ("K003", "KA16", "KI16"),
            ("K004", "KA22", "KI18"), ("K005", "KA30", "KI21")]

# Default geometry if a fact sheet lacks values (paper Table 1, FAG example)
DEFAULT_GEOMETRY = {"n_balls": 8, "ball_diameter_mm": 6.75, "pitch_diameter_mm": 28.55, "contact_angle_deg": 0.0}

_FILE_RE = re.compile(r"^(N\d\d_M\d\d_F\d\d)_(K[AIB]?\d\d\d?|K[AIB]\d\d)_(\d+)$")


def parse_pu_filename(stem: str) -> tuple[str, str, int]:
    """'N15_M07_F10_KA04_3' -> ('N15_M07_F10', 'KA04', 3)."""
    m = re.match(r"^(N\d\d_M\d\d_F\d\d)_([A-Z]+\d+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"unrecognised Paderborn file stem {stem!r}")
    setting, code, k = m.group(1), m.group(2), int(m.group(3))
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}")
    if code not in BEARINGS:
        raise ValueError(f"unknown bearing code {code!r}")
    return setting, code, k


@dataclass
class PUMeasurement:
    vibration: np.ndarray  # float32, 64 kHz
    speed_rpm: np.ndarray  # 4 kHz
    torque: np.ndarray
    force: np.ndarray
    temp: np.ndarray
    datetime: dt.datetime | None
    raw_names: list[str]


def _signals(v) -> dict:
    return {str(c.Name): np.asarray(np.atleast_1d(c.Data)) for c in np.atleast_1d(v.Y)}


def read_pu_mat(path: str | Path) -> PUMeasurement:
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise KeyError(f"{path}: expected one struct, found {keys}")
    v = m[keys[0]]
    s = _signals(v)
    when = None
    try:
        when = dt.datetime.strptime(str(v.Description.General.DateTime), "%d.%m.%Y %H:%M:%S")
    except Exception:  # noqa: BLE001 - metadata optional
        pass
    return PUMeasurement(
        vibration=s["vibration_1"].astype(np.float32), speed_rpm=s.get("speed", np.array([np.nan])),
        torque=s.get("torque", np.array([np.nan])), force=s.get("force", np.array([np.nan])),
        temp=s.get("temp_2_bearing_module", np.array([np.nan])), datetime=when, raw_names=sorted(s))


_NUM = r"([0-9]+(?:[.,][0-9]+)?)"


def parse_fact_sheet_text(text: str) -> dict:
    """Extract geometry/manufacturer from the text of a '<code>.pdf' damage profile."""
    def grab(label: str):
        mm = re.search(label + r"\s*(?:mm|pc\.|°|-)?\s*" + _NUM, text)
        return float(mm.group(1).replace(",", ".")) if mm else None

    man = re.search(r"Manufacturer\s*-\s*([A-Za-z/]+)", text)
    return {
        "pitch_diameter_mm": grab(r"Pitch circle diameter"),
        "n_balls": grab(r"Number of rolling elements"),
        "ball_diameter_mm": grab(r"Rolling element diameter"),
        "contact_angle_deg": grab(r"Nominal pressure angle"),
        "manufacturer": man.group(1) if man else None,
    }


def parse_fact_sheet(pdf_path: str | Path) -> dict:
    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)
    out = parse_fact_sheet_text(text)
    out["text_chars"] = len(text)
    return out


def geometry_or_default(fs: dict) -> tuple[dict, bool]:
    g = {k: fs.get(k) for k in DEFAULT_GEOMETRY}
    used_default = False
    for k, v in DEFAULT_GEOMETRY.items():
        if g[k] is None:
            g[k] = v
            used_default = True
    return g, used_default
