"""University of Ottawa UORED-VAFCLS reader + frozen label mapping (external validation; protocol EV-V1).

Source: Sehri, Dumond & Bouchard, Data in Brief 49 (2023) 109327, doi 10.1016/j.dib.2023.109327 (CC BY 4.0).
Files: 3_MatLab_Raw_Data_Files (.mat)/<group>/<L>_<n>_<s>.mat, one variable named like the file stem, array
(420000, 5): [accelerometer (m/s^2), microphone (V), speed (RPM), load (N), temperature differential (degC)];
fs = 42 000 Hz, 10 s. <L> in {H, I, O, B, C}; <n> bearing 1..20; <s> 0 healthy / 1 developing / 2 faulty.
Bearings 1-5 inner race, 6-10 outer race, 11-15 ball, 16-20 cage (paper Fig. 1).
Geometry (paper Table 2): NSK 6203ZZ and FAFNIR 203KD, pitch 28.50 mm, ball 6.77 mm, 8 balls (contact angle not stated
-> 0 assumed).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.io as sio

FS = 42000.0
COLS = ["accel_ms2", "acoustic_V", "speed_rpm", "load_N", "temp_diff_C"]
GEOMETRY = {"n_balls": 8, "ball_diameter_mm": 6.77, "pitch_diameter_mm": 28.50, "contact_angle_deg": 0.0}
NSK_BEARINGS = {14, 16, 17, 18, 20}  # paper Fig. 1 (others FAFNIR 203KD)
_RE = re.compile(r"^([HIOBC])_(\d+)_(\d)$")


def parse_stem(stem: str) -> dict:
    m = _RE.match(stem)
    if not m:
        raise ValueError(f"unrecognised Ottawa file stem {stem!r}")
    letter, n, s = m.group(1), int(m.group(2)), int(m.group(3))
    if not 1 <= n <= 20 or s not in (0, 1, 2):
        raise ValueError(stem)
    fault_of_bearing = "IR" if n <= 5 else "OR" if n <= 10 else "BALL" if n <= 15 else "CAGE"
    if letter == "H" and s != 0 or letter != "H" and s == 0:
        raise ValueError(f"inconsistent state in {stem!r}")
    expected_letter = {"IR": "I", "OR": "O", "BALL": "B", "CAGE": "C"}[fault_of_bearing]
    if letter not in ("H", expected_letter):
        raise ValueError(f"letter {letter} inconsistent with bearing {n} in {stem!r}")
    return {"stem": stem, "letter": letter, "bearing": n, "state": s, "bearing_fault_type": fault_of_bearing,
            "manufacturer": "NSK 6203ZZ" if n in NSK_BEARINGS else "FAFNIR 203KD"}


def label_mapping(info: dict) -> dict:
    """Frozen mapping (EV-V1 §4). primary in {H, IR, OR} or None; role in {primary, developing, ood_faulty, ood_developing}."""
    s, t = info["state"], info["bearing_fault_type"]
    if s == 0:
        return {"primary": "H", "role": "primary"}
    if t in ("IR", "OR"):
        return {"primary": t if s == 2 else None, "role": "primary" if s == 2 else "developing", "developing_label": t if s == 1 else None}
    return {"primary": None, "role": "ood_faulty" if s == 2 else "ood_developing"}


def find_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*.mat") if "MatLab" in str(p.parent.parent) or "MatLab" in str(p.parent))


def read_ottawa(path: Path) -> np.ndarray:
    m = sio.loadmat(str(path))
    keys = [k for k in m if not k.startswith("__")]
    if len(keys) != 1:
        raise KeyError(f"{path.name}: {keys}")
    a = np.asarray(m[keys[0]], dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 5:
        raise ValueError(f"{path.name}: shape {a.shape}")
    return a
