"""Parse KAIST varying-load condition labels from file names and internal metadata.

The KAIST varying-load dataset (Jung et al., 2023, Data in Brief,
doi:10.1016/j.dib.2023.109049; Mendeley ztmf3m7h5x v6) names files as
``<load>Nm_<condition>[_<severity>].<ext>``, e.g. ``0Nm_BPFI_03.mat``,
``2Nm_Unbalalnce_0583mg.mat`` (sic; the typo exists in the distributed archive).

Internal metadata uses a different vocabulary (e.g. vibration ``run_name``
``210625_0Nm_Inner_0.3mm``; TDMS file property ``LogFile_0Nm_Inner_1mm``).
Both vocabularies are parsed into the same canonical ``Condition`` so that
filename labels can be cross-checked against embedded labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional

# Canonical fault groups
HEALTHY = "Healthy"
IR = "IR"
OR = "OR"
MISALIGN = "Misalign"
UNBALANCE = "Unbalance"

BEARING_GROUPS = (HEALTHY, IR, OR)

# 7-way primary labels and 3-way secondary labels (blueprint section 2)
LABELS_7 = ("H", "IR_0p3", "IR_1p0", "IR_3p0", "OR_0p3", "OR_1p0", "OR_3p0")
LABELS_3 = ("H", "IR", "OR")


@dataclass(frozen=True)
class Condition:
    load_Nm: int
    fault_group: str
    severity_value: Optional[float]  # mm for IR/OR/Misalign, mg for Unbalance
    severity_unit: Optional[str]

    @property
    def in_bearing_scope(self) -> bool:
        return self.fault_group in BEARING_GROUPS

    @property
    def label7(self) -> Optional[str]:
        if self.fault_group == HEALTHY:
            return "H"
        if self.fault_group in (IR, OR) and self.severity_value is not None:
            return f"{self.fault_group}_{_mm_tag(self.severity_value)}"
        return None

    @property
    def label3(self) -> Optional[str]:
        if self.fault_group == HEALTHY:
            return "H"
        if self.fault_group in (IR, OR):
            return self.fault_group
        return None

    @property
    def condition_id(self) -> str:
        """Load-independent condition identifier, e.g. ``IR_0p3`` or ``Unbalance_583mg``."""
        if self.fault_group == HEALTHY:
            return "H"
        if self.severity_value is None:
            return self.fault_group
        if self.severity_unit == "mm":
            return f"{self.fault_group}_{_mm_tag(self.severity_value)}"
        return f"{self.fault_group}_{int(round(self.severity_value))}{self.severity_unit}"

    @property
    def recording_id(self) -> str:
        """Load-specific recording identifier, e.g. ``L0_IR_0p3``."""
        return f"L{self.load_Nm}_{self.condition_id}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(label7=self.label7, label3=self.label3,
                 condition_id=self.condition_id, recording_id=self.recording_id,
                 in_bearing_scope=self.in_bearing_scope)
        return d


def same_condition(a: Optional[Condition], b: Optional[Condition], rel_tol: float = 0.01) -> bool:
    """Same load, group and severity (severity within ``rel_tol``; e.g. 583 vs 584 mg)."""
    if a is None or b is None:
        return False
    if (a.load_Nm, a.fault_group, a.severity_unit) != (b.load_Nm, b.fault_group, b.severity_unit):
        return False
    if a.severity_value is None or b.severity_value is None:
        return a.severity_value is None and b.severity_value is None
    return abs(a.severity_value - b.severity_value) <= rel_tol * max(abs(a.severity_value), 1e-12)


def _mm_tag(v: float) -> str:
    return f"{v:.1f}".replace(".", "p")


_FILE_RE = re.compile(
    r"^(?P<load>\d+)Nm_(?P<cond>[A-Za-z]+)(?:_(?P<sev>[0-9.]+)(?P<unit>mg|mm)?)?$"
)

_COND_ALIASES = {
    "normal": HEALTHY,
    "bpfi": IR,
    "inner": IR,
    "bpfo": OR,
    "outer": OR,
    "misalign": MISALIGN,
    "misalignment": MISALIGN,
    "unbalance": UNBALANCE,
    "unbal": UNBALANCE,  # vibration run_name abbreviation
    "unbalalnce": UNBALANCE,  # typo present in distributed 2 Nm vibration filenames
}


def parse_filename(stem: str) -> Condition:
    """Parse a KAIST varying-load file stem (no extension).

    Filename severity codes: ``03``/``10``/``30`` are tenths of mm for IR/OR;
    ``01``/``03``/``05`` are tenths of mm for Misalign; ``0583mg`` etc. are mg.
    """
    m = _FILE_RE.match(stem)
    if not m:
        raise ValueError(f"Unrecognised KAIST file stem: {stem!r}")
    load = int(m.group("load"))
    cond_raw = m.group("cond").lower()
    if cond_raw not in _COND_ALIASES:
        raise ValueError(f"Unknown condition token {cond_raw!r} in {stem!r}")
    group = _COND_ALIASES[cond_raw]
    sev, unit = m.group("sev"), m.group("unit")
    if group == HEALTHY:
        if sev is not None:
            raise ValueError(f"Healthy file with severity: {stem!r}")
        return Condition(load, group, None, None)
    if sev is None:
        raise ValueError(f"Missing severity in {stem!r}")
    if group == UNBALANCE:
        if unit != "mg":
            raise ValueError(f"Unbalance severity must be in mg: {stem!r}")
        return Condition(load, group, float(sev), "mg")
    # IR / OR / Misalign: two-digit code in tenths of a millimetre
    if unit not in (None, "mm"):
        raise ValueError(f"Unexpected unit in {stem!r}")
    value = float(sev) / 10.0 if unit is None else float(sev)
    return Condition(load, group, value, "mm")


_INTERNAL_RE = re.compile(
    r"(?P<load>\d+)\s*Nm_(?P<cond>[A-Za-z]+)(?:_(?P<sev>[0-9.]+)\s*(?P<unit>mm|mg)?)?",
    re.IGNORECASE,
)


def parse_internal_label(text: str) -> Optional[Condition]:
    """Parse labels embedded in file metadata.

    Examples: ``210625_0Nm_Inner_0.3mm`` (Testlab run_name),
    ``LogFile_0Nm_Inner_1mm`` / ``LogFile_0Nm_Outer_03mm`` (FlexLogger name).
    Internal numbers are interpreted as millimetres when the string contains a
    decimal point or an explicit ``mm`` unit with a value <= 5; a leading-zero
    two-digit code without a decimal (``03mm``) is interpreted as tenths of mm,
    matching the filename convention. Ambiguous inputs return the literal
    reading; the caller records the raw string so humans can re-check.
    """
    if not text:
        return None
    m = _INTERNAL_RE.search(text)
    if not m:
        return None
    cond_raw = m.group("cond").lower()
    if cond_raw not in _COND_ALIASES:
        return None
    group = _COND_ALIASES[cond_raw]
    load = int(m.group("load"))
    sev, unit = m.group("sev"), m.group("unit")
    if group == HEALTHY or sev is None:
        return Condition(load, group, None, None)
    if group == UNBALANCE:
        # FlexLogger names write the mass as "0.583", "0.1169", "0.3318" (no unit) for
        # 583/1169/3318 mg; strip the leading "0." when no unit is given.
        if unit is None and sev.startswith("0.") and len(sev) > 2:
            return Condition(load, group, float(sev[2:]), "mg")
        return Condition(load, group, float(sev), "mg")
    if "." in sev:
        value = float(sev)
    elif len(sev) >= 2 and sev.startswith("0"):
        value = float(sev) / 10.0  # e.g. "03mm" -> 0.3 mm, "05" -> 0.5 mm
    else:
        value = float(sev)  # e.g. "1mm" -> 1.0 mm, "3mm" -> 3.0 mm
    return Condition(load, group, value, "mm")
