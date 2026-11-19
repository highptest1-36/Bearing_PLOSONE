"""Rolling-element bearing characteristic frequencies.

Standard kinematic formulas (e.g. Randall & Antoni, MSSP 2011):

    FTF  = fr/2 * (1 - (d/D) cos(theta))
    BPFO = N fr/2 * (1 - (d/D) cos(theta))
    BPFI = N fr/2 * (1 + (d/D) cos(theta))
    BSF  = D/(2 d) * fr * (1 - ((d/D) cos(theta))^2)

``fr`` is the shaft rotation frequency in Hz. These are kinematic (no-slip)
reference values; real peaks typically deviate by ~1-2 % because of slip and
load-dependent contact angle.

NSK 6205 DDU geometry as reported by Jung et al. (2023, Data in Brief,
doi:10.1016/j.dib.2023.109049, Table 1): N=9, d=7.90 mm, D=38.5 mm, theta=0,
fr=50.17 Hz (3010 rpm) -> FTF 19.94, BPFI 272.07, BPFO 179.43 Hz.
The paper lists BSF = 234.19 Hz, which equals D/d * fr * (1-(d/D)^2), i.e.
TWICE the standard BSF (117.1 Hz). The paper value corresponds to the
"ball-defect impact" frequency 2*BSF; ball faults are not part of this partition,
so this discrepancy does not affect the IR/OR analysis, but it is documented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    name: str
    n_balls: int
    ball_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0

    @property
    def ratio(self) -> float:
        return self.ball_diameter_mm / self.pitch_diameter_mm * math.cos(math.radians(self.contact_angle_deg))


NSK_6205_DDU = BearingGeometry("NSK 6205 DDU", 9, 7.90, 38.5, 0.0)

# Values printed in the source paper (used only as unit-test references).
PAPER_SHAFT_HZ = 50.17
PAPER_RPM = 3010
PAPER_VALUES_HZ = {"FTF": 19.94, "BPFI": 272.07, "BPFO": 179.43, "BSF_paper": 234.19}


def ftf(fr: float, g: BearingGeometry = NSK_6205_DDU) -> float:
    return fr / 2.0 * (1.0 - g.ratio)


def bpfo(fr: float, g: BearingGeometry = NSK_6205_DDU) -> float:
    return g.n_balls * fr / 2.0 * (1.0 - g.ratio)


def bpfi(fr: float, g: BearingGeometry = NSK_6205_DDU) -> float:
    return g.n_balls * fr / 2.0 * (1.0 + g.ratio)


def bsf(fr: float, g: BearingGeometry = NSK_6205_DDU) -> float:
    """Standard ball spin frequency (one rotation of the ball about its own axis)."""
    return g.pitch_diameter_mm / (2.0 * g.ball_diameter_mm) * fr * (1.0 - g.ratio ** 2)


def all_frequencies(fr: float, g: BearingGeometry = NSK_6205_DDU) -> dict[str, float]:
    return {
        "fr": fr,
        "FTF": ftf(fr, g),
        "BPFO": bpfo(fr, g),
        "BPFI": bpfi(fr, g),
        "BSF": bsf(fr, g),
        "2xBSF": 2.0 * bsf(fr, g),
    }


def orders(g: BearingGeometry = NSK_6205_DDU) -> dict[str, float]:
    """Characteristic frequencies as multiples of shaft frequency (speed-independent)."""
    return {k: v for k, v in all_frequencies(1.0, g).items() if k != "fr"}


def shaft_harmonic_collisions(f: float, fr: float, tol_hz: float, max_order: int = 40) -> list[int]:
    """Shaft harmonics k*fr that fall within ``tol_hz`` of frequency ``f``."""
    return [k for k in range(1, max_order + 1) if abs(k * fr - f) <= tol_hz]
