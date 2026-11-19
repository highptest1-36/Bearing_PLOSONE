"""Physics concepts V3 (protocol V2.2 §4). V2 definitions in physics.py are untouched.

Differences from V2
  * BPFO shaft sidebands removed (no physical basis for a stationary outer race; reversed on PU).
  * Shaft guard g_k = df + 3 k sigma_fr (resolution- and speed-uncertainty-aware) instead of a fixed +-1 Hz.
  * Identifiability distance delta_F (bins) between the calibrated family region and the nearest guarded shaft line;
    family valid iff delta_F >= 1.
  * Harmonics overlapping a shaft guard or the other family's regions are excluded (overlap count reported).
  * Explicit masks; deterministic numerical representation (no NaN reaches a model).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import Regions, SEARCH_REL

FAMS = ("BPFO", "BPFI")
CONCEPTS_V3 = ["BPFO_snr1", "BPFO_harm", "BPFO_nharm", "BPFI_snr1", "BPFI_harm", "BPFI_nharm", "BPFI_sb", "contrast"]
RELIAB_V3 = ["ident_BPFO", "ident_BPFI", "valid_frac", "fam_max", "abs_contrast", "nharm_max", "n_excluded"]
RING_HZ = 20.0
HARM_DET_DB = 6.0
HARMONICS = (1, 2, 3)


def _guards(fr: float, df: float, sigma_fr: float, f_max: float) -> list[tuple[float, float]]:
    out = []
    for k in range(1, int(f_max / fr) + 2):
        g = df + 3.0 * k * sigma_fr
        out.append((k * fr - g, k * fr + g))
    return out


def _mask(f: np.ndarray, intervals) -> np.ndarray:
    m = np.zeros(len(f), bool)
    for lo, hi in intervals:
        m |= (f >= lo) & (f <= hi)
    return m


def _overlaps(a: tuple[float, float], intervals) -> bool:
    return any(min(a[1], hi) > max(a[0], lo) for lo, hi in intervals)


def family_regions(fr: float, df: float, sigma_fr: float, o: float, r: float, w: float) -> dict[int, tuple[float, float]]:
    out = {}
    for h in HARMONICS:
        c = h * o * fr * (1 + r)
        hw = max(h * w * o * fr, df) + 3.0 * h * o * sigma_fr
        out[h] = (c - hw, c + hw)
    return out


def identifiability(fr: float, df: float, sigma_fr: float, region: tuple[float, float]) -> float:
    """delta (bins) = (distance from region edge to the nearest guarded shaft line) / df."""
    c = 0.5 * (region[0] + region[1])
    hw = 0.5 * (region[1] - region[0])
    k = max(1, int(round(c / fr)))
    best = np.inf
    for kk in (k - 1, k, k + 1):
        if kk < 1:
            continue
        g = df + 3.0 * kk * sigma_fr
        best = min(best, (abs(c - kk * fr) - hw - g) / df)
    return float(best)


def snr_region(f: np.ndarray, E: np.ndarray, region: tuple[float, float], excl: np.ndarray) -> tuple[float, float]:
    band = (f >= region[0]) & (f <= region[1]) & ~excl
    if not band.any():
        return np.nan, np.nan
    i = int(np.flatnonzero(band)[np.argmax(E[band])])
    ring = (f >= region[0] - RING_HZ) & (f <= region[1] + RING_HZ) & ~excl & ~band
    ring[max(i - 2, 0): i + 3] = False
    if ring.sum() < 3:
        return np.nan, float(f[i])
    floor = float(np.median(E[ring]))
    if floor <= 0 or E[i] <= 0:
        return np.nan, float(f[i])
    return float(20 * np.log10(E[i] / floor)), float(f[i])


def scan_v3(f: np.ndarray, E: np.ndarray, fr: float, kin: dict[str, float], sigma_fr: float = 0.0) -> dict[str, float]:
    """Label-free line search (+-3.5 %) with the V3 shaft guard; returns relative offset r and SNR per family."""
    df = f[1] - f[0]
    g = _mask(f, _guards(fr, df, sigma_fr, f[-1]))
    out = {}
    for F in FAMS:
        fk = kin[F] * fr
        snr, fp = snr_region(f, E, (fk * (1 - SEARCH_REL), fk * (1 + SEARCH_REL)), g)
        out[f"{F}_scan_snr"] = snr
        out[f"{F}_scan_r"] = fp / fk - 1.0 if np.isfinite(fp) else np.nan
    return out


def _line_region_has_free_bins(f, region, gmask) -> bool:
    band = (f >= region[0]) & (f <= region[1])
    return bool((band & ~gmask).any())


def line_identifiable(f: np.ndarray, E: np.ndarray, f0: float, fr: float, df: float, sigma_fr: float) -> tuple[bool, float]:
    """Protocol V2.4: the measured peak f0 is identifiable iff it lies >= 1 bin outside the guarded nearest shaft
    harmonic (delta >= 1) and is a local maximum of E. Returns (ok, delta_bins)."""
    if not np.isfinite(f0):
        return False, -5.0
    k = max(1, int(round(f0 / fr)))
    delta = min((abs(f0 - kk * fr) - (df + 3.0 * kk * sigma_fr)) / df for kk in (k - 1, k, k + 1) if kk >= 1)
    i = int(round((f0 - f[0]) / df))
    local = (i <= 0 or E[i] >= E[i - 1]) and (i >= len(E) - 1 or E[i] >= E[i + 1])
    return bool(delta >= 1.0 and local), float(delta)


@dataclass
class V3Result:
    c: np.ndarray        # 8 concepts (raw values; masked entries = 0.0)
    m: np.ndarray        # 8 masks (0/1)
    q: np.ndarray        # 7 reliability features
    info: dict


def features_v3(f: np.ndarray, E: np.ndarray, fr: float, kin: dict[str, float], reg: Regions,
                sigma_fr: float = 0.0) -> V3Result:
    df = float(f[1] - f[0])
    guards = _guards(fr, df, sigma_fr, f[-1])
    gmask = _mask(f, guards)
    regs = {F: family_regions(fr, df, sigma_fr, kin[F], reg.centre[F], reg.half[F]) for F in FAMS}
    # BPFI shaft sidebands of the fundamental region
    lo, hi = regs["BPFI"][1]
    sb = [(lo - fr, hi - fr), (lo + fr, hi + fr)]
    other = {"BPFO": list(regs["BPFI"].values()) + sb, "BPFI": list(regs["BPFO"].values())}
    vals, masks, info = {}, {}, {}
    harm_val = {}
    n_excl = 0
    nharm = {}
    for F in FAMS:
        info[f"delta_region_{F}"] = identifiability(fr, df, sigma_fr, regs[F][1])  # V2.3 (reported only)
        excl = gmask | _mask(f, other[F])
        snrs = {}
        valid = True
        delta_line = -5.0
        for h in HARMONICS:
            r_h = regs[F][h]
            if _overlaps(r_h, other[F]) or (h > 1 and _overlaps(r_h, guards)) and not _line_region_has_free_bins(f, r_h, gmask):
                n_excl += 1
                if h == 1:
                    valid = False
                continue
            s, fp = snr_region(f, E, r_h, excl)
            ok, d = line_identifiable(f, E, fp, fr, df, sigma_fr)
            if h == 1:
                info[f"f0_{F}"] = fp
                delta_line = d
                valid = valid and ok and np.isfinite(s)
            if ok and np.isfinite(s):
                snrs[h] = s
            elif h > 1:
                n_excl += 1
        info[f"delta_{F}"] = delta_line
        valid = valid and (1 in snrs)
        info[f"valid_{F}"] = bool(valid)
        if valid:
            vals[f"{F}_snr1"] = snrs[1]
            vals[f"{F}_harm"] = float(np.mean(list(snrs.values())))
            vals[f"{F}_nharm"] = float(sum(v >= HARM_DET_DB for v in snrs.values()))
            harm_val[F] = vals[f"{F}_harm"]
        nharm[F] = vals.get(f"{F}_nharm", 0.0)
        for k in ("snr1", "harm", "nharm"):
            masks[f"{F}_{k}"] = valid
    # BPFI sidebands
    if info["valid_BPFI"] and not any(_overlaps(s, guards) for s in sb):
        excl = gmask | _mask(f, list(regs["BPFO"].values()))
        s = [snr_region(f, E, x, excl)[0] for x in sb]
        s = [v for v in s if np.isfinite(v)]
        if s:
            vals["BPFI_sb"] = float(np.mean(s))
    masks["BPFI_sb"] = "BPFI_sb" in vals
    if info["valid_BPFO"] and info["valid_BPFI"]:
        vals["contrast"] = vals["BPFO_harm"] - vals["BPFI_harm"]
    masks["contrast"] = "contrast" in vals
    c = np.array([vals.get(k, 0.0) if masks[k] else 0.0 for k in CONCEPTS_V3], float)
    m = np.array([float(masks[k]) for k in CONCEPTS_V3])
    fam_max = max(harm_val.values()) if harm_val else 0.0
    q = np.array([np.clip(info["delta_BPFO"], -5, 20), np.clip(info["delta_BPFI"], -5, 20), m.mean(), fam_max,
                  abs(vals["contrast"]) if masks["contrast"] else 0.0, max(nharm.values()), float(n_excl)])
    return V3Result(c, m, q, info)


def standardise(C: np.ndarray, M: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z = m * (c - mu) / sd with mu, sd from VALID training entries; returns z, mu, sd (masked entries -> 0)."""
    Ct, Mt = C[train_idx], M[train_idx].astype(bool)
    mu = np.array([Ct[Mt[:, j], j].mean() if Mt[:, j].any() else 0.0 for j in range(C.shape[1])])
    sd = np.array([Ct[Mt[:, j], j].std() if Mt[:, j].sum() > 1 else 1.0 for j in range(C.shape[1])])
    sd = np.where(sd > 1e-8, sd, 1.0)
    Z = M * (C - mu) / sd
    return Z, mu, sd
