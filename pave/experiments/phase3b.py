"""Phase 3B data/feature plumbing (protocol V2.2 + V2.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pave.features import physics as PH
from pave.features import physics_v3 as V3

L3 = ["H", "IR", "OR"]
LIDX = {k: i for i, k in enumerate(L3)}


class V3Cache:
    """V3 concepts for a table of spectra rows, calibrated on a given set of TRAINING bearings (cached)."""

    def __init__(self, f: np.ndarray, E: np.ndarray, bearings: np.ndarray, labels: np.ndarray, fr: np.ndarray,
                 sigma: np.ndarray, kin: np.ndarray, rows: np.ndarray | None = None):
        self.f, self.E, self.b, self.lab = f, E, np.asarray(bearings), np.asarray(labels)
        self.fr, self.sigma, self.kin = np.asarray(fr, float), np.asarray(sigma, float), np.asarray(kin, float)
        self.rows = np.arange(len(self.b)) if rows is None else np.asarray(rows)
        sc = [V3.scan_v3(f, E[i].astype(float), self.fr[i], {"BPFO": self.kin[i, 0], "BPFI": self.kin[i, 1]}, self.sigma[i])
              for i in range(len(self.b))]
        self.scan = pd.DataFrame(sc)
        self._c: dict = {}

    def features(self, train_bearings) -> tuple[np.ndarray, np.ndarray, np.ndarray, PH.Regions]:
        key = frozenset(train_bearings)
        if key not in self._c:
            tr = np.isin(self.b, list(key))
            lab = np.where(np.isin(self.lab[tr], L3), self.lab[tr], "?")
            reg = PH.calibrate_regions({F: self.scan[f"{F}_scan_r"].to_numpy()[tr] for F in V3.FAMS},
                                       {F: self.scan[f"{F}_scan_snr"].to_numpy()[tr] for F in V3.FAMS}, lab)
            C = np.zeros((len(self.b), 8))
            M = np.zeros((len(self.b), 8))
            Q = np.zeros((len(self.b), 7))
            for i in self.rows:
                r = V3.features_v3(self.f, self.E[i].astype(float), self.fr[i],
                                   {"BPFO": self.kin[i, 0], "BPFI": self.kin[i, 1]}, reg, self.sigma[i])
                C[i], M[i], Q[i] = r.c, r.m, r.q
            self._c[key] = (C, M, Q, reg)
        return self._c[key]


def standardise_q(Q: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    mu = Q[train_rows].mean(0)
    sd = Q[train_rows].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (Q - mu) / sd


def b0_fit(fam_max: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(fam_max, float)
    faulty = y != 0
    best = (-1.0, 0.0)
    for t in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197))):
        ba = 0.5 * (np.mean(s[faulty] >= t) + np.mean(s[~faulty] < t))
        if ba > best[0]:
            best = (ba, float(t))
    return best[1]


def b0_predict(C: np.ndarray, M: np.ndarray, tau: float) -> np.ndarray:
    io, ii, ic = V3.CONCEPTS_V3.index("BPFO_harm"), V3.CONCEPTS_V3.index("BPFI_harm"), V3.CONCEPTS_V3.index("contrast")
    ho = np.where(M[:, io] > 0, C[:, io], -np.inf)
    hi = np.where(M[:, ii] > 0, C[:, ii], -np.inf)
    fam = np.maximum(ho, hi)
    pred = np.where(fam >= tau, np.where(M[:, ic] > 0, np.where(C[:, ic] > 0, 2, 1), np.where(ho >= hi, 2, 1)), 0)
    P = np.zeros((len(C), 3))
    P[np.arange(len(C)), pred] = 1.0
    return P


def compatibility(C: np.ndarray, M: np.ndarray, pred: np.ndarray, tau: float) -> np.ndarray:
    """kappa (dB): agreement between the predicted class and the measured evidence (protocol V2.2 §8)."""
    io, ii, ic = V3.CONCEPTS_V3.index("BPFO_harm"), V3.CONCEPTS_V3.index("BPFI_harm"), V3.CONCEPTS_V3.index("contrast")
    ho = np.where(M[:, io] > 0, C[:, io], 0.0)
    hi = np.where(M[:, ii] > 0, C[:, ii], 0.0)
    fam = np.maximum(ho, hi)
    con = np.where(M[:, ic] > 0, C[:, ic], np.where(pred == 2, ho - tau, tau - hi))
    return np.where(pred == 0, tau - fam, np.where(pred == 2, con, -con))
