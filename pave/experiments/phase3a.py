"""Phase 3A: nested bearing-group validation on PU-BD5 (protocol V2.1).

Outer loop  : 5 folds; test = one Table-10 row (3 bearings: H, OR, IR); outer-train = the other 4 rows (12 bearings).
Inner loop  : 4 folds inside outer-train; inner-val = one row, inner-train = the other 3 rows.
Everything fitted (physics-region calibration, scalers, thresholds, hyper-parameters, epoch counts, temperature
scaling) uses only the data available at that level; outer-test bearings are never touched before prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pave.data.paderborn import BD5_ROWS, BEARINGS
from pave.features import physics as PH

L3 = ["H", "IR", "OR"]
LIDX = {k: i for i, k in enumerate(L3)}
ROWS = [frozenset(r) for r in BD5_ROWS]


# ----------------------------------------------------------------------------- folds
def outer_folds() -> list[dict]:
    out = []
    for k in range(5):
        tr = frozenset().union(*[ROWS[i] for i in range(5) if i != k])
        out.append({"fold": f"O{k+1}", "k": k, "test": ROWS[k], "train": tr,
                    "train_rows": [i for i in range(5) if i != k]})
    return out


def inner_folds(outer: dict) -> list[dict]:
    out = []
    for j in outer["train_rows"]:
        tr = frozenset().union(*[ROWS[i] for i in outer["train_rows"] if i != j])
        out.append({"fold": f"{outer['fold']}-I{j+1}", "val": ROWS[j], "train": tr})
    return out


def check_nested(outer: dict) -> None:
    assert not (outer["train"] & outer["test"]), "outer leak"
    seen = set()
    for i in inner_folds(outer):
        assert not (i["train"] & i["val"]), "inner leak"
        assert not ((i["train"] | i["val"]) & outer["test"]), "outer-test bearing used inside inner loop"
        assert (i["train"] | i["val"]) == outer["train"]
        seen |= i["val"]
    assert seen == outer["train"], "every outer-train bearing must be inner-validated once"


# ----------------------------------------------------------------------------- physics features with calibration
class PhysicsCache:
    """Physics features for a set of windows, calibrated on a given set of TRAINING bearings (cached)."""

    def __init__(self, ds, kin: np.ndarray, scan: pd.DataFrame, window_idx: np.ndarray):
        self.ds, self.kin, self.scan, self.idx = ds, kin, scan, np.asarray(window_idx)
        self.y = np.array([LIDX.get(v, -1) if isinstance(v, str) else -1 for v in ds.meta.label3])
        self._c: dict[frozenset, tuple] = {}

    def features(self, train_bearings: frozenset) -> tuple[np.ndarray, PH.Regions]:
        key = frozenset(train_bearings)
        if key not in self._c:
            m = self.ds.meta.bearing.isin(key).to_numpy()
            lab = np.array(L3 + ["?"])[np.where(self.y[m] >= 0, self.y[m], 3)]
            reg = PH.calibrate_regions({f: self.scan[f"{f}_scan_r"].to_numpy()[m] for f in PH.FAMILIES},
                                       {f: self.scan[f"{f}_scan_snr"].to_numpy()[m] for f in PH.FAMILIES}, lab)
            F = np.full((len(self.ds.meta), len(PH.FEATURE_NAMES)), np.nan)
            f = np.arange(self.ds.env.shape[2], dtype=float)
            for i in self.idx:
                fr = self.ds.meta.fr_hz.iat[i]
                ko = {"BPFO": self.kin[i, 0], "BPFI": self.kin[i, 1]}
                d = PH.window_features(f, self.ds.env[i, 0].astype(float), fr, ko, reg)
                F[i] = [d[k] for k in PH.FEATURE_NAMES]
            self._c[key] = (F, reg)
        return self._c[key]


def impute(Xtr: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    out = []
    for X in (Xtr,) + others:
        X = X.copy()
        r, c = np.where(~np.isfinite(X))
        X[r, c] = med[c]
        out.append(X)
    return out


# ----------------------------------------------------------------------------- B0 physics rule
FAM_MAX = [PH.FEATURE_NAMES.index("BPFO_harm"), PH.FEATURE_NAMES.index("BPFI_harm")]
CONTRAST = PH.FEATURE_NAMES.index("contrast")


def b0_fit(F: np.ndarray, y: np.ndarray) -> float:
    """Single threshold tau on max family score maximising H-vs-faulty balanced accuracy on TRAINING windows."""
    s = np.nanmax(F[:, FAM_MAX], axis=1)
    s = np.where(np.isfinite(s), s, -np.inf)
    faulty = y != 0
    cands = np.unique(np.quantile(s[np.isfinite(s)], np.linspace(0.01, 0.99, 197)))
    best = (-1, cands[0])
    for t in cands:
        ba = 0.5 * (np.mean(s[faulty] >= t) + np.mean(s[~faulty] < t))
        if ba > best[0]:
            best = (ba, t)
    return float(best[1])


def b0_predict(F: np.ndarray, tau: float) -> np.ndarray:
    s = np.nanmax(F[:, FAM_MAX], axis=1)
    c = np.nan_to_num(F[:, CONTRAST])
    pred = np.where(~(s >= tau), 0, np.where(c > 0, 2, 1))
    P = np.zeros((len(F), 3))
    P[np.arange(len(F)), pred] = 1.0
    return P


# ----------------------------------------------------------------------------- B1 logistic regression
LR_GRID = [0.01, 0.1, 1.0, 10.0]


def lr_model(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def full_proba(model, X: np.ndarray) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    out[:, model[-1].classes_] = p
    return out


# ----------------------------------------------------------------------------- metrics
def window_metrics(P: np.ndarray, y: np.ndarray) -> dict:
    pred = P.argmax(1)
    from pave.evaluation.metrics import ece
    Pc = np.clip(P, 1e-12, 1)
    Pc = Pc / Pc.sum(1, keepdims=True)
    return {"n": int(len(y)), "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "bal_acc": float(balanced_accuracy_score(y, pred)), "ece": float(ece(P, y)),
            "nll": float(log_loss(y, Pc, labels=[0, 1, 2]))}


def bearing_table(P: np.ndarray, y: np.ndarray, bearings: np.ndarray, settings: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame(P, columns=["pH", "pIR", "pOR"])
    df["y"], df["b"] = y, bearings
    df["win_ok"] = P.argmax(1) == y
    g = df.groupby("b").agg(pH=("pH", "mean"), pIR=("pIR", "mean"), pOR=("pOR", "mean"), y=("y", "first"),
                            frac_windows_correct=("win_ok", "mean"), n_windows=("y", "size"))
    g["pred"] = g[["pH", "pIR", "pOR"]].to_numpy().argmax(1)
    g["correct"] = g.pred == g.y
    return g.reset_index().rename(columns={"b": "bearing"})


def bearing_metrics(bt: pd.DataFrame) -> dict:
    y, p = bt.y.to_numpy(), bt.pred.to_numpy()
    return {"n_bearings": int(len(y)), "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
            "acc": float(np.mean(y == p)), "bal_acc": float(balanced_accuracy_score(y, p)),
            "recall": [float(np.mean(p[y == k] == k)) if (y == k).any() else float("nan") for k in range(3)],
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            "failures": bt[~bt.correct].bearing.tolist()}


def stratified_bootstrap(bt: pd.DataFrame, stat, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [bt[bt.y == k].index.to_numpy() for k in range(3)]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals.append(stat(bt.loc[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    from scipy.stats import binomtest
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "exact_p": p}


def bearing_info(code: str) -> dict:
    b = BEARINGS[code]
    return {"group": b["group"], "method": b["method"], "extent": b["extent"]}
