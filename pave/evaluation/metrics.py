"""Classification metrics at window and group (recording / bearing) level."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def basic(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "n": int(len(y)), "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else float("nan"),
        "recall_per_class": [float(v) for v in recall_score(y, pred, labels=labels, average=None, zero_division=0)],
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def evaluate(probs: np.ndarray, y: np.ndarray, groups: dict[str, np.ndarray], n_classes: int) -> dict:
    """Window-level metrics + group-level metrics (mean probability per group, majority label check)."""
    pred = probs.argmax(1)
    out = {"window": {**basic(y, pred, n_classes), "ece": ece(probs, y), "nll": float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))))}}
    for gname, g in groups.items():
        df = pd.DataFrame(probs, columns=[f"p{i}" for i in range(n_classes)])
        df["g"] = g
        df["y"] = y
        agg = df.groupby("g").agg({**{f"p{i}": "mean" for i in range(n_classes)}, "y": "first"})
        gy = agg["y"].to_numpy().astype(int)
        gp = agg[[f"p{i}" for i in range(n_classes)]].to_numpy().argmax(1)
        out[gname] = {**basic(gy, gp, n_classes), "groups": agg.index.tolist(), "group_pred": gp.tolist(), "group_true": gy.tolist()}
    return out
