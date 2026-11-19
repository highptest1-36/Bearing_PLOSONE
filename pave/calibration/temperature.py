"""Temperature scaling (Guo et al. 2017) fitted on held-out (out-of-fold) probabilities."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-lp[np.arange(len(y)), y].mean())


def probs_to_logits(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, 1e-12, 1.0))


def fit_temperature(P_heldout: np.ndarray, y: np.ndarray) -> float:
    """Return T in [0.05, 20] minimising NLL of softmax(log P / T) on held-out data."""
    L = probs_to_logits(P_heldout)
    r = minimize_scalar(lambda t: _nll(L, y, t), bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    z = probs_to_logits(P) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)
