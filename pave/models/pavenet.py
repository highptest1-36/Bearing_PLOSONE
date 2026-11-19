"""PAVE-Net (Candidate B, physics-anchored) and the concept-bottleneck comparator (Candidate A).

Protocol V2.2 §5 / V2.3. Inputs per 1-s window:
  zm   : [z (8), m (8)]  standardised V3 physics concepts + validity masks of the 4-s measurement (no NaN)
  q    : 7 standardised reliability features (label-free evidence quality)
  raw  : 1-s vibration (C, L) for the latent encoder

PAVE-Net logits:
  l = W_p zm (+ W_q q if head_q) + b_p  +  gamma * (1 - g) * B * tanh(W_l h / B),   g = sigmoid(u^T q + b_u)
  mode "physics"  : gamma = 0 (no latent; B4 / A1, or A2 with head_q)
  mode "always"   : g == 0  (A3: latent always on, still bounded)
  mode "gated"    : full model (A4)
Loss = CE(l, y) + beta * mean(1 - g)    (beta = 0.05; only in gated mode)
"""
from __future__ import annotations

import torch
from torch import nn

from .cnn1d import CNN1D


class LatentEncoder(nn.Module):
    """CNN-1D of B3 without its classification head -> 128-d embedding."""

    def __init__(self, in_ch: int = 1, width: int = 32):
        super().__init__()
        base = CNN1D(in_ch, 3, width=width)
        self.features = base.features
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.dim = 4 * width

    def forward(self, x):
        return self.pool(self.features(x))


class PAVENet(nn.Module):
    def __init__(self, n_zm: int = 16, n_q: int = 7, n_classes: int = 3, mode: str = "gated", head_q: bool = False,
                 bound: float = 3.0, gamma: float = 1.0, in_ch: int = 1):
        super().__init__()
        assert mode in ("physics", "always", "gated")
        self.mode, self.head_q, self.bound, self.gamma = mode, head_q, bound, gamma
        self.phys = nn.Linear(n_zm + (n_q if head_q else 0), n_classes)
        if mode != "physics":
            self.enc = LatentEncoder(in_ch)
            self.lat = nn.Linear(self.enc.dim, n_classes)
        if mode == "gated":
            self.gate = nn.Linear(n_q, 1)

    def forward(self, zm, q, raw=None, return_parts: bool = False):
        lp = self.phys(torch.cat([zm, q], 1) if self.head_q else zm)
        g = torch.ones(zm.shape[0], 1, device=zm.device)
        ll = torch.zeros_like(lp)
        h = None
        if self.mode != "physics":
            h = self.enc(raw)
            ll = self.bound * torch.tanh(self.lat(h) / self.bound)
            g = torch.sigmoid(self.gate(q)) if self.mode == "gated" else torch.zeros_like(g)
        logits = lp + self.gamma * (1 - g) * ll
        if return_parts:
            return logits, {"phys": lp, "latent": ll, "gate": g, "h": h}
        return logits

    def reg_loss(self, parts, beta: float = 0.05):
        if self.mode != "gated":
            return torch.zeros((), device=parts["phys"].device)
        return beta * (1 - parts["gate"]).mean()


class ConceptBottleneck(nn.Module):
    """Candidate A: raw -> CNN -> predicted standardised concepts c_hat (8) -> linear classifier (c_hat only)."""

    def __init__(self, n_concepts: int = 8, n_classes: int = 3, in_ch: int = 1):
        super().__init__()
        self.enc = LatentEncoder(in_ch)
        self.to_c = nn.Linear(self.enc.dim, n_concepts)
        self.cls = nn.Linear(n_concepts, n_classes)

    def forward(self, raw, return_parts: bool = False):
        h = self.enc(raw)
        c = self.to_c(h)
        logits = self.cls(c)
        if return_parts:
            return logits, {"c_hat": c, "h": h}
        return logits


def masked_concept_loss(c_hat, z, m, delta: float = 1.0):
    """Huber loss over valid concepts only; returns 0 when a batch has no valid concept."""
    l = nn.functional.huber_loss(c_hat, z, reduction="none", delta=delta) * m
    return l.sum() / m.sum().clamp_min(1.0)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
