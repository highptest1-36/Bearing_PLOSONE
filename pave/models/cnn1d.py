"""Compact 1-D CNN baseline (protocol V2 §7)."""
from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int, k: int, stride: int = 1, pool: bool = True) -> nn.Sequential:
    layers = [nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False), nn.BatchNorm1d(cout),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(in_ch, w, 64, stride=8),
            _block(w, w, 7),
            _block(w, 2 * w, 7),
            _block(2 * w, 2 * w, 7),
            _block(2 * w, 4 * w, 7),
            _block(4 * w, 4 * w, 7, pool=False),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
