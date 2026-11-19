"""Small MLP over measured physics concepts (Phase 3A, B2)."""
from __future__ import annotations

from torch import nn


class ConceptMLP(nn.Module):
    def __init__(self, n_in: int, n_classes: int = 3, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)
