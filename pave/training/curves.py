"""Training with per-epoch validation curves and fixed-epoch refits (protocol V2.1 §7).

``fit_curve``  : train for ``max_epochs`` and record validation probabilities after every epoch (inner loop).
``fit_fixed``  : train for exactly ``epochs`` epochs with no validation (outer refit; no early stopping on test).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .engine import predict, set_seed


def _class_weights(y: np.ndarray, n_classes: int, device: str) -> torch.Tensor:
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    return torch.tensor(cnt.sum() / (n_classes * np.maximum(cnt, 1)), dtype=torch.float32, device=device)


def _epoch(model, opt, scaler, lossf, X, y, bs, rng, device, transform, amp):
    model.train()
    idx = rng.permutation(len(X))
    for i in range(0, len(X), bs):
        b = np.sort(idx[i:i + bs])
        xb = X[b]
        if transform is not None:
            xb = transform(xb)
        xb = torch.as_tensor(np.ascontiguousarray(xb), dtype=torch.float32, device=device)
        yb = torch.as_tensor(y[b], dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp and device.startswith("cuda")):
            loss = lossf(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


def _setup(model, lr, wd, y, n_classes, device, amp, sched_epochs):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sched_epochs)
    lossf = nn.CrossEntropyLoss(weight=_class_weights(y, n_classes, device))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    return opt, sched, lossf, scaler


def fit_curve(make_model, Xtr, ytr, Xva, n_classes: int, lr: float, wd: float, bs: int, max_epochs: int,
              seed: int, device: str, transform=None, amp: bool = True, eval_every: int = 1) -> dict:
    """``make_model``: zero-argument factory, called AFTER seeding so the seed controls weight initialisation."""
    set_seed(seed)
    model = make_model()
    rng = np.random.default_rng(seed)
    opt, sched, lossf, scaler = _setup(model, lr, wd, ytr, n_classes, device, amp, max_epochs)
    probs = {}
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        _epoch(model, opt, scaler, lossf, Xtr, ytr, bs, rng, device, transform, amp)
        sched.step()
        if ep % eval_every == 0:
            probs[ep] = predict(model, Xva, device, transform=transform)
    return {"val_probs": probs, "seconds": time.time() - t0, "model": model}


def fit_fixed(make_model, Xtr, ytr, n_classes: int, lr: float, wd: float, bs: int, epochs: int, seed: int,
              device: str, transform=None, amp: bool = True, sched_epochs: int | None = None) -> dict:
    """``sched_epochs``: cosine period; set equal to the inner-loop max_epochs so the LR at epoch e matches the curve.
    ``make_model`` is called after seeding. Returns the trained model."""
    set_seed(seed)
    model = make_model()
    rng = np.random.default_rng(seed)
    opt, sched, lossf, scaler = _setup(model, lr, wd, ytr, n_classes, device, amp, sched_epochs or epochs)
    t0 = time.time()
    for _ in range(epochs):
        _epoch(model, opt, scaler, lossf, Xtr, ytr, bs, rng, device, transform, amp)
        sched.step()
    return {"seconds": time.time() - t0, "model": model}
