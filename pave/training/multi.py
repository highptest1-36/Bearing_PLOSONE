"""Generic AdamW trainer/predictor for Phase 3B models (seed set BEFORE model construction).

Data container: dict with
  raw : float16 array (N, C, L) or None   (1-s windows, z-scored on the fly with train statistics rs=(mu, sd))
  zm  : float32 (N, 16)  [z, m] of the window's 4-s measurement (or 1-s for B2)
  q   : float32 (N, 7)   standardised reliability features
  y   : int (N,)
kind: "cnn" (B3), "pave" (PAVENet; raw optional), "cbm" (ConceptBottleneck), "mlp" (ConceptMLP on zm)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from .engine import set_seed

LAMBDA_C = 1.0


def _class_w(y, n, device):
    cnt = np.bincount(y, minlength=n).astype(float)
    return torch.tensor(cnt.sum() / (n * np.maximum(cnt, 1)), dtype=torch.float32, device=device)


def _batch(D, idx, rs, device):
    out = {}
    if D.get("raw") is not None:
        x = D["raw"][idx].astype(np.float32)
        if rs is not None:
            x = (x - rs[0]) / rs[1]
        out["raw"] = torch.as_tensor(x, device=device)
    for k in ("zm", "q"):
        if D.get(k) is not None:
            out[k] = torch.as_tensor(D[k][idx], dtype=torch.float32, device=device)
    return out


def _forward(model, kind, b, parts=False):
    if kind == "cnn":
        return (model(b["raw"]), {}) if parts else model(b["raw"])
    if kind == "mlp":
        return (model(b["zm"]), {}) if parts else model(b["zm"])
    if kind == "pave":
        return model(b["zm"], b["q"], b.get("raw"), return_parts=parts)
    if kind == "cbm":
        return model(b["raw"], return_parts=parts)
    raise ValueError(kind)


def fit(make_model, kind: str, D: dict, idx: np.ndarray, epochs: int, lr: float, wd: float, bs: int, seed: int,
        device: str, rs=None, amp: bool = True, n_classes: int = 3) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = make_model().to(device)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    y = D["y"]
    lossf = nn.CrossEntropyLoss(weight=_class_w(y[idx], n_classes, device))
    use_amp = amp and device.startswith("cuda") and kind in ("cnn", "pave", "cbm")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(idx)
        for i in range(0, len(perm), bs):
            bi = np.sort(perm[i:i + bs])
            b = _batch(D, bi, rs, device)
            yb = torch.as_tensor(y[bi], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits, parts = _forward(model, kind, b, parts=True)
                loss = lossf(logits.float(), yb)
                if kind == "pave":
                    loss = loss + model.reg_loss(parts)
                if kind == "cbm":
                    from pave.models.pavenet import masked_concept_loss
                    z, m = b["zm"][:, :8], b["zm"][:, 8:]
                    loss = loss + LAMBDA_C * masked_concept_loss(parts["c_hat"].float(), z, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    info = {"train_seconds": time.time() - t0, "epochs": epochs, "lr": lr, "weight_decay": wd, "batch_size": bs,
            "seed": seed, "optimizer": "AdamW", "amp": use_amp,
            "peak_gpu_mem_MB": torch.cuda.max_memory_allocated() / 2 ** 20 if device.startswith("cuda") else None}
    return model, info


@torch.no_grad()
def predict(model, kind: str, D: dict, idx: np.ndarray, device: str, rs=None, bs: int = 256, parts: bool = False):
    model.eval()
    P, out = [], {}
    for i in range(0, len(idx), bs):
        bi = idx[i:i + bs]
        b = _batch(D, bi, rs, device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda") and kind in ("cnn", "pave", "cbm")):
            logits, pr = _forward(model, kind, b, parts=True)
        P.append(torch.softmax(logits.float(), 1).cpu().numpy())
        if parts:
            pr = dict(pr)
            pr["logits"] = logits
            for k, v in pr.items():
                if v is not None:
                    out.setdefault(k, []).append(v.float().cpu().numpy())
    P = np.concatenate(P)
    if parts:
        return P, {k: np.concatenate(v) for k, v in out.items()}
    return P
