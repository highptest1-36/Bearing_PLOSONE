"""Minimal, deterministic-as-possible training loop (AdamW, cosine schedule, early stopping on val macro-F1)."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 25
    patience: int = 6
    amp: bool = True
    class_weights: bool = True


def _batches(n: int, bs: int, shuffle: bool, rng: np.random.Generator):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str, bs: int = 128, transform=None) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        if transform is not None:
            xb = transform(xb)
        xb = torch.as_tensor(np.ascontiguousarray(xb), dtype=torch.float32, device=device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out.append(torch.softmax(model(xb).float(), 1).cpu().numpy())
    return np.concatenate(out)


def fit(model: nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, n_classes: int,
        cfg: TrainConfig, seed: int, device: str, transform=None, log=print) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    if cfg.class_weights:
        cnt = np.bincount(ytr, minlength=n_classes).astype(float)
        w = torch.tensor(cnt.sum() / (n_classes * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    else:
        w = None
    lossf = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))
    best = {"f1": -1.0, "epoch": -1, "state": None}
    hist = []
    bad = 0
    t0 = time.time()
    for ep in range(cfg.max_epochs):
        model.train()
        tot = 0.0
        for b in _batches(len(Xtr), cfg.batch_size, True, rng):
            xb = Xtr[np.sort(b)] if isinstance(Xtr, np.memmap) else Xtr[b]
            yb = ytr[np.sort(b)] if isinstance(Xtr, np.memmap) else ytr[b]
            if transform is not None:
                xb = transform(xb)
            xb = torch.as_tensor(np.ascontiguousarray(xb), dtype=torch.float32, device=device)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg.amp and device.startswith("cuda")):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(b)
        sched.step()
        pv = predict(model, Xva, device, transform=transform)
        f1 = f1_score(yva, pv.argmax(1), labels=list(range(n_classes)), average="macro", zero_division=0)
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_macro_f1": float(f1)})
        if f1 > best["f1"]:
            best = {"f1": float(f1), "epoch": ep, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_val_macro_f1": best["f1"], "history": hist,
            "train_seconds": time.time() - t0}
