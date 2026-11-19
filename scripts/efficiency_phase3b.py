"""Efficiency: parameters, batch-1 latency (GPU and CPU) per 1-s window, 4-s physics extraction time, training cost."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pave.features import physics_v3 as V3  # noqa: E402
from pave.features.physics import Regions  # noqa: E402
from pave.models.cnn1d import CNN1D  # noqa: E402
from pave.models.mlp import ConceptMLP  # noqa: E402
from pave.models.pavenet import ConceptBottleneck, PAVENet, n_params  # noqa: E402
from pave.signal.envelope import envelope  # noqa: E402

MAKERS = {"B2": lambda: ConceptMLP(16, 3, 64, 0.1), "B3": lambda: CNN1D(1, 3), "B4": lambda: PAVENet(16, 7, 3, mode="physics"),
          "A2": lambda: PAVENet(16, 7, 3, mode="physics", head_q=True), "A3": lambda: PAVENet(16, 7, 3, mode="always"),
          "A4": lambda: PAVENet(16, 7, 3, mode="gated"), "A6": lambda: ConceptBottleneck(8, 3, 1)}


def time_it(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1e3), float(np.percentile(t, 95) * 1e3)


def main():
    rows = []
    for name, mk in MAKERS.items():
        for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
            m = mk().to(dev).eval()
            x = torch.randn(1, 1, 32000, device=dev)
            zm = torch.randn(1, 16, device=dev)
            q = torch.randn(1, 7, device=dev)

            def f():
                with torch.no_grad():
                    if name in ("B3", "A6"):
                        m(x)
                    elif name == "B2":
                        m(zm)
                    else:
                        m(zm, q, x if name in ("A3", "A4") else None)
                if dev == "cuda":
                    torch.cuda.synchronize()
            med, p95 = time_it(f)
            rows.append({"model": name, "device": dev, "params": n_params(m), "latency_ms_median": med, "latency_ms_p95": p95})
    # physics extraction (4-s envelope + V3) on CPU
    x = np.random.default_rng(0).standard_normal(256000)
    reg = Regions(centre={"BPFO": 0.0, "BPFI": 0.0}, half={"BPFO": 0.01, "BPFI": 0.01})

    def phys():
        e = envelope(x, 64000.0, (1000.0, 20000.0))
        w = np.hanning(len(e))
        E = np.abs(np.fft.rfft(e * w)) * 2 / w.sum()
        V3.features_v3(np.arange(4001) * 0.25, E[:4001], 25.0, {"BPFO": 3.05, "BPFI": 4.95}, reg)
    med, p95 = time_it(phys, n=30, warm=3)
    rows.append({"model": "V3 physics extraction (4-s, CPU)", "device": "cpu", "params": 0, "latency_ms_median": med, "latency_ms_p95": p95})
    E = pd.DataFrame(rows)
    ck = pd.read_csv(ROOT / "results/phase3b/checkpoints.csv")
    tr = ck.groupby("model").agg(train_seconds_median=("train_seconds", "median"), peak_gpu_MB_max=("peak_gpu_mem_MB", "max"))
    E = E.merge(tr, left_on="model", right_index=True, how="left")
    out = ROOT / "results/phase3b/tables/efficiency.csv"
    E.round(4).to_csv(out, index=False)
    print(E.round(3).to_string())


if __name__ == "__main__":
    main()
