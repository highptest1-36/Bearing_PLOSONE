"""Build Phase 2 window caches.  python -I scripts/build_windows.py --dataset kaist|pu"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pave.data.read_mat import read_vibration_mat  # noqa: E402
from pave.data.windowing import build_kaist, build_pu  # noqa: E402

OUT = ROOT / "results" / "phase2" / "windows"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["kaist", "pu"])
    a = ap.parse_args()
    t0 = time.time()
    if a.dataset == "kaist":
        build_kaist(ROOT / "03_MANIFESTS" / "recordings.csv", OUT / "kaist",
                    lambda rel: read_vibration_mat(ROOT / rel, dtype=np.float32))
    else:
        m = ROOT / "03_MANIFESTS" / "paderborn"
        build_pu(m / "pu_files.csv", m / "pu_bearings.csv", ROOT / "results" / "phase2" / "cache_pu" / "vib", OUT / "pu",
                 log=lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True))
    print(f"{a.dataset} windows built in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
