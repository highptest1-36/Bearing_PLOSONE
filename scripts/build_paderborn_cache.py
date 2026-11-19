"""Build the Paderborn inventory + vibration cache from the ORIGINAL .rar archives (read-only).

For each bearing archive:
  1. SHA-256 of the archive; list entries; check 80 .mat + fact sheet + measuring log.
  2. Extract the two PDFs permanently to 01_RAW_DATA/Paderborn_KAt_BearingDataCenter/docs/<code>/.
  3. Extract the .mat files to a temporary scratch folder, record SHA-256 + metadata of every file,
     save vibration_1 as float32 .npy to results/phase2/cache_pu/vib/<code>/<stem>.npy, then delete the
     temporary .mat copies (the .rar archive remains the original).

Run with an isolated interpreter (downloaded data):  python -I scripts/build_paderborn_cache.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pave.data.paderborn import BEARINGS, FS_VIB, geometry_or_default, parse_fact_sheet, parse_pu_filename, read_pu_mat  # noqa: E402

PU = ROOT / "01_RAW_DATA" / "Paderborn_KAt_BearingDataCenter"
ARCH = PU / "archives"
DOCS = PU / "docs"
CACHE = ROOT / "results" / "phase2" / "cache_pu"
MAN = ROOT / "03_MANIFESTS" / "paderborn"
TAR = r"C:\Windows\System32\tar.exe"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def process(code: str, tmp_root: Path) -> tuple[list[dict], dict]:
    arc = ARCH / f"{code}.rar"
    entries = subprocess.run([TAR, "-tf", str(arc)], capture_output=True, text=True, check=True).stdout.split()
    mats = sorted(e for e in entries if e.endswith(".mat"))
    pdfs = sorted(e for e in entries if e.endswith(".pdf"))
    d = DOCS / code
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([TAR, "-xf", str(arc), "-C", str(d), "--strip-components", "1", *pdfs], check=True)
    fs_path = d / f"{code}.pdf"
    fact = parse_fact_sheet(fs_path) if fs_path.exists() else {}
    geom, defaulted = geometry_or_default(fact)
    tmp = Path(tempfile.mkdtemp(prefix=f"pu_{code}_", dir=tmp_root))
    try:
        subprocess.run([TAR, "-xf", str(arc), "-C", str(tmp), *mats], check=True)
        out = CACHE / "vib" / code
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for e in mats:
            p = tmp / e
            stem = Path(e).stem
            setting, c2, k = parse_pu_filename(stem)
            assert c2 == code, (c2, code)
            try:
                r = read_pu_mat(p)
            except Exception as exc:  # corrupted file: record and continue (never silently)
                rows.append({"bearing": code, "setting": setting, "measurement": k, "file_in_archive": e,
                             "mat_sha256": sha256(p), "mat_size": p.stat().st_size, "n_vib": 0,
                             "read_error": repr(exc)[:200]})
                log(f"{code}: UNREADABLE {e}: {exc!r}")
                continue
            x = r.vibration
            np.save(out / f"{stem}.npy", x)
            sp = r.speed_rpm.astype(float)
            rows.append({
                "bearing": code, "setting": setting, "measurement": k, "file_in_archive": e,
                "mat_sha256": sha256(p), "mat_size": p.stat().st_size,
                "n_vib": int(x.size), "duration_s": (x.size - 1) / FS_VIB,
                "vib_nan": int(np.isnan(x).sum()), "vib_inf": int(np.isinf(x).sum()),
                "vib_rms": float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), "vib_std": float(x.std()),
                "vib_max_abs": float(np.abs(x).max()),
                "vib_longest_const": int(np.max(np.diff(np.flatnonzero(np.concatenate(([1], np.diff(x) != 0, [1])))))),
                "speed_rpm_mean": float(np.nanmean(sp)), "speed_rpm_std": float(np.nanstd(sp)),
                "torque_mean": float(np.nanmean(r.torque)), "force_mean": float(np.nanmean(r.force)),
                "temp_mean": float(np.nanmean(r.temp)),
                "datetime": r.datetime.isoformat() if r.datetime else None, "signals": "|".join(r.raw_names),
            })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    b = BEARINGS[code]
    brow = {"bearing": code, **{k: v for k, v in b.items()}, "archive_sha256": sha256(arc),
            "archive_size": arc.stat().st_size, "n_entries": len(entries), "n_mat": len(mats),
            "pdfs": "|".join(Path(p).name for p in pdfs), **{f"fs_{k}": v for k, v in fact.items()},
            **{f"geom_{k}": v for k, v in geom.items()}, "geom_defaulted": defaulted}
    return rows, brow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--tmp", default=None, help="scratch folder for temporary extraction")
    a = ap.parse_args()
    MAN.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(a.tmp) if a.tmp else Path(tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    codes = sorted(BEARINGS)
    for code in codes:
        if a.only and code not in a.only:
            continue
        done = MAN / "per_bearing" / f"{code}.json"
        if done.exists():
            continue
        if not (ARCH / f"{code}.rar").exists():
            log(f"{code}: archive missing - skipped")
            continue
        t0 = time.time()
        rows, brow = process(code, tmp_root)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps({"files": rows, "bearing": brow}, indent=1, default=str))
        log(f"{code}: {len(rows)} files in {time.time()-t0:.0f}s")
    # assemble
    files, bearings = [], []
    for p in sorted((MAN / "per_bearing").glob("*.json")):
        d = json.loads(p.read_text())
        files += d["files"]
        bearings.append(d["bearing"])
    pd.DataFrame(files).to_csv(MAN / "pu_files.csv", index=False)
    pd.DataFrame(bearings).to_csv(MAN / "pu_bearings.csv", index=False)
    log(f"manifest: {len(bearings)} bearings, {len(files)} files")


if __name__ == "__main__":
    main()
