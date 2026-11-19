"""PAVE-Net Phase 1 data audit (read-only on raw data).

Stages (run in order; per-file results are cached so stages can be re-run):

  inventory   recursive file listing, SHA-256, CRC32 vs. distributed zip archives
  vibration   per-recording vibration audit (stats, quality, speed, PSD, envelope evidence)
  tdms        per-file temperature + motor-current audit
  analyze     pairing/alignment, manifests, physics + temperature analyses, splits, figures

Usage:
  python scripts/audit_data.py --stage inventory
  python scripts/audit_data.py --stage vibration [--only 0Nm_BPFI_03] [--force]
  python scripts/audit_data.py --stage tdms
  python scripts/audit_data.py --stage analyze
  python scripts/audit_data.py --stage all
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pave.data.manifest import (ARCHIVES, DATASET_DIR, MODALITY_DIRS, PROJECT_ROOT, hash_file,  # noqa: E402
                                relpath, zip_index)

OUT = ROOT / "results" / "audit"
CACHE = OUT / "cache"
MAN = ROOT / "03_MANIFESTS"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stage_inventory(force: bool = False) -> None:
    import pandas as pd

    cache_file = CACHE / "hashes.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() and not force else {}
    rows = []
    zips = {k: zip_index(DATASET_DIR / v) for k, v in ARCHIVES.items() if (DATASET_DIR / v).exists()}
    all_files = sorted(p for p in DATASET_DIR.rglob("*") if p.is_file())
    for p in all_files:
        st = p.stat()
        key = relpath(p)
        c = cache.get(key)
        if not c or c["size"] != st.st_size or c["mtime"] != st.st_mtime:
            log(f"hashing {key} ({st.st_size/1e6:.0f} MB)")
            h = hash_file(p)
            c = {"sha256": h.sha256, "crc32": h.crc32, "size": h.size, "mtime": st.st_mtime}
            cache[key] = c
            cache_file.write_text(json.dumps(cache, indent=1))
        modality = next((k for k, d in MODALITY_DIRS.items() if p.parent.name == d), "archive" if p.suffix == ".zip" else "other")
        zi = zips.get(modality, {}).get(p.name)
        rows.append({
            "relative_path": key, "file_name": p.name, "modality_dir": modality, "extension": p.suffix,
            "size_bytes": st.st_size, "sha256": c["sha256"], "crc32": f"{c['crc32']:08x}",
            "in_archive": zi is not None,
            "archive_crc32": f"{zi.CRC:08x}" if zi else "",
            "archive_size": zi.file_size if zi else "",
            "archive_date": "%04d-%02d-%02d %02d:%02d:%02d" % zi.date_time if zi else "",
            "crc_match_archive": (zi.CRC == c["crc32"] and zi.file_size == st.st_size) if zi else None,
        })
    df = pd.DataFrame(rows)
    # archive entries not extracted
    missing = []
    for k, idx in zips.items():
        extracted = {r["file_name"] for r in rows if r["modality_dir"] == k}
        for name, zi in idx.items():
            if name not in extracted:
                missing.append({"archive": ARCHIVES[k], "entry": zi.filename, "size": zi.file_size})
    MAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(MAN / "file_inventory.csv", index=False)
    dups = df[df.duplicated("sha256", keep=False) & (df.extension != ".zip")].sort_values("sha256")
    prov = {
        "dataset_dir": relpath(DATASET_DIR),
        "n_files": len(df), "by_modality": df.groupby("modality_dir").size().to_dict(),
        "archive_entries_not_extracted": missing,
        "crc_mismatches": df[df.crc_match_archive == False].relative_path.tolist(),  # noqa: E712
        "files_not_in_archive": df[(df.in_archive == False) & (df.extension != ".zip")].relative_path.tolist(),  # noqa: E712
        "duplicate_content_groups": dups.groupby("sha256").relative_path.apply(list).tolist(),
        "archives": {k: {"file": v, "sha256": next((r["sha256"] for r in rows if r["file_name"] == v), None),
                         "n_entries": len(zips.get(k, {}))} for k, v in ARCHIVES.items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (MAN / "provenance.json").write_text(json.dumps(prov, indent=2))
    log(f"inventory: {len(df)} files; crc mismatches={len(prov['crc_mismatches'])}; "
        f"unextracted={len(missing)}; duplicate groups={len(prov['duplicate_content_groups'])}")


def stage_vibration(only: list[str] | None, force: bool) -> None:
    from pave.audit.vibration_audit import audit_vibration_file

    files = sorted((DATASET_DIR / MODALITY_DIRS["vibration"]).glob("*.mat"))
    for p in files:
        if only and p.stem not in only:
            continue
        if (CACHE / f"vib_{p.stem}.json").exists() and not force:
            continue
        t0 = time.time()
        try:
            audit_vibration_file(p, CACHE)
            log(f"vibration {p.name} done in {time.time()-t0:.0f}s")
        except Exception as e:  # record the failure, continue with other files
            (CACHE / f"vib_{p.stem}.error.txt").write_text(repr(e))
            log(f"vibration {p.name} FAILED: {e!r}")


def stage_tdms(only: list[str] | None, force: bool) -> None:
    from pave.audit.tdms_audit import audit_tdms_file

    files = sorted((DATASET_DIR / MODALITY_DIRS["current_temp"]).glob("*.tdms"))
    for p in files:
        if only and p.stem not in only:
            continue
        if (CACHE / f"tdms_{p.stem}.json").exists() and not force:
            continue
        t0 = time.time()
        try:
            audit_tdms_file(p, CACHE)
            log(f"tdms {p.name} done in {time.time()-t0:.0f}s")
        except Exception as e:
            (CACHE / f"tdms_{p.stem}.error.txt").write_text(repr(e))
            log(f"tdms {p.name} FAILED: {e!r}")


def stage_analyze() -> None:
    from pave.audit import analysis

    analysis.run_all(CACHE, OUT, MAN, ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["inventory", "vibration", "tdms", "analyze", "all"])
    ap.add_argument("--only", nargs="*", help="file stems to process (vibration/tdms stages)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.stage in ("inventory", "all"):
        stage_inventory(a.force)
    if a.stage in ("tdms", "all"):
        stage_tdms(a.only, a.force)
    if a.stage in ("vibration", "all"):
        stage_vibration(a.only, a.force)
    if a.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()
