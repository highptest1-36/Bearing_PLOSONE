"""Dataset location, file inventory, hashing and archive provenance checks.

Raw data are treated as READ-ONLY: every function here opens files in binary
read mode only.
"""
from __future__ import annotations

import hashlib
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "01_RAW_DATA"
DATASET_DIRNAME = ("Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine "
                   "Under Varying Load Conditions for Fault Diagnosis")
DATASET_DIR = RAW_ROOT / DATASET_DIRNAME
DATASET_DOI = "10.17632/ztmf3m7h5x.6"  # stated in the project blueprint; local copy has no version file
PAPER_DOI = "10.1016/j.dib.2023.109049"

MODALITY_DIRS = {"vibration": "vibration", "current_temp": "current,temp", "acoustic": "acoustic"}
ARCHIVES = {"vibration": "vibration.zip", "current_temp": "current,temp.zip", "acoustic": "acoustic.zip"}


@dataclass
class FileHash:
    sha256: str
    crc32: int
    size: int


def hash_file(path: Path, chunk: int = 8 * 1024 * 1024) -> FileHash:
    h = hashlib.sha256()
    crc = 0
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            crc = zlib.crc32(b, crc)
            size += len(b)
    return FileHash(h.hexdigest(), crc & 0xFFFFFFFF, size)


def zip_index(zpath: Path) -> dict[str, zipfile.ZipInfo]:
    """Map basename -> ZipInfo for every file entry of an archive (no extraction)."""
    with zipfile.ZipFile(zpath) as z:
        return {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}


def relpath(p: Path) -> str:
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()
