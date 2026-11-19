"""Flat PLOS ONE submission folder PLOS_ONE_SUBMISSION/ (one level, no subfolders).

main.tex          manuscript source (copy of manuscript/pavenet_main.tex, bibliography renamed to references)
main.pdf          compiled manuscript (captions only; PLOS requires figures as separate files)
main.bbl          compiled bibliography, so that main.tex also compiles without running BibTeX
references.bib    bibliography without comment lines
plos2025.bst      PLOS bibliography style (from the official template)
Fig1.tif..Fig8.tif, S1_Table.pdf..S5_Table.pdf, S1_File.zip..S3_File.zip   upload items
The folder is rebuilt from scratch and compiled once to check it; auxiliary files are removed afterwards.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PLOS_ONE_SUBMISSION"
MS = ROOT / "manuscript"
PKG = ROOT / "FINAL_PLOS_ONE_PACKAGE"


def run(cmd):
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    tex = (MS / "pavenet_main.tex").read_text(encoding="utf-8")
    assert tex.count("\\bibliography{verified_references}") == 1
    (OUT / "main.tex").write_text(tex.replace("\\bibliography{verified_references}", "\\bibliography{references}"), encoding="utf-8")
    bib = (ROOT / "02_LITERATURE/verified_references.bib").read_text(encoding="utf-8").splitlines()
    clean, blank = [], False
    for ln in bib:
        if ln.lstrip().startswith("%"):
            continue
        if not ln.strip():
            if blank or not clean:
                continue
            blank = True
        else:
            blank = False
        clean.append(ln)
    (OUT / "references.bib").write_text("\n".join(clean).strip() + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "00_AUTHOR_GUIDELINES/PLOS_latex_template/plos2025.bst", OUT / "plos2025.bst")
    for n in range(1, 9):
        shutil.copy2(PKG / "figures" / f"Fig{n}.tif", OUT / f"Fig{n}.tif")
    for n in range(1, 6):
        shutil.copy2(MS / f"S{n}_Table.pdf", OUT / f"S{n}_Table.pdf")
    for n in range(1, 4):
        shutil.copy2(PKG / "supporting_information" / f"S{n}_File.zip", OUT / f"S{n}_File.zip")
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    run(["bibtex", "main"])
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    log = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
    undefined = [l for l in log.splitlines() if "undefined" in l.lower()]
    for ext in ("aux", "log", "blg", "out"):
        (OUT / f"main.{ext}").unlink(missing_ok=True)
    assert not [p for p in OUT.iterdir() if p.is_dir()], "subfolder present"
    print("undefined references/citations:", len(undefined))
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:16s} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
