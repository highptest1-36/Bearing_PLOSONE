"""Build the PLOS ONE manuscript PDFs from the single source file manuscript/pavenet_main.tex.

pavenet_main.pdf         submission version, captions only (PLOS: figures are uploaded separately as TIFF files)
pavenet_main_review.pdf  internal reading copy; a temporary copy of the source with the figures inserted below each caption
S1_Table.pdf ... S5_Table.pdf  supporting tables, one file per item
The source file itself is not modified by this script.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assemble_clean_tex import typography  # noqa: E402

MS = ROOT / "manuscript"
PKG = ROOT / "FINAL_PLOS_ONE_PACKAGE"
TPL = ROOT / "00_AUTHOR_GUIDELINES/PLOS_latex_template"


def run(cmd):
    r = subprocess.run(cmd, cwd=MS, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.stdout


def latex(main, job):
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={job}", main])
    run(["bibtex", job])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={job}", main])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={job}", main])
    log = (MS / f"{job}.log").read_text(encoding="utf-8", errors="replace")
    warn = [l for l in log.splitlines() if ("undefined" in l.lower() and "reference" in l.lower()) or "Citation" in l and "undefined" in l]
    over = [l for l in log.splitlines() if l.startswith("Overfull \\hbox")]
    print(job, "->", (MS / f"{job}.pdf").exists(), "| undefined refs/cites:", len(warn), "| overfull hboxes:", len(over))
    for l in warn[:10] + over[:10]:
        print("   ", l)


def review_copy():
    """Insert each figure below its caption in a temporary copy of the source."""
    src = (MS / "pavenet_main.tex").read_text(encoding="utf-8")

    def add(m):
        n = m.group(2)
        return m.group(1) + f"\\centering\\includegraphics[width=\\linewidth]{{figures/Fig{n}.png}}\n\\label{{fig{n}}}"
    out, k = re.subn(r"(\\begin\{figure\}\[!h\]\n\\caption\{.*?\}\n)\\label\{fig(\d)\}", add, src, flags=re.S)
    assert k == 8, k
    (MS / "pavenet_main_review.tex").write_text(out, encoding="utf-8")


def main():
    (MS / "tables").mkdir(exist_ok=True)
    (MS / "figures").mkdir(exist_ok=True)
    for p in (PKG / "tables").glob("*.tex"):
        (MS / "tables" / p.name).write_text(typography(p.read_text(encoding="utf-8")), encoding="utf-8")
    for p in (PKG / "figures").glob("*.png"):
        shutil.copy2(p, MS / "figures" / p.name)
    shutil.copy2(ROOT / "02_LITERATURE/verified_references.bib", MS / "verified_references.bib")
    shutil.copy2(TPL / "plos2025.bst", MS / "plos2025.bst")
    latex("pavenet_main.tex", "pavenet_main")
    review_copy()
    latex("pavenet_main_review.tex", "pavenet_main_review")
    (MS / "pavenet_main_review.tex").unlink()
    head = (MS / "supporting_tables.tex").read_text(encoding="utf-8").split("\\begin{document}")[0]
    for n, (tab, has_cites) in enumerate(SI_TABLES, start=1):
        job = f"S{n}_Table"
        body = (f"\\begin{{document}}\n\\setcounter{{table}}{{{n - 1}}}\n\\input{{tables/{tab}}}\n"
                + ("\\bibliography{verified_references}\n" if has_cites else "") + "\\end{document}\n")
        (MS / f"{job}.tex").write_text(head + body, encoding="utf-8")
        if has_cites:
            latex(f"{job}.tex", job)
        else:
            for _ in range(2):
                run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-jobname={job}", f"{job}.tex"])
            print(job, "->", (MS / f"{job}.pdf").exists())


SI_TABLES = [("S1_Table_related_work", True), ("S_Table_paired_differences", False), ("S_Table_external_secondary", False),
             ("S_Table_external_selective", False), ("S_Table_external_pev", False)]


if __name__ == "__main__":
    sys.exit(main())
