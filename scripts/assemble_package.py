"""Assemble FINAL_PLOS_ONE_PACKAGE/ (no raw third-party data, no upload). Run after build_manuscript.py."""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "FINAL_PLOS_ONE_PACKAGE"
DOCS = ROOT / "docs"

TABLE_MAP = [("Table1_datasets", "Table 1"), ("Table2_splits", "Table 2"), ("Table_config", "Table 3"), ("Table3_baselines", "Table 4"),
             ("Table4_ablation", "Table 5"), ("Table6_pev", "Table 6"), ("Table7_selective", "Table 7"), ("Table5_concepts_probes", "Table 8"),
             ("Table8_external", "Table 9"), ("Table9_efficiency", "Table 10"), ("S1_Table_related_work", "S1 Table"),
             ("S_Table_paired_differences", "S2 Table"), ("S_Table_external_secondary", "S3 Table"), ("S_Table_external_selective", "S4 Table"),
             ("S_Table_external_pev", "S5 Table")]


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def zipdir(zpath: Path, files: list[Path], base: Path):
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.is_dir():
                for g in sorted(f.rglob("*")):
                    if g.is_file() and "__pycache__" not in g.parts:
                        z.write(g, g.relative_to(base).as_posix())
            else:
                z.write(f, f.relative_to(base).as_posix())


def main():
    ms = ROOT / "manuscript"
    # manuscript
    for n in ("pavenet_main.tex", "pavenet_main.pdf", "pavenet_main_review.pdf", "supporting_tables.tex", "supporting_tables.pdf",
              "S1_Table.pdf", "S2_Table.pdf", "S3_Table.pdf", "S4_Table.pdf", "S5_Table.pdf",
              "verified_references.bib", "plos2025.bst", "submission_statements.md"):
        copy(ms / n, PKG / "manuscript" / n)
    shutil.rmtree(PKG / "manuscript" / "sections", ignore_errors=True)  # v3: single source file, no sections
    copy(ms / "tables", PKG / "manuscript" / "tables")
    # tables map
    (PKG / "tables" / "TABLE_MAP.md").write_text("# Table numbering in the compiled manuscript\n\n| File | Number in PDF |\n|---|---|\n" +
                                                 "\n".join(f"| {f}.tex / .csv | {n} |" for f, n in TABLE_MAP) + "\n", encoding="utf-8")
    # protocols
    prot = sorted(DOCS.glob("RESEARCH_PROTOCOL_V2*.md")) + [DOCS / "EXTERNAL_VALIDATION_PROTOCOL_V1.md"] + sorted(DOCS.glob("PROTOCOL_FREEZE*.txt"))
    for p in prot:
        copy(p, PKG / "protocols" / p.name)
    copy(ROOT / "results/phase4/EVALUATED.lock", PKG / "protocols" / "EVALUATED.lock")
    # supporting information files
    si = PKG / "supporting_information"
    zipdir(si / "S1_File.zip", sorted(DOCS.glob("RESEARCH_PROTOCOL_V2*.md")) + [p for p in sorted(DOCS.glob("PROTOCOL_FREEZE*.txt")) if "EV" not in p.name], ROOT)
    zipdir(si / "S2_File.zip", [DOCS / "EXTERNAL_VALIDATION_PROTOCOL_V1.md", DOCS / "PROTOCOL_FREEZE_EV_V1.txt", ROOT / "results/phase4/EVALUATED.lock"], ROOT)
    zipdir(si / "S3_File.zip", [ROOT / "scripts/audit_foldlocal_selection.py", ROOT / "scripts/audit_prevalence.py",
                               ROOT / "results/phase5_audit"], ROOT)
    (si / "S1-S5_Tables.pdf").unlink(missing_ok=True)  # superseded: PLOS wants one file per SI item
    for n in range(1, 6):
        copy(ms / f"S{n}_Table.pdf", si / f"S{n}_Table.pdf")
    # reports
    for n in ("DATA_AUDIT_REPORT.md", "DATA_FEASIBILITY_ASSESSMENT.md", "PHASE2_REPORT.md", "PHASE3A_REPORT.md", "PHASE3B_DESIGN.md",
              "PHASE3B_REPORT.md", "PHASE4_EXTERNAL_VALIDATION_REPORT.md", "DECISION_LOG.md", "RESEARCH_PROGRESS.md",
              "PRE_SUBMISSION_METHOD_AUDIT.md", "CLAIM_EVIDENCE_MATRIX.md", "REVISION_PLAN.md", "REVIEWER_RESPONSE_MEMO.md",
              "PLOS_COMPLIANCE_CHECKLIST.md"):
        copy(DOCS / n, PKG / "reports" / n)
    copy(ROOT / "02_LITERATURE/verified_references_report.md", PKG / "reports" / "verified_references_report.md")
    # review
    copy(DOCS / "FINAL_REVIEWER_AUDIT.md", PKG / "review" / "FINAL_REVIEWER_AUDIT.md")
    copy(ms / "submission_statements.md", PKG / "review" / "submission_statements_DRAFT.md")
    copy(ms / "COVER_LETTER_DRAFT.md", PKG / "review" / "COVER_LETTER_DRAFT.md")
    copy(DOCS / "audit" / "PLOS_REQUIREMENTS_SOURCE_2026-11-19.md", PKG / "review" / "PLOS_REQUIREMENTS_SOURCE_2026-11-19.md")
    for n in ("manuscript_v1_to_v2.diff", "tracked_changes_v1_to_v2.html"):
        copy(ROOT / "manuscript_archive/revision_2026-11-19" / n, PKG / "review" / "revision_v1_to_v2" / n)
    copy(ROOT / "manuscript_archive/v1_2026-11-19_pre_audit/pavenet_main_review.pdf", PKG / "review" / "revision_v1_to_v2" / "pavenet_main_review_v1.pdf")
    for n in ("manuscript_v2_to_v3.diff", "tracked_changes_v2_to_v3.html"):
        copy(ROOT / "manuscript_archive/revision_2026-11-19_style" / n, PKG / "review" / "revision_v2_to_v3" / n)
    copy(ROOT / "manuscript_archive/v2_2026-11-19_pre_style_revision/pavenet_main_review.pdf", PKG / "review" / "revision_v2_to_v3" / "pavenet_main_review_v2.pdf")
    # reproducibility
    rp = PKG / "reproducibility"
    for n in ("REPRODUCE.md", "requirements.txt", "environment_report.txt", "README.md"):
        copy(ROOT / n, rp / n)
    zipdir(rp / "code_snapshot.zip", [ROOT / "pave", ROOT / "scripts", ROOT / "tests", ROOT / "03_MANIFESTS"], ROOT)
    for src, dst in (("results/phase3b/tables", "results/phase3b_tables"), ("results/phase3b/pev", "results/phase3b_pev"),
                     ("results/phase4/tables", "results/phase4_tables"), ("results/phase4/pev", "results/phase4_pev"),
                     ("results/phase4/posthoc", "results/phase4_posthoc"), ("results/phase5_audit", "results/phase5_audit_posthoc")):
        copy(ROOT / src, rp / dst)
    for n in ("ottawa_audit.csv", "ottawa_speed_audit.csv", "ottawa_segments4s.csv"):
        copy(ROOT / "results/phase4" / n, rp / "results/phase4_audit" / n)
    copy(ROOT / "results/phase3b/checkpoints.csv", rp / "checkpoints" / "checkpoints_phase3b.csv")
    copy(ROOT / "results/phase4/tables/checkpoints_evaluated.csv", rp / "checkpoints" / "checkpoints_FINAL_evaluated_on_ottawa.csv")
    (rp / "README_REPRODUCIBILITY.md").write_text(
        "# Reproducibility package\n\n"
        "- `code_snapshot.zip`: `pave/`, `scripts/`, `tests/`, `03_MANIFESTS/` (split manifests and SHA-256 of the original data files).\n"
        "- `REPRODUCE.md`: commands for every phase; `requirements.txt`, `environment_report.txt`: software versions "
        "(Python 3.10.6, PyTorch 2.7.1+cu126, NVIDIA RTX 3060 Laptop 6 GB).\n"
        "- `results/`: all result tables from which the manuscript tables and figures are generated.\n"
        "- `checkpoints/`: model, fold, seed and SHA-256 of every checkpoint. The checkpoint files themselves (results/phase3b/checkpoints/) "
        "are kept locally and are **not** included or uploaded; they can be shared on request or deposited by the authors.\n"
        "- Raw datasets are third-party and are **not redistributed**; obtain them from the original sources (see `REPRODUCE.md`) and "
        "verify them with the hashes in `03_MANIFESTS/` and `results/phase4_audit/ottawa_audit.csv`.\n"
        "- The Ottawa evaluation is single-use: the evaluation script refuses to run when `results/phase4/EVALUATED.lock` exists.\n",
        encoding="utf-8")
    # checksums
    lines = []
    for f in sorted(PKG.rglob("*")):
        if f.is_file() and f.name != "SHA256SUMS.txt":
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(PKG).as_posix()}")
    (rp / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = sum(1 for f in PKG.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in PKG.rglob("*") if f.is_file()) / 1e6
    print(f"package: {n} files, {size:.1f} MB")


if __name__ == "__main__":
    main()
