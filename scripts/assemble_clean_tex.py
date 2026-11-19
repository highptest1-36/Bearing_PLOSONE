"""Assemble the single, comment-free manuscript file manuscript/pavenet_main.tex (style revision of 2026-11-19).

Input: the hand-written text in manuscript_archive/revision_2026-11-19_style/pavenet_main_template.tex, whose
@@TABLE:name@@ placeholders are replaced by the generated tables (FINAL_PLOS_ONE_PACKAGE/tables/name.tex). The table
values are copied verbatim from the generated files; only captions, notes and typography are changed: no bold header
cells, no custom rule commands, no comment lines and no en or em dashes.
Usage: python scripts/assemble_clean_tex.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "manuscript_archive/revision_2026-11-19_style/pavenet_main_template.tex"
TAB = ROOT / "FINAL_PLOS_ONE_PACKAGE/tables"
OUT = ROOT / "manuscript/pavenet_main.tex"

CAPTIONS = {
    "Table1_datasets": ("Datasets and their roles.",
                        "H, healthy; IR, inner race; OR, outer race. PU-BD5 comprises the healthy bearings K001 to K005 and bearings "
                        "with damage from accelerated lifetime tests (IR: KI04, KI14, KI16, KI18, KI21; OR: KA04, KA15, KA16, KA22, KA30)."),
    "Table2_splits": ("Bearing-disjoint evaluation design.",
                      "Protocol hashes (SHA-256 prefixes): V2 60d0c66a, V2.1 15343677, V2.2 078d726b, V2.3 6b4e4fdd, V2.4 dd758b7f, "
                      "EV-V1 bb297d52. Within an outer fold, no data of a test bearing is used for training, calibration or threshold "
                      "selection. The inclusion decision used inner OOF predictions pooled over all outer folds."),
    "Table_config": ("Fixed training configurations (protocol V2.2).",
                     "Seeds 42, 43 and 44; class-weighted cross-entropy. lr, learning rate; wd, weight decay; AMP, automatic mixed precision."),
    "Table3_baselines": ("Bearing-level results on PU-BD5.",
                         "Outer-test results of three-seed ensembles on 15 bearings (5 H, 5 IR, 5 OR). CI, class-stratified bootstrap over "
                         "bearings; FA, false-alarm rate. NLL and ECE are window-level values after temperature scaling. Physics models use "
                         "concepts from the 64-kHz signal, and CNN-based models use 1-s windows at 32 kHz. Paired differences are given in S2 Table."),
    "Table4_ablation": ("Ablation study on PU-BD5.",
                        "Weak-evidence bearings: KA15, KA22, KA30, KI14, KI16, KI21; strong-evidence bearings: KA04, KA16, KI04, KI18. "
                        "Inner-OOF values pool 60 bearing predictions over the five outer folds and formed the basis of the inclusion rule. "
                        "A5 (A4 with PE-SD) is reported in Table 7."),
    "Table6_pev": ("Physics-evidence verification on the 10 faulty PU-BD5 bearings.",
                   "Own fault family, suppression strength 1.0. Bearing means over 20 records per bearing; bootstrap CI over bearings. "
                   "Delta p, change in the probability of the true class; T < C, number of bearings with a lower value under targeted "
                   "suppression than under the control."),
    "Table7_selective": ("Selective diagnosis on PU-BD5.",
                         "R1, maximum probability; R2, negative entropy; R3, evidence clarity; R4, physics-evidence-aware score fitted on "
                         "inner OOF predictions. Risk at coverages 12/15 and 9/15, with the number of accepted errors in parentheses."),
    "Table5_concepts_probes": ("Nuisance information in physics concepts and learned representations.",
                               "Bearings unseen by the models of each fold. Mean [min, max] over the five outer folds. Bearing-level "
                               "targets: leave-one-bearing-out AUROC (chance 0.5); setting: four-class window accuracy (chance 0.25)."),
    "Table8_external": ("Single external evaluation on Ottawa.",
                        "30 primary recordings from 20 bearings. * Primary model, fixed before Ottawa was analysed. CI, bootstrap over "
                        "bearings; Localis., IR/OR accuracy among detected faults; sens., pre-specified speed-uncertainty analysis; "
                        "rec.-norm., pre-specified per-recording standardization."),
    "Table9_efficiency": ("Computational cost.",
                          "Batch-1 inference on one 1-s window; training time per model and seed on 12 bearings."),
}
CELL_FIXES = {
    "Table_config": [(r"$B$ = 3, $\gamma$ = 1, $\beta$ = 0.05", r"B = 3, $\gamma$ = 1, gate penalty 0.05"),
                     (r"$\lambda_c$ = 1 (masked Huber)", "concept loss weight 1 (masked Huber)")],
    "Table8_external": [(r"B1-V3, 4 s, $\sigma_r$ = 1\% (sens.)", r"B1-V3, 4 s, speed unc.\ 1\% (sens.)"),
                        (r"B1-V3, 4 s $^\ast$", r"B1-V3, 4 s *")],
    "Table6_pev": [(r"$\Delta p$", "Delta p"), (r"\textbf{Targeted $-$ control}", "Targeted minus control"),
                   (r"\textbf{Bearings T $<$ C}", r"Bearings T $<$ C")],
}


def clean_table(name: str) -> str:
    s = (TAB / f"{name}.tex").read_text(encoding="utf-8")
    for a, b in CELL_FIXES.get(name, []):
        assert a in s, (name, a)
        s = s.replace(a, b)
    title, note = CAPTIONS[name]
    s = re.sub(r"\\caption\{.*?\}\}\n", lambda m: f"\\caption{{\\textbf{{{title}}}}}\n", s, count=1, flags=re.S)
    s = re.sub(r"\\begin\{flushleft\}.*?\\end\{flushleft\}", lambda m: f"\\begin{{flushleft}} {note} \\end{{flushleft}}", s,
               count=1, flags=re.S)
    return typography(s).strip()


def typography(s: str) -> str:
    """Upright, plain table typography: bold only in the caption title, plain rules, no en or em dashes."""
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s.replace("\\caption{\\textbf{", "\\caption{\\TB{"))
    s = s.replace("\\caption{\\TB{", "\\caption{\\textbf{")
    s = re.sub(r"(\d\.\d\d)/\\allowbreak\{\}", r"\1, ", s)
    s = s.replace("\\thickhline", "\\hline").replace("/\\allowbreak{}", "/")
    s = re.sub(r"(?<=[0-9A-Za-z])--(?=[0-9A-Za-z])", " to ", s)
    s = re.sub(r"(?<![-\w])---?(?![-\w])", "-", s)
    return s


def main():
    t = TPL.read_text(encoding="utf-8")
    t = re.sub(r"@@TABLE:(\w+)@@", lambda m: clean_table(m.group(1)), t)
    problems = [ln for ln in t.splitlines() if re.search(r"(?<!\\)%", ln)]
    problems += [ln for ln in t.splitlines() if "--" in ln or "\u2013" in ln or "\u2014" in ln]
    if problems:
        print("\n".join(problems[:20]))
        raise SystemExit("comment or dash found")
    t = t + ("" if t.endswith("\n") else "\n")
    if OUT.exists() and OUT.read_text(encoding="utf-8") != t and "--force" not in sys.argv:
        raise SystemExit(f"{OUT} differs from the template output (edited by hand?). Not overwritten; use --force to replace it.")
    OUT.write_text(t, encoding="utf-8")
    print("written", OUT, len(t.split()), "tokens")


if __name__ == "__main__":
    sys.exit(main())
