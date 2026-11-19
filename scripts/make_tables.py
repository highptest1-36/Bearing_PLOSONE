"""Generate LaTeX tables for the manuscript from verified result files (no manual values).

Writes FINAL_PLOS_ONE_PACKAGE/tables/*.tex and *.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R3 = ROOT / "results/phase3b/tables"
R4 = ROOT / "results/phase4/tables"
OUT = ROOT / "FINAL_PLOS_ONE_PACKAGE/tables"
ORDER = ["B0", "B1-V2", "B1-V3-1s", "B1-V3-4s", "B2", "B3", "B4", "A2", "A3", "A4", "A6"]
LABEL = {"B0": "B0 physics rule", "B1-V2": "B1-V2 (P1, V2 concepts)", "B1-V3-1s": "B1-V3, 1 s", "B1-V3-4s": "B1-V3, 4 s",
         "B2": "B2-V3 MLP, 1 s", "B3": "B3 CNN-1D", "B4": "A1/B4 physics head", "A2": "A2 + reliability features",
         "A3": "A3 + latent, no gate", "A4": "A4 PAVE-Net (gated)", "A6": "A6 concept bottleneck"}
INPUT = {"B0": "4 s phys.", "B1-V2": "1 s phys.", "B1-V3-1s": "1 s phys.", "B1-V3-4s": "4 s phys.", "B2": "1 s phys.",
         "B3": "1 s raw", "B4": "4 s phys.", "A2": "4 s phys.", "A3": "4 s phys. + 1 s raw", "A4": "4 s phys. + 1 s raw",
         "A6": "1 s raw (4 s targets)"}


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


TEXT_CM = 16.5  # PLOS body text width 6.5 in


def _cell(c):
    """Allow line breaks after '/' in plain cells (e.g. 1.00/0.60/0.40) and after ', ' inside intervals."""
    s = str(c)
    return s if "\\" in s or "$" in s else s.replace("/", "/\\allowbreak{}")


def write(name, header, rows, caption, label, note="", colspec=None, first_cm=None):
    n = len(header)
    font = "\\scriptsize" if n >= 8 else "\\footnotesize"
    if colspec is None:
        first_cm = first_cm or (2.6 if n >= 8 else 3.2)
        pad = 0.42 * n  # 2 x \tabcolsep per column
        w = max((TEXT_CM - pad - first_cm) / (n - 1), 0.9)
        colspec = f"|p{{{first_cm:.2f}cm}}" + "".join(f"|>{{\\centering\\arraybackslash}}p{{{w:.2f}cm}}" for _ in range(n - 1)) + "|"
    lines = ["\\begin{table}[!ht]", "\\centering", f"\\caption{{{caption}}}", f"{{{font}", f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
             " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\ \\thickhline"]
    for r in rows:
        lines.append(" & ".join(_cell(c) for c in r) + " \\\\ \\hline")
    lines += ["\\end{tabular}}", f"\\begin{{flushleft}} {note} \\end{{flushleft}}" if note else "", f"\\label{{{label}}}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows, columns=header).to_csv(OUT / f"{name}.csv", index=False)


def pu_tables():
    main = {r["model"]: r for r in json.loads((R3 / "main_results.json").read_text())}
    rows = []
    for m in ORDER:
        r = main[m]
        rec = r["recall_H_IR_OR"]
        rows.append([LABEL[m], INPUT[m], f3(r["bearing_macro_f1"]), f"[{f2(r['ci95'][0])}, {f2(r['ci95'][1])}]", f3(r["bearing_bal_acc"]),
                     "/".join(f2(x) for x in rec), f2(r["fault_sensitivity"]), f2(r["healthy_false_alarm"]),
                     f3(r["window_macro_f1"]), f3(r["nll_ts"]), f3(r["ece_ts"])])
    write("Table3_baselines", ["Model", "Input", "Bearing macro-F1", "95\\% CI", "Bal. acc.", "Recall H, IR, OR", "Fault sens.",
                               "Healthy FA", "Window macro-F1", "NLL", "ECE"], rows,
          "\\textbf{Bearing-level results on the 15 PU-BD5 bearings (nested outer test; three-seed ensembles).}", "tab:baselines",
          "Bearing = experimental unit (5 H, 5 IR, 5 OR). CI: class-stratified bootstrap over bearings. NLL and ECE are window-level "
          "after temperature scaling fitted on inner out-of-fold predictions. FA: false-alarm rate. Paired differences are given in S2 Table; "
          "none is statistically supported. Physics models use 4-s or 1-s concepts from the 64-kHz signal (1--20 kHz); CNN-based models use "
          "1-s windows at 32 kHz.")
    rows = []
    for m in ORDER:
        r = main[m]
        for ref in ("B1-V2", "B1-V3-4s"):
            if f"diff_vs_{ref}" in r:
                mc = r[f"vs_{ref}_mcnemar"]
                rows.append([LABEL[m], ref, f"{r[f'diff_vs_{ref}']:+.3f}", f"[{r[f'diff_vs_{ref}_ci95'][0]:+.2f}, {r[f'diff_vs_{ref}_ci95'][1]:+.2f}]",
                             f"{mc['a_only_correct']}/{mc['b_only_correct']}", f"{mc['exact_p']:.3f}"])
    write("S_Table_paired_differences", ["Model", "Reference", "$\\Delta$ macro-F1", "95\\% CI", "Discordant (model/ref.)", "McNemar p"], rows,
          "\\textbf{Paired bearing-level differences on PU-BD5.}", "tab:paired", "Exact McNemar test on bearing-level correctness (15 bearings).")
    # ablation
    inc = json.loads((R3 / "inclusion_decision.json").read_text())
    abl = [("A0", "B0", "physics rule"), ("A1", "B4", "physics concepts V3 (linear head)"), ("A2", "A2", "+ evidence reliability"),
           ("A3", "A3", "+ bounded latent (no gate)"), ("A4", "A4", "full gated fusion"), ("A6", "A6", "concept bottleneck (predicted concepts)")]
    rows = []
    for aid, m, desc in abl:
        r = main[m]
        io = inc["inner_oof"].get(m, {})
        rows.append([aid, desc, f3(r["bearing_macro_f1"]), f"{r['weak_bearings_correct']}/6", f"{r['strong_bearings_correct']}/4",
                     f3(r["nll_ts"]), f3(io.get("inner_oof_bearing_macro_f1")), f3(io.get("inner_oof_window_nll"))])
    write("Table4_ablation", ["ID", "Component", "Outer macro-F1", "Weak-evidence correct", "Strong-evidence correct", "Outer NLL",
                              "Inner-OOF macro-F1", "Inner-OOF NLL"], rows,
          "\\textbf{Ablation study (PU-BD5).}", "tab:ablation",
          f"Weak-evidence bearings: KA15, KA22, KA30, KI14, KI16, KI21; strong: KA04, KA16, KI04, KI18. Inner-OOF values (60 bearing "
          f"predictions pooled over the five outer folds; each bearing appears four times) were the only basis of the pre-specified inclusion "
          f"rule; decision: keep latent branch = {inc['keep_latent_branch']}, "
          f"final recommended model = {inc['final_recommended_model']}. A post hoc re-application of the rule within each outer fold selected the "
          f"same model in all five folds (see Results). A5 (A4 + PE-SD) is reported in the selective-diagnosis table.")
    # selective
    SMAP = {"S1": "R1", "S2": "R2", "S3": "R3", "S4": "R4"}
    sd = pd.read_csv(R3 / "selective_diagnosis.csv")
    rows = [[r.base, SMAP[r.score], f3(r.AURC), f"{r.risk_at_12_of_15:.3f} ({r.accepted_errors_12})", f"{r.risk_at_9_of_15:.3f} ({r.accepted_errors_9})",
             f3(r.error_detection_auroc), r.n_errors_total] for r in sd.itertuples()]
    write("Table7_selective", ["Base model", "Score", "AURC", "Risk@12/15 (errors)", "Risk@9/15 (errors)", "Error-det. AUROC", "Errors"],
          rows, "\\textbf{Selective diagnosis on PU-BD5 at matched bearing coverage.}", "tab:selective",
          "R1 max probability, R2 negative entropy, R3 evidence clarity, R4 physics-evidence-aware score (fitted on inner OOF).")
    # PEV
    pev = pd.read_csv(R3 / "pev_targeted_minus_control.csv")
    pev = pev[(pev.strength == 1.0)]
    rows = []
    for metric, name in (("d_target_harm", "target-family evidence (dB)"), ("dp_B1-V3-4s", "$\\Delta p$ B1-V3-4s"), ("dp_A2", "$\\Delta p$ A2"),
                         ("dp_A4", "$\\Delta p$ A4 PAVE-Net"), ("dp_A6", "$\\Delta p$ A6 concept bottleneck"), ("dp_B3", "$\\Delta p$ B3 CNN"),
                         ("d_A4_phys", "A4 physics logit"), ("d_A4_latent", "A4 latent contribution")):
        for ctrl in ("Ce", "Cw"):
            x = pev[(pev.metric == metric) & (pev.control == ctrl)]
            if len(x):
                x = x.iloc[0]
                rows.append([name, {"Ce": "energy-matched", "Cw": "width-matched"}[ctrl], f"{x.mean_T_minus_C:+.3f}",
                             f"[{x.ci95_lo:+.3f}, {x.ci95_hi:+.3f}]", f"{x.n_bearings_T_lt_C}/{x.n_bearings}"])
    write("Table6_pev", ["Quantity", "Control", "Targeted $-$ control", "95\\% CI", "Bearings T $<$ C"], rows,
          "\\textbf{Physics-evidence verification on the 10 faulty PU-BD5 bearings (own fault family, strength 1.0).}", "tab:pev",
          "Bearing means over 20 records per bearing; bootstrap CI over bearings. $\\Delta p$: change in probability of the true class.")
    eff = pd.read_csv(R3 / "efficiency.csv")
    rows = [[r.model, r.device, int(r.params), f"{r.latency_ms_median:.2f}", f"{r.latency_ms_p95:.2f}",
             "--" if pd.isna(r.train_seconds_median) else f"{r.train_seconds_median:.1f}", "--" if pd.isna(r.peak_gpu_MB_max) else f"{r.peak_gpu_MB_max:.0f}"]
            for r in eff.itertuples()]
    write("Table9_efficiency", ["Model", "Device", "Parameters", "Latency median (ms)", "Latency p95 (ms)", "Training time (s)", "Peak GPU (MB)"], rows,
          "\\textbf{Computational cost.}", "tab:efficiency", "Batch-1 inference on one 1-s window; training time per model per seed on 12 bearings (RTX 3060 Laptop GPU).")


def config_table():
    rows = [["B1 (all)", "logistic regression", "C = 1, balanced, StandardScaler", "--", "--"],
            ["B2-V3", "MLP 2$\\times$64, dropout 0.1", "AdamW lr $10^{-3}$, wd $10^{-2}$", "256", "20"],
            ["B3", "CNN-1D (6 blocks, 0.23 M)", "AdamW lr $10^{-3}$, wd $10^{-2}$, cosine, AMP", "64", "25"],
            ["A1--A4", "linear physics head (+ gate, + bounded latent)", "AdamW lr $10^{-3}$, wd $10^{-2}$; $B$ = 3, $\\gamma$ = 1, $\\beta$ = 0.05", "64", "25"],
            ["A6", "CNN-1D $\\rightarrow$ 8 concepts $\\rightarrow$ linear", "AdamW lr $10^{-3}$, wd $10^{-2}$; $\\lambda_c$ = 1 (masked Huber)", "64", "25"]]
    write("Table_config", ["Model", "Architecture", "Optimisation", "Batch", "Epochs"], rows,
          "\\textbf{Fixed training configurations (protocol V2.2).}", "tab:config", "Seeds 42, 43, 44 (set before model construction); class-weighted cross-entropy.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    config_table()
    if (R3 / "main_results.json").exists():
        pu_tables()
    from scripts.make_tables_extra import concept_table, dataset_tables, external_tables, related_work_table
    dataset_tables()
    concept_table()
    external_tables()
    related_work_table()
    print(sorted(p.name for p in OUT.glob("*.tex")))
