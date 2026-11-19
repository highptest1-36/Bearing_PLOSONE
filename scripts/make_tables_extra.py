"""Additional manuscript tables (Table 1, 2, 5, 8 and supplementary tables) from verified result files; called by make_tables.py."""
from __future__ import annotations

import json

import pandas as pd

from scripts.make_tables import LABEL, ORDER, R3, R4, ROOT, f2, f3, write


def dataset_tables():
    meas = pd.read_csv(ROOT / "results/phase3b/env4s/meas.csv")
    fin = json.loads((ROOT / "results/phase3b/folds/FINAL.json").read_text())
    bd = sorted(fin["train_bearings"])
    pm = meas[meas.bearing.isin(bd)]
    win = pd.read_csv(ROOT / "results/phase2/windows/pu/meta.csv")
    nw = int(win.bearing.isin(bd).sum())
    seg = pd.read_csv(ROOT / "results/phase4/ottawa_segments4s.csv").groupby("stem").first()
    rows = [["Paderborn PU-BD5", "Development (nested CV, FINAL training)", "15 (5 H, 5 IR, 5 OR)", "6203",
             "64 kHz; 4 s", "4 settings: 900/1500 rpm, 0.1/0.7 Nm, 400/1000 N", f"{len(pm)} records (4 s); {nw} windows (1 s)"],
            ["Ottawa UORED-VAFCLS", "External, evaluated once", "20 (H state of all 20; IR 1--5; OR 6--10)", "6203 (NSK, FAFNIR)",
             "42 kHz; 10 s", f"one speed reading per file, {seg.rpm_sample0.min():.0f}--{seg.rpm_sample0.max():.0f} rpm; 400 N (ball faults 0 N)",
             "30 primary, 10 developing, 20 ball/cage (OOD) recordings"],
            ["KAIST", "Audit only (not evaluated)", "--", "--", "--", "--", "labels confounded with acquisition session"]]
    write("Table1_datasets", ["Dataset", "Role", "Physical bearings", "Type", "Sampling; length", "Operating conditions", "Units analysed"], rows,
          "\\textbf{Datasets and their roles.}", "tab:datasets",
          "PU-BD5: healthy bearings K001--K005 and bearings with real damage from accelerated lifetime tests (IR: KI04, KI14, KI16, KI18, KI21; "
          "OR: KA04, KA15, KA16, KA22, KA30). The physical bearing is the unit of analysis "
          "in both evaluated datasets.",
          colspec="|p{1.5cm}|p{1.9cm}|p{1.9cm}|p{1.2cm}|p{1.2cm}|p{2.6cm}|p{2.3cm}|")
    import pave.experiments.phase3a as P3A
    rows = []
    for o in P3A.outer_folds():
        rows.append([o["fold"], ", ".join(sorted(o["test"])), "12 bearings of the other 4 rows", "4 inner folds (one row held out)",
                     "temperature, R4, inclusion rule"])
    rows.append(["FINAL", "--", "all 15 PU-BD5 bearings", "--", "PU-pooled outer-test predictions (temperature, R4, thresholds)"])
    rows.append(["External", "20 Ottawa bearings (30 primary recordings)", "none (frozen FINAL models)", "--", "nothing from Ottawa"])
    write("Table2_splits", ["Split", "Test bearings", "Training", "Inner loop", "Fitted on inner OOF / PU only"], rows,
          "\\textbf{Bearing-disjoint evaluation design.}", "tab:splits",
          "Governing protocols (SHA-256 prefixes): V2 60d0c66a, V2.1 15343677, V2.2 078d726b, V2.3 6b4e4fdd, V2.4 dd758b7f, EV-V1 bb297d52. "
          "Within an outer fold, no window, record or measurement of a test bearing is used for training, calibration or threshold selection. "
          "The architecture-inclusion decision used inner OOF predictions pooled over all outer folds, in which each bearing appears in the "
          "inner validation sets of the four folds where it is a training bearing (see Methods).",
          colspec="|l|p{2.8cm}|p{2.6cm}|p{2.3cm}|p{3.2cm}|")


def concept_table():
    pr = pd.read_csv(R3 / "shortcut_probes_summary.csv")
    reps = [("V3_concepts_4s", "V3 concepts (4 s, measured)"), ("A6_predicted_concepts", "A6 predicted concepts"),
            ("B3_embedding", "B3 CNN embedding"), ("A4_latent", "A4 latent embedding")]
    targets = [("manufacturer_IBU", "Manufacturer"), ("year_2015", "Year"), ("origin_artificial(damaged only)", "Damage origin"),
               ("speed_900rpm_window_auroc", "Speed (window)"), ("setting_4class_window_acc", "Setting (window, acc.)")]
    rows = []
    for rk, rn in reps:
        r = [rn]
        for tk, _ in targets:
            x = pr[(pr.representation == rk) & (pr.target == tk)]
            r.append("--" if x.empty else f"{x['mean'].iloc[0]:.2f} [{x['min'].iloc[0]:.2f}, {x['max'].iloc[0]:.2f}]")
        rows.append(r)
    write("Table5_concepts_probes", ["Representation"] + [t for _, t in targets], rows,
          "\\textbf{Nuisance information in physics concepts and learned representations (PU, bearings unseen by each fold's models).}",
          "tab:probes",
          "Mean [min, max] over the five outer folds. Bearing-level targets: leave-one-bearing-out AUROC (chance 0.5). Setting: 4-class window "
          "accuracy (chance 0.25). Presence of information does not imply that a classifier uses it.")


def external_tables():
    ex = pd.read_json(R4 / "external_primary.json").set_index("model")
    names = dict(LABEL, **{"B1-V3-4s_sigma1pct": "B1-V3, 4 s, $\\sigma_r$ = 1\\% (sens.)", "B3_recnorm": "B3, rec.-norm. (sec.)",
                           "A4_recnorm": "A4, rec.-norm. (sec.)", "A6_recnorm": "A6, rec.-norm. (sec.)"})
    rows = []
    for m in ORDER + ["B1-V3-4s_sigma1pct", "B3_recnorm", "A4_recnorm", "A6_recnorm"]:
        r = ex.loc[m]
        rows.append([names[m] + (" $^\\ast$" if m == "B1-V3-4s" else ""), f3(r.macro_f1), f"[{f2(r.ci95[0])}, {f2(r.ci95[1])}]", f3(r.bal_acc),
                     "/".join(f2(x) for x in r.recall_H_IR_OR), f2(r.healthy_false_alarm), f2(r.fault_sensitivity), f2(r.localisation_on_detected),
                     f3(r.window_nll) if pd.notna(r.window_nll) else "--", f3(r.window_ece) if pd.notna(r.window_ece) else "--"])
    write("Table8_external", ["Model", "Macro-F1", "95\\% CI", "Bal. acc.", "Recall H, IR, OR", "Healthy FA", "Fault sens.", "Localis.", "NLL", "ECE"], rows,
          "\\textbf{Single external evaluation on Ottawa UORED-VAFCLS (30 primary recordings, 20 bearings).}", "tab:external",
          "$^\\ast$ Primary model, fixed by the development gate before Ottawa was analysed. CI: bootstrap over the 20 physical bearings. "
          "Localis.: IR/OR accuracy among detected faults. sens.: pre-specified speed-uncertainty sensitivity analysis; rec.-norm.: pre-specified "
          "per-recording input standardisation. Window NLL/ECE after PU-fitted temperature scaling.")
    sec = pd.read_csv(R4 / "external_secondary.csv")
    rows = []
    for m in ORDER:
        d = sec[(sec.model == m) & (sec.role == "developing")].iloc[0]
        o = sec[(sec.model == m) & (sec.role == "ood_faulty")].iloc[0]
        od = sec[(sec.model == m) & (sec.role == "ood_developing")].iloc[0]
        rows.append([LABEL[m], f2(d.detected), f2(d.correct_type), f"{o.pred_H}/{o.pred_IR}/{o.pred_OR}", f"{od.pred_H}/{od.pred_IR}/{od.pred_OR}"])
    write("S_Table_external_secondary", ["Model", "Developing: detected", "Developing: correct type", "Ball/cage faulty: H/IR/OR",
                                         "Ball/cage developing: H/IR/OR"],
          rows, "\\textbf{Ottawa secondary analyses: developing faults (10 recordings) and out-of-scope ball and cage faults (10 + 10).}",
          "tab:extsec", "Models that predict a fault for every Ottawa recording, including all healthy ones, trivially reach 1.0 developing-fault detection.",
          colspec="|l|p{2.2cm}|p{2.2cm}|p{2.6cm}|p{2.6cm}|")
    sd = pd.read_csv(R4 / "external_selective.csv")
    rows = [[r["base"], {"S1": "R1", "S2": "R2", "S3": "R3", "S4": "R4"}[r["score"]], f3(r["AURC"]), int(r["n_errors"]), f3(r["error_detection_auroc"]),
             f"{r['pu_threshold_cov0.8_realised_coverage']:.2f} / {r['pu_threshold_cov0.8_risk']:.2f}"] for _, r in sd.iterrows()]
    write("S_Table_external_selective", ["Base model", "Score", "AURC", "Errors (of 30)", "Error-det. AUROC", "PU threshold (cov. 0.8): coverage / risk"],
          rows, "\\textbf{Ottawa selective diagnosis with PU-frozen scores and thresholds.}", "tab:extsel", "Recording level; 30 primary recordings.")
    pv = pd.read_csv(ROOT / "results/phase4/pev/pev_ottawa_targeted_minus_control.csv")
    nm = {"evidence_dB": "own-family evidence (dB)", "B1-V3-4s": "$\\Delta p$ B1-V3-4s", "A4": "$\\Delta p$ A4 PAVE-Net", "B3": "$\\Delta p$ B3 CNN",
          "A4_phys_logit": "A4 physics logit", "A4_latent_contrib": "A4 latent contribution"}
    rows = [[nm[r.quantity], f"{r.median_T_minus_Ce:+.3f}", f"[{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}]", f"{r.n_negative}/{r.n}"] for r in pv.itertuples()]
    write("S_Table_external_pev", ["Quantity", "Targeted $-$ energy-matched control (median)", "95\\% CI", "Recordings T $<$ C"], rows,
          "\\textbf{Physics-evidence verification on the 10 faulty Ottawa recordings (pre-specified secondary analysis).}", "tab:extpev",
          "Own fault family, strength 1.0; recording means over two 4-s records; bootstrap CI over recordings (one recording per bearing).")


def related_work_table():
    Y, N = "yes", "no"
    rows = [
        ["Smith \\& Randall 2015 \\cite{smith2015rolling}", "envelope and spectral benchmark analysis", N, "-- (CWRU, per-record)", N, N, N, N],
        ["Hendriks et al. 2022 \\cite{hendriks2022towards}", "--", Y, "random vs. bearing-aware splits (CWRU)", N, N, N, N],
        ["Wheat et al. 2024 \\cite{wheat2024impact}", "frequency and envelope features", Y, "part-to-part (CMHT, Paderborn KAt)", N, N, N, N],
        ["Vieira et al. 2026 \\cite{vieira2026towards}", "--", Y, "bearing-wise (CWRU, PU, UORED, HUST)", "within-dataset", N, N, N],
        ["Siddique et al. 2026 \\cite{siddique2026physics}", "kinematic orders, null orders, conformal alarm", N, "3 frozen PU folds",
         "UORED, frozen, nominal speed", "fixed order tolerance", "decision-level (no faithfulness test)", N],
        ["Koh et al. 2020 \\cite{koh2020concept}", "concept bottleneck (generic ML)", Y, "--", "--", "--", "concept interventions", N],
        ["This study", "8 identifiability-masked envelope concepts", Y, "nested, 5 outer $\\times$ 4 inner (PU-BD5)", "Ottawa, frozen, single evaluation",
         "$\\sigma_r$ in regions and guards; sensitivity analysis", "targeted vs. width- and energy-matched suppression", "PE-SD (R1--R4)"],
    ]
    write("S1_Table_related_work", ["Study", "Physics integration", "Learned model", "Bearing-disjoint evaluation", "External dataset",
                                    "Speed uncertainty", "Explanation faithfulness", "Selective diagnosis"], rows,
          "\\textbf{Related-work matrix.}", "tab:related",
          "Entries summarise only what the cited articles report (checked against full texts or publisher records). '--': not applicable or not reported.",
          colspec="|p{1.9cm}|p{1.9cm}|p{0.9cm}|p{1.9cm}|p{1.7cm}|p{1.5cm}|p{1.9cm}|p{1.1cm}|")
