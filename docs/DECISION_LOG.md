# Decision log

| Date | Decision | Reason / evidence | Test data seen? |
|---|---|---|---|
| 2026-11-18 | Channel mapping Point1..4 → x_A, y_A, x_B, y_B; TDMS ai0/ai1 → T_A, T_B (ASSUMED) | paper column order; no metadata | n/a (audit) |
| 2026-11-18 | Use resolved TDMS pairing (0 Nm IR 0.3 ↔ 1.0 swap) | internal names + clock-corrected timing + current-derived speed | n/a |
| 2026-11-18 | Mark OR 0.3 mm temperature NON_CONCURRENT at all loads | TDMS ≈ 32 min after vibration; speed mismatch 7–10× concurrent max | n/a |
| 2026-11-18 | Store raw vibration values; label unit "stored (m/s² per metadata; paper: g)" | Test.Lab MKS metadata vs paper | n/a |
| 2026-11-18 | Use per-recording measured shaft speed for all fault frequencies | speed 49.88–50.19 Hz; VFD re-tuned for fault runs only | n/a |
| 2026-11-18 | Control frequency set widened to 24 orders in 2.3–19 × fr | original 2.3–11.7 range admitted only 12 controls | n/a |
| 2026-11-18 | **POST-HOC:** add ±3.5 % fault-family search + ≥ 6 dB family contrast, threshold = q99 of healthy windows | pre-specified ±1 % kinematic rule matched only 8/21 recordings; lines found −1.2…−1.6 % (IR), +0.9…+2.5 % (OR) | all recordings inspected (audit, no model) |
| 2026-11-18 | Splits frozen: A LOLO (val = last 20 %, purge 2 s), B 6 permutations, C within-recording | blueprint §3; hashes in `03_MANIFESTS/splits/split_index.csv` | no model results exist |
| 2026-11-18 | Recommend MODIFY (drop thermal fusion claim) — **awaiting author decision** | `DATA_FEASIBILITY_ASSESSMENT.md` | — |

Note: because this audit inspected all recordings (including what will be test folds) to validate data and physics,
any concept definition, band, tolerance or threshold carried into modelling must be re-fit on source folds only and
frozen before test evaluation. The audit-time thresholds are descriptive and must not be reused as tuned parameters.

## Phase 2 (2026-11-18/19)

| Date | Decision | Reason / evidence | Test data seen? |
|---|---|---|---|
| 2026-11-18 | Author approved MODIFY: temperature removed from primary claims; framework renamed *Physics-Audited Vibration Evidence Network*; 3-class primary; 7-class KAIST exploratory; Paderborn added with bearing-disjoint evaluation; AdamW | author decision after Phase 1 | — |
| 2026-11-18 | Blueprint V1 archived (`99_ARCHIVE/blueprint_v1_multimodal/`, SHA-256 unchanged); V2 written; **protocol V2 frozen** (hash in `docs/PROTOCOL_FREEZE.txt`) | author instruction A | PU: only metadata of one file read |
| 2026-11-18 | PU-BD5 = Lessmeier 2016 Table 10 rows, fold k test row k, val row k+1; PU-A2R artificial→real; KB23/24/27 challenge only | protocol V2 §4.2–4.4 | before any PU signal analysis |
| 2026-11-18 | Per-bearing geometry from each archive's fact sheet (`<code>.pdf`) instead of one constant set | K001 fact sheet: IBU pitch 29.05 mm vs FAG 28.55 mm in paper Table 1 | metadata only |
| 2026-11-18 | `N15_M01_F10_KA08_2.mat` excluded (unreadable by scipy: "Expecting matrix here"; header/whosmat readable) | read error recorded in `pu_files.csv` | — |
| 2026-11-18 | Smoke-test outputs moved out of the main results tree (`runs/_smoke/`); first smoke records deleted | smoke runs (2 epochs) would otherwise be reused by the resume logic | KAIST fold test0 smoke seen |
| 2026-11-19 | **POST-HOC (not in frozen protocol):** analytic metadata-confound controls (manufacturer-only, year-only rules fit on train bearings) | PU audit found Healthy ≡ IBU and Healthy measured in 2015 vs most damaged in 2014; decided after PU audit and after the PU fold-1 smoke results were printed | PU fold-1 smoke results seen |

## Phase 3A (2026-11-19)

| Date | Decision / event | Reason / evidence | Test data seen? |
|---|---|---|---|
| 2026-11-19 | Author decisions: P1 = primary benchmark (B1); investigate nested bearing-group validation; manufacturer/year confounding = central finding | author instruction | Phase 2 results seen |
| 2026-11-19 | Protocol amendment **V2.1** written and frozen (`docs/PROTOCOL_FREEZE_V2.1.txt`); V2 file unchanged (hash re-checked) | nested validation, confound endpoints, PU no longer "unseen" | before any Phase 3A model/intervention/probe |
| 2026-11-19 | **Erratum (Phase 2):** in `pave/training/engine.fit` the model was constructed before `set_seed`, so the seed labels 42/43/44 controlled data order and dropout but NOT weight initialisation. Phase 2 runs remain reproducible when re-run in the same order (PyTorch default generator), but seed-to-seed spread understates initialisation variance. Phase 3A code (`pave/training/curves.py`) constructs the model after seeding; a test checks that different seeds change initialisation. The Phase 2 report itself is not edited. | found by `tests/test_phase3a.py::test_fit_fixed_reproduces_curve_epoch` | — |
| 2026-11-19 | V2.1 §6 fallback rule **triggered**: inner-fold spread of macro-F1 > 0.2 for every B1 configuration in 4/5 outer folds (min spread 0.25–0.47); B3 selected configuration spread 0.56–0.64 (other B3 configurations' per-inner values not stored; B2 per-inner values not stored — protocol deviation). Per protocol the defaults (B1 C = 1; B3 lr 1e-3, 25 epochs) are reported as primary for B1/B3, nested-selected results alongside. | `results/phase3a/tables/nested_selection.csv` | outer-test results computed in the same run (selection itself used inner data only) |
| 2026-11-19 | Intervention code fixes before any reported intervention result: (1) modification band-limited (out-of-band leakage 5.9 % → < 1 %); (2) concept extraction f/E length mismatch; (3) energy-matched control loop could not terminate with dense shaft guards (fragment-width stop criterion) → nominal-width tracking + max 30 iterations. Earlier aborted runs produced no saved results. | tests in `tests/test_interventions.py` | one debug run on KA04 (outputs overwritten by the full run) |

## Phase 3B (2026-11-19)

| Date (local) | Decision / event | Reason / evidence | Results inspected? |
|---|---|---|---|
| 2026-11-19 | Protocol **V2.2** frozen (`078d726b…`): concept V3, fixed hyper-parameters, B0–B4, A2–A6, PEV, PE-SD, decision rules, external-validation plan | author instructions (Phase 3B prompt) | Phase 1–3A outputs; Ottawa paper + file listing only |
| 2026-11-19 | Protocol **V2.3** (`6b4e4fdd…`): physics context of fusion models = 4-s measurement | analytic identifiability table: no 6203 family identifiable at 1 s (δ −0.24…−1.19 bins) | V3 synthetic tests + analytic table only |
| 2026-11-19 | 55 PU measurements shorter than 4 s (1–6 060 samples short) zero-padded to the 0.25-Hz FFT grid | applies to all classes; documented in `scripts/build_env4s.py` | none |
| 2026-11-19 | Protocol **V2.4** (`dd758b7f…`): identifiability judged on the measured line (δ ≥ 1, local maximum) instead of the whole calibrated region | V2.3 rule gave 0 % valid families at every setting (label-free smoke diagnostic) | label-free validity fractions + calibrated regions of a 1-epoch smoke run; no metrics |
| 2026-11-19 | Smoke outputs kept separately (`results/phase3b/_smoke_v23_region_rule/`, `results/phase3b/_smoke/`); never used for results | transparency | — |
| 2026-11-19 | Literature verified online (28 entries, `02_LITERATURE/verified_references.bib` + report); corrections: KAIST Data in Brief vol. 48; Machines 2026 paper by Siddique et al. reports 0 % IR / 20 % OR detection on Ottawa with nominal speed | needed for related work | — |
| 2026-11-19 | Phase 3B gate **GO**; final recommended model **B1-V3-4s** (latent branch not retained: inner-OOF A4 0.541 vs A2 0.687, NLL 1.389 vs 0.774) | pre-specified inclusion rule on inner OOF only (`results/phase3b/tables/inclusion_decision.json`) | outer-test results computed after the decision file was written |

## Phase 4 — External validation (2026-11-19)

| Date (local) | Decision / event | Reason / evidence | Results inspected? |
|---|---|---|---|
| 2026-11-19 | Label-free Ottawa audit (`scripts/external_ottawa.py --stage audit`): speed channel zero except sample 0 (1700–2190 rpm); load at sample 0 (400 N / 0 N) | needed to define the speed policy | file integrity, speed/load encoding, 1× cross-check; no labels, features or predictions |
| 2026-11-19 | EV-V1 §3 speed policy revised **before freeze**: median of speed column → reading at sample 0; σ_r = 0.5 rpm/60; fallback 1750 rpm if outside 600–3000 | the median would be 0 | label-free audit only |
| 2026-11-19 (UTC 22:29:55 on 11-18) | **EV-V1 frozen** (`docs/PROTOCOL_FREEZE_EV_V1.txt`, 45 hashed files incl. 21 FINAL checkpoints); primary model B1-V3-4s | single-evaluation rule | synthetic dry-run of the evaluation code (random signals) only |
| 2026-11-19 05:30:24 | **Single Ottawa evaluation executed**; `results/phase4/EVALUATED.lock` written; all hashes matched. Primary: B1-V3-4s macro-F1 0.518 [0.31, 0.68]; healthy FA 0.55 | EV-V1 | — |
| 2026-11-19 | Pre-specified secondary PEV on the 10 faulty Ottawa recordings run **after** the lock by a separate script (`scripts/pev_ottawa.py`); no training/tuning; consistency check that the original variant reproduces frozen features | listed in EV-V1 §8 but not implemented in the evaluation script | primary results known (does not affect them) |
| 2026-11-19 | Post-hoc exploratory analysis (`scripts/posthoc_ottawa_exploratory.py`): false alarms concentrated in healthy recordings of later-IR bearings (5/5) | to explain the external failure; labelled post hoc everywhere | primary results known |
| 2026-11-19 | Manuscript, figures (Fig 1–8) and tables generated programmatically from result files; no number typed by hand into tables | reproducibility | — |

## Phase 5 — Pre-submission methodological audit and manuscript v2 (2026-11-19)

| Date | Decision | Reason | Results inspected |
|---|---|---|---|
| 2026-11-19 | Manuscript v1 archived unchanged (`manuscript_archive/v1_2026-11-19_pre_audit/`, with SHA256SUMS) before any edit | preserve the historical record | — |
| 2026-11-19 | Audit finding: the inclusion rule used inner-OOF predictions pooled over all outer folds (each bearing 4 times), so outer-test bearings informed the global selection through other folds; manuscript and Table 2 note overstated independence | methodological correctness | `inclusion_decision.json`, stored fold outputs |
| 2026-11-19 | **POST HOC** fold-local re-application of the frozen rule (`scripts/audit_foldlocal_selection.py`): B1-V3-4s chosen in 5/5 folds (O4 by the V2.3 tie-break); outer macro-F1 unchanged (0.780) | sensitivity analysis for the finding above; stored predictions only, no retraining | stored Phase 3B outputs |
| 2026-11-19 | **POST HOC** prevalence re-weighting of the reported confusion matrices (`scripts/audit_prevalence.py`): B1-V3-4s Ottawa macro-F1 0.518 → 0.596 at 5/5/5; balanced accuracy of the CNN-based models (0.53–0.60) close to that of B1-V3-4s (0.62) | interpret PU vs Ottawa macro-F1; arithmetic on already-reported tables, no Ottawa signal or label file read, no model run | `external_primary.csv`, `per_bearing_predictions.csv` |
| 2026-11-19 | Wording: "pre-registered" → internally hash-frozen protocols (no public registry); reuse of the 15 PU bearings across Phases 3A/3B disclosed; input mismatch between the CNN and the physics models disclosed; causal/faithfulness language restricted; PU-BD5 damage description corrected (accelerated lifetime, not "natural") | audit Tasks 2–6 | — |
| 2026-11-19 | Formatting for PLOS: authors and acknowledgment supplied by the researchers inserted; software, code/data availability and statistics statements added; scores renamed R1–R4; figures re-rendered (Arial ≥ 8 pt, RGB TIFF); SI split into S1–S5_Table.pdf + S3 File; dataset DOIs added (verified by DOI resolution) | PLOS requirements (`docs/PLOS_COMPLIANCE_CHECKLIST.md`) | — |
| 2026-11-19 | Fig 8 re-rendered: re-reads the two raw Ottawa recordings already shown in v1 (I_1_2, H_3_0) for plotting only; probabilities taken from the frozen prediction file; no model run, no metric recomputed | figure-format fix | — |
| 2026-11-19 | No frozen protocol, prediction, checkpoint or hash modified; Ottawa not re-evaluated; nothing submitted or uploaded | constraints | — |

## Style revision v2 → v3 (2026-11-19, requested by the authors)

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 2026-11-19 | Manuscript rewritten as one comment-free file `manuscript/pavenet_main.tex`. Impersonal voice with no "we/our"; short sentences; continuous paragraphs with no lists; milder claims; an organisation paragraph at the end of the Introduction; abstract of 163 words; shorter figure and table captions; only two equations kept (identifiability margin, PAVE-Net fusion), standard formulas cited; no em or en dashes in the authored text | author style requirements | `manuscript_archive/revision_2026-11-19_style/` (template, diff, HTML) |
| 2026-11-19 | Funding: no specific funding (the university support is non-financial), Acknowledgments unchanged. Affiliation name confirmed. CRediT roles proposed | author instructions | `manuscript/submission_statements.md`, `docs/AUTHOR_TODO_V3.md` |
| 2026-11-19 | Fig 1 and Fig 2 redrawn: pseudo-3D blocks, bearing icons, orthogonal arrows. Fig 3, 6, 7 and 8 re-laid out: no text over data; Fig 8 has separate zoom panels instead of insets. Fig 8 again re-reads the same two raw Ottawa recordings, for plotting only | author figure requirements | `scripts/fig_schematics.py`, `scripts/make_figures.py` |
| 2026-11-19 | No number changed: tables are converted verbatim from the generated files by `scripts/assemble_clean_tex.py`; no model run; Ottawa not re-evaluated; frozen hashes re-verified (45/45); 110 tests pass | constraints | — |
