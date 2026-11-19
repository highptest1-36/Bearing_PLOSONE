# PAVE-Net — Research progress log

## Stage 0 — Project inspection for Phases 3B–4 (2026-11-19)

**Status: BLOCKED (technical, not scientific).** The shell working directory was moved into
`01_RAW_DATA/Ottawa_data/<UORED-VAFCLS>/<UORED-VAFCLS>` (path contains an en-dash and parentheses) by an inspection
`cd`. From that point every shell command, including `echo`, failed at start-up with
exit code 1. No command after that point executed; no data or results were modified. Fix: restart the shell
from `D:\paper\PAVE_NET_RESEARCH`.

### Verified before the blocker
- Phase 3A state: `python -m pytest -q tests` → **95 passed** (run at the start of this stage).
- Frozen protocols present and unchanged: V2 (`60d0c66a…`), V2.1 (`15343677…`) (hashes last verified end of Phase 3A).
- Datasets located:
  - KAIST: `01_RAW_DATA/Vibration, Acoustic, Temperature, and Motor Current Dataset .../`
  - Paderborn: `01_RAW_DATA/Paderborn_KAt_BearingDataCenter/archives/*.rar` (32) + cache `results/phase2/cache_pu/`
  - **Ottawa UORED-VAFCLS**: `01_RAW_DATA/Ottawa_data/<UORED-VAFCLS>/<UORED-VAFCLS>/` with
    `1_CSV_Raw_Data_Files (.csv)` (60 files), `2_Excel_Raw_Data_Files (.xlsx)` (60), `3_MatLab_Raw_Data_Files (.mat)` (60,
    sub-folders `1_Healthy`, `2_Inner_Race_Faults`, `3_Outer_Race_Faults`, `4_Ball_Faults`, `5_Cage_Faults`), and two spectrogram
    zip archives; total 3.7 GB. The dataset **version is not recorded in the local files** (the authors state V5,
    doi:10.17632/y2px5tg92h.5; the paper cites .2 and .3) — to be recorded as author-stated, unverified.
- Literature in `02_LITERATURE/`: Lessmeier 2016 (Paderborn) and `Ottawa_Dataset_Paper.pdf`.
  **Missing:** the mandatory comparison paper *Machines* 2026, doi:10.3390/machines14101140 (not downloaded; must be
  verified online or supplied).
- Author materials in `00_AUTHOR_GUIDELINES/`: `submission_guideline.pdf`, `PLOS Manuscript Body Formatting Guidelines.pdf`,
  `PLOS Affiliations Formatting Guidelines.pdf`, `criteria_submit.md`, `figure_guideline.md`, and the official
  `PLOS_latex_template/` (`plos_latex_template.tex`, `plos2025.bst`, sample `.bib`, rendered PDF).
- Toolchain: MiKTeX `pdflatex`, `xelatex`, `latexmk` available; `pandoc` not installed.

### Ottawa dataset facts (from Sehri, Dumond & Bouchard, *Data in Brief* 49 (2023) 109327, doi:10.1016/j.dib.2023.109327, CC BY 4.0)
- 20 bearings; each run from healthy to failure; 3 states per bearing: healthy (`X-n-0`), developing (`-1`), faulty (`-2`);
  bearings 1–5 inner race, 6–10 outer race, 11–15 ball, 16–20 cage → **60 files, 20 physical bearings**.
- Healthy data from files 1–5 of a 50-file run, developing from files 25–30, faulty from files 45–50.
- Each file: 10 s at **42 000 Hz** (420 000 samples); columns: accelerometer (m/s²), microphone (V), speed (RPM, Hall
  sensor, two-tooth gear), load (N), temperature differential (°C).
- Bearings NSK 6203ZZ and FAFNIR 203KD: pitch diameter 28.50 mm, ball diameter 6.77 mm, 8 balls (contact angle not stated).
  Fig. 1 assigns NSK 6203ZZ to bearings 14, 16, 17, 18, 20 and FAFNIR 203KD to the rest; the text says the first 5 *tests*
  used 6203ZZ — test order ≠ bearing number (source inconsistency to record).
- Load 400 N for healthy, inner, outer and cage data; ball-fault data recorded with no load.
- Nominal motor speed 1750 RPM (accelerometer PCB 623C01 on the motor drive-end bearing inside the casing; NI USB-6212).
- **Speed check needed:** the paper's Fig. 5 marks outer-race harmonics at ≈111 Hz spacing; with the stated geometry BPFO ≈
  3.05·fr, i.e. fr ≈ 36 Hz (≈2180 RPM) rather than 29.2 Hz (1750 RPM). The measured speed channel must be used
  (label-independent) and the discrepancy documented before any external evaluation.

### Execution plan (sequential gates)
1. Restore shell; re-run tests (expect 95).
2. Write and freeze **protocol V2.2** (concept V3, fixed hyper-parameters, models B0–B4, Candidate A/B, PEV, PE-SD, shortcut
   probes, ablations A0–A6, decision rules) — before any Phase 3B run.
3. Implement concept V3 (identifiability indicators, masks, deterministic fill; 1-s and 4-s), with synthetic tests
   (speed uncertainty, nearby shaft harmonics) and real Paderborn diagnostics.
4. Phase 3B on PU-BD5 (5 outer folds; inner OOF only for temperature scaling and PE-SD; seeds 42/43/44; checkpoints + hashes).
5. PEV, PE-SD, shortcut probes, ablations, efficiency → `docs/PHASE3B_REPORT.md` with GO / MODIFY / STOP.
6. Only on GO: Ottawa metadata/speed audit (label-free), `docs/EXTERNAL_VALIDATION_PROTOCOL_V1.md` frozen with checkpoint
   hashes, single external evaluation → `docs/PHASE4_EXTERNAL_VALIDATION_REPORT.md`.
7. Literature verification (incl. Machines 2026), PLOS ONE LaTeX manuscript (official template), figures/tables from result
   files, reproducibility package, `docs/FINAL_REVIEWER_AUDIT.md`, `FINAL_PLOS_ONE_PACKAGE/`.

## Stage 1 — Environment restored; Phase 3B (2026-11-19)
- Shell restored by restarting from the project root; tests 95/95 before new work.
- Protocols V2.2 → V2.4 frozen (see `docs/PROTOCOL_FREEZE_V2.*.txt`), each before any Phase 3B classification result.
- Concept V3 implemented (`pave/features/physics_v3.py`, 14 synthetic tests); 4-s envelope cache (2 559 measurements).
- Models: `pave/models/pavenet.py` (PAVE-Net gated / physics / always modes; concept bottleneck), `pave/training/multi.py`.
- Full Phase 3B run started (`scripts/run_phase3b.py`, log `results/phase3b/log_run.txt`).
- Literature verification completed (28 references).

## Stage 2 — Phase 3B results and gate (2026-11-19)
- Full nested run (126 checkpoints, ~78 GPU-min) + FINAL models (21 checkpoints, ~11 min).
- Gate **GO**; final recommended model **B1-V3-4s** (bearing macro-F1 0.780 [0.56, 0.93]); latent branch not retained by the
  pre-specified inner-OOF rule. Report: `docs/PHASE3B_REPORT.md`.

## Stage 3 — External validation (2026-11-19)
- Label-free audit → EV-V1 speed policy revised before freeze → EV-V1 frozen (45 hashed files).
- **Single evaluation** 05:30:24 (`results/phase4/EVALUATED.lock`): B1-V3-4s macro-F1 0.518 [0.31, 0.68]; B0 0.614; CNN-based ≤ 0.31.
- Pre-specified secondary PEV on Ottawa (after lock) and a post-hoc exploratory analysis (labelled).
- Report: `docs/PHASE4_EXTERNAL_VALIDATION_REPORT.md`.

## Stage 4 — Figures and tables (2026-11-19)
- Fig 1–8 (`scripts/fig_schematics.py`, `scripts/make_figures.py`), TIFF 600 dpi LZW + PDF + PNG.
- Tables 1–10 and S1–S5 Tables generated from result files (`scripts/make_tables.py`, `scripts/make_tables_extra.py`).

## Stage 5 — Manuscript (2026-11-19)
- PLOS ONE LaTeX (official 2025 template): `manuscript/pavenet_main.tex` + `sections/01–04`; abstract 286 words; 28 verified references.
- Compiled: `pavenet_main.pdf` (submission, captions only), `pavenet_main_review.pdf` (figures embedded, 22 pp.),
  `supporting_tables.pdf`. 0 undefined references/citations.

## Stage 6 — Reproducibility, audit, package (2026-11-19)
- 110/110 tests pass. `REPRODUCE.md` extended to Phase 3B/4/manuscript.
- `docs/FINAL_REVIEWER_AUDIT.md` (17 categories, mean ≈ 7.6/10; major/minor concerns; submission checklist).
- `FINAL_PLOS_ONE_PACKAGE/` assembled (`scripts/assemble_package.py`), checksums in `reproducibility/SHA256SUMS.txt`.

## Remaining (requires the researchers)
Authors/affiliations/funding/competing interests; data & code deposition (repository + DOI); licence confirmation; full read-through;
cover letter; submission. Nothing has been submitted or uploaded.

## Stage 7 — Pre-submission method audit and manuscript v2 (2026-11-19)
- Audit documents: `docs/PRE_SUBMISSION_METHOD_AUDIT.md`, `docs/CLAIM_EVIDENCE_MATRIX.md`, `docs/REVISION_PLAN.md`,
  `docs/REVIEWER_RESPONSE_MEMO.md`, `docs/PLOS_COMPLIANCE_CHECKLIST.md`.
- Post hoc analyses (stored predictions only), in `results/phase5_audit/`:
  - fold-local selection gives the same model in 5/5 folds;
  - prevalence re-weighting.
- Manuscript v2:
  - v1 archived in `manuscript_archive/v1_2026-11-19_pre_audit/`;
  - unified diff and word-level HTML in `manuscript_archive/revision_2026-11-19/`;
  - authors and acknowledgment inserted;
  - PLOS statements added (software, data/code, statistics);
  - figures re-rendered to PLOS specification;
  - S1–S5 Table as separate files, plus S3 File.
- Remaining for the authors: `docs/REVISION_PLAN.md` §C.
