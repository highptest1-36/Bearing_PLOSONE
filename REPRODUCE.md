# Reproducing the Phase 1 data audit

Requirements: Python 3.10, `pip install -r requirements.txt`, the KAIST varying-load dataset extracted under
`01_RAW_DATA/<dataset folder>/{vibration,current,temp,acoustic}` (read-only). ~3 GB free RAM is enough; files are
processed one at a time.

```bash
python -m pytest -q tests                              # 61 tests (synthetic fixtures + real-manifest checks)
python scripts/audit_data.py --stage inventory         # SHA-256/CRC32 of 98 files vs archives  (~1 min)
python scripts/audit_data.py --stage tdms              # 45 temperature/current files          (~1 min)
python scripts/audit_data.py --stage vibration         # 45 vibration files                    (~12 min)
python scripts/audit_data.py --stage analyze           # pairing, manifests, splits, tables, figures (~1 min)
# or: python scripts/audit_data.py --stage all
```

Per-file results are cached in `results/audit/cache/`; add `--force` to recompute.

| Output | Location |
|---|---|
| File inventory, hashes, provenance | `03_MANIFESTS/file_inventory.csv`, `provenance.json` |
| Recording manifest / pairing | `03_MANIFESTS/recordings.csv`, `pairs.csv` |
| Split manifests (A LOLO, B permutations, C within-recording) | `03_MANIFESTS/splits/` |
| Tables | `results/audit/tables/` |
| Figures (PNG, 300 dpi) | `results/audit/figures/` |
| Machine-readable summary | `results/audit/audit_summary.json` |
| Reports | `docs/DATA_AUDIT_REPORT.md`, `docs/DATA_FEASIBILITY_ASSESSMENT.md`, `docs/audit/`, `docs/DATA_DICTIONARY.md` |

Environment used: see `environment_report.txt`.

## Phase 2 (Paderborn + baselines)

```bash
# download the 32 archives from https://groups.uni-paderborn.de/kat/BearingDataCenter/<code>.rar
#   into 01_RAW_DATA/Paderborn_KAt_BearingDataCenter/archives/   (see 03_MANIFESTS/paderborn/pu_bearings.csv for SHA-256)
python -I scripts/build_paderborn_cache.py --tmp <scratch dir>   # ~4 min; extracts .mat temporarily, keeps the .rar
python -I scripts/audit_paderborn.py                              # metadata audit
python -I scripts/build_paderborn_splits.py                       # frozen bearing-disjoint split manifests
python -I scripts/build_windows.py --dataset kaist                # ~0.5 min
python -I scripts/build_windows.py --dataset pu                   # ~3 min
python -I scripts/run_phase2.py --dataset kaist                   # KAIST LOLO, 3-way + exploratory 7-way
python -I scripts/run_phase2.py --dataset pu --protocol PU_BD5    # bearing-disjoint, 5 folds
python -I scripts/run_phase2.py --dataset pu --protocol PU_A2R
python -I scripts/report_phase2.py                                # tables, figures, gates.json
python -I scripts/physics_diagnostics_phase2.py                   # physics + metadata-confound diagnostics
python -m pytest -q tests
```
Total Phase 2 GPU training time on an RTX 3060 Laptop: 15.5 min (99 CNN runs).

## Phase 3A (protocol V2.1)

```bash
python -I scripts/pu_verification_phase3a.py     # 32-bearing table, fold composition, Lessmeier Table 8/10 comparison
python -I scripts/run_phase3a.py                 # nested B0-B3 (5 outer x 4 inner folds), ~50 GPU-min; resumable per fold
python -I scripts/report_phase3a.py              # bearing-level metrics, bootstrap CIs, McNemar, selection diagnostics
python -I scripts/concept_audit_phase3a.py       # concept reliability + confound analyses (S-4b, S-4c)
python -I scripts/interventions_phase3a.py       # manipulation check + exploratory model response (~8 min)
python -m pytest -q tests                        # 95 tests
```

## Phase 3B (protocols V2.2–V2.4)

```bash
python -I scripts/build_env4s.py                 # 4-s envelope spectra of all PU measurements (~3 min)
python -I scripts/run_phase3b.py                 # 5 outer folds x 4 inner folds, 11 models, seeds 42-44 (~78 GPU-min, RTX 3060 6 GB)
python -I scripts/run_phase3b.py --final         # FINAL models on all 15 PU-BD5 bearings (~11 min)
python -I scripts/report_phase3b.py              # bearing-level metrics, inclusion rule, PE-SD, shortcut probes
python -I scripts/pev_phase3b.py                 # physics-evidence verification (300 records x 13 variants)
python -I scripts/pev_report_phase3b.py
python -I scripts/efficiency_phase3b.py
```

## Phase 4 — Ottawa external validation (protocol EV-V1; evaluated ONCE)

Place the 60 `.mat` files of UORED-VAFCLS (Mendeley Data doi 10.17632/y2px5tg92h.5, CC BY 4.0) under
`01_RAW_DATA/Ottawa_data/` (any sub-folder; files are found by name; SHA-256 in `results/phase4/ottawa_audit.csv`).

```bash
python -I scripts/external_ottawa.py --stage audit       # label-free integrity + speed audit
python -I scripts/external_ottawa.py --stage features    # FINAL-PU-calibrated Ottawa features (no labels)
python -I scripts/external_ottawa_eval.py                # verifies all frozen hashes; refuses to run if EVALUATED.lock exists
python -I scripts/pev_ottawa.py                          # pre-specified secondary PEV (after the lock)
python -I scripts/posthoc_ottawa_exploratory.py          # post-hoc exploratory analysis (labelled as such)
```
To re-run the evaluation in an independent copy, the lock file must be absent; in this repository it is present and must stay.

## Manuscript, figures and tables

```bash
python -I scripts/fig_schematics.py              # Fig 1-2
python -I scripts/make_figures.py                # Fig 3-8 from result files (asserts consistency with frozen tables)
python -I scripts/make_tables.py                 # all LaTeX/CSV tables from result files
python -I scripts/build_manuscript.py            # pdflatex + bibtex (MiKTeX); submission PDF, review PDF with figures, S tables
python -m pytest -q tests                        # 110 tests
```
Checksums of all deliverables: `FINAL_PLOS_ONE_PACKAGE/reproducibility/SHA256SUMS.txt`.

## Pre-submission audit (post hoc; stored predictions only) and manuscript v2
```
python scripts/audit_foldlocal_selection.py      # fold-local re-application of the inclusion rule -> results/phase5_audit/
python scripts/audit_prevalence.py               # prevalence re-weighting of reported confusion matrices
python -I scripts/fig_schematics.py && python -I scripts/make_figures.py   # Fig1-Fig8 (Arial >= 8 pt, RGB TIFF)
python -I scripts/make_tables.py                 # tables
python scripts/assemble_clean_tex.py            # v3: single clean manuscript/pavenet_main.tex from the archived template + tables
python scripts/build_manuscript.py               # pavenet_main.pdf, pavenet_main_review.pdf, S1-S5_Table.pdf
python scripts/make_tracked_changes_v2_v3.py    # v2 -> v3 diff and word-level view
python -I scripts/make_tracked_changes.py        # word-level v1 -> v2 view
python scripts/assemble_package.py               # FINAL_PLOS_ONE_PACKAGE/
python scripts/make_submission_folder.py         # PLOS_ONE_SUBMISSION/ (flat upload folder: main.tex, bib, bst, TIFFs, SI)
```
