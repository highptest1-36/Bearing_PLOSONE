# External validation protocol EV-V1 — University of Ottawa UORED-VAFCLS

Status: written before any Ottawa signal was analysed; §3 revised after the label-free audit (speed encoding) and before any
Ottawa feature, prediction or label-dependent statistic. **Binding once its SHA-256, the checkpoint hashes and the
evaluation-code hashes are recorded in `docs/PROTOCOL_FREEZE_EV_V1.txt`.**
Governing development protocols: V2.2 (`078d726b…`), V2.3 (`6b4e4fdd…`), V2.4 (`dd758b7f…`).

## 1. Dataset and version
- University of Ottawa Rolling-element Dataset – Vibration and Acoustic Faults under Constant Load and Speed conditions
  (UORED-VAFCLS); descriptive paper: Sehri, Dumond & Bouchard, Data in Brief 49 (2023) 109327 (CC BY 4.0).
- Version: user-stated version 5 (doi 10.17632/y2px5tg92h.5). The local files do not encode a version; the SHA-256 of every
  file used is recorded in `results/phase4/ottawa_audit.csv` and is the operative identification.
- Files: the 60 MATLAB files in `3_MatLab_Raw_Data_Files (.mat)` (one array 420 000 × 5 per file). CSV/XLSX copies and the
  spectrogram archives are not used.

## 2. Physical units
20 physical bearings (1–20); each contributes up to three recordings (healthy, developing, faulty). Recordings of one bearing
are repeated observations. Uncertainty is estimated by bootstrap resampling of physical bearings (all recordings of a
resampled bearing kept together; 10 000 resamples).

## 3. Geometry and speed
- Geometry (paper Table 2): 8 balls, ball diameter 6.77 mm, pitch diameter 28.50 mm, contact angle 0° (not stated → 0).
  BPFO = 3.0498·f_r, BPFI = 4.9502·f_r.
- **Label-free audit performed before this protocol was frozen** (`results/phase4/ottawa_audit.csv`,
  `results/phase4/ottawa_speed_audit.csv`): all 60 files are 420 000 × 5 with no NaN/Inf; the speed column is **zero except at
  sample 0**, which holds a single per-file reading (1700–2190 rpm across files); the load column likewise holds one value at
  sample 0 (400 N or 0 N). The nominal 1750 rpm is therefore not the operating speed of every file. A cross-check with the
  strongest accelerometer peak within ±10 % of the reading at 1×/2× shaft rate differs by −8.5 % … +8.6 % (median −0.7 %); it is
  too noisy to be used and is reported only.
- **Speed policy (frozen):** f_r = (value of the speed column at sample 0)/60 per file; σ_r = 0.5 rpm/60 (half the 1-rpm reading
  resolution). Failure handling: if the reading is outside 600–3000 rpm, use 1750 rpm and flag the file.
- **Pre-specified sensitivity analysis (primary model only):** σ_r = 0.01·f_r (wider regions and guards).

## 4. Label mapping (frozen; `pave/data/ottawa.py`, unit-tested)
| Recording | Primary task | Role |
|---|---|---|
| H_n_0, n = 1..20 | H | primary (20 recordings) |
| I_n_2, n = 1..5 | IR | primary (5) |
| O_n_2, n = 6..10 | OR | primary (5) |
| I_n_1, O_n_1 | — | secondary: developing faults (10) |
| B_n_1/2, C_n_1/2 (n = 11..20) | — | out of scope (ball/cage; 20 recordings): OOD analysis only, never relabelled as healthy |

## 5. Preprocessing (identical to development code paths)
- Signal: accelerometer column 1 (m/s²), fs = 42 000 Hz.
- 4-s physics records: samples [0, 4 s) and [4 s, 8 s); envelope band 1–20 kHz; FFT length 4·fs (Δf = 0.25 Hz); V3 concepts
  with the FINAL PU calibration (relative regions, standardisation μ/σ and q statistics from the 15 PU-BD5 bearings).
- 1-s windows: 10 per recording; V2 and V3-1s concepts (Δf = 1 Hz); CNN input resampled 42 → 32 kHz (polyphase 16/21).
- Window-to-record mapping for fusion models: windows 0–3 → record 0, 4–7 → record 1, windows 8–9 → record 1 (nearest).
- **Amplitude normalisation for CNN-based models (primary):** PU training statistics as frozen in the checkpoints
  (no target adaptation). Physics concepts are ratios (dB) and unit-independent.
- Pre-specified secondary analysis (label-free, test-time): per-recording standardisation of CNN inputs.

## 6. Models (frozen FINAL checkpoints; 3-seed ensembles)
B0, B1-V2, B1-V3-1s, B1-V3-4s, B2, B3, B4/A1, A2, A3, A4 (PAVE-Net), A6 — retrained on all 15 PU-BD5 bearings with the
V2.2 configurations; logistic models refitted deterministically on the same data. **Primary model = the final
recommended model of the development gate** (`results/phase3b/tables/inclusion_decision.json`). All others are reported.

## 7. Calibration and selective diagnosis (frozen from PU)
- Temperature per model: NLL-optimal on the pooled PU outer-fold test windows (15 bearings).
- PE-SD: S1–S4 as in V2.2 §8; S4 coefficients fitted on pooled PU outer-test windows; τ = FINAL B0 threshold.
- Acceptance thresholds: the bearing-level score of the 12th (coverage 0.8) and 9th (0.6) ranked PU-BD5 bearing; applied to
  Ottawa recordings; realised coverage reported. Matched-coverage comparison: top 80 % / 60 % of the 30 primary recordings.

## 8. Metrics
Primary: recording-level macro-F1 on the 30 primary recordings (recording decision = argmax of the mean window probability),
cluster-bootstrap 95 % CI over physical bearings. Secondary: balanced accuracy, per-class recall/precision, healthy
false-alarm rate (20), fault sensitivity (10), IR/OR localisation among detected faults, window NLL, ECE, Brier,
selective risk/coverage/AURC, developing-fault detection, OOD predicted-class distribution and rejection, PEV on the 10
faulty IR/OR recordings (B1-V3-4s, A4, B3), concept validity rates, speed-policy diagnostics.

## 9. Single-evaluation rule and failure handling
The evaluation script refuses to run if `results/phase4/EVALUATED.lock` exists. Any later analysis is labelled post-hoc and
cannot replace the primary result. Files that fail to load are reported and excluded (no substitution). If any frozen
artefact hash does not match, the evaluation is aborted.
