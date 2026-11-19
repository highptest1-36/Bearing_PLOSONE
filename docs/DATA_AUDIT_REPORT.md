# PAVE-Net Phase 1 — Comprehensive data audit report

**Dataset:** KAIST *Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load
Conditions for Fault Diagnosis* (Jung et al., Data in Brief 2023, doi:10.1016/j.dib.2023.109049; Mendeley ztmf3m7h5x,
version not encoded locally). **Date:** 2026-11-18. **Status:** audit complete; **no model was trained.**

Every number below was computed from the local files by `scripts/audit_data.py` and can be regenerated (see
`REPRODUCE.md`). Labels used: **VERIFIED** (measured from files), **PAPER** (stated in the source article, not checkable),
**INTERPRETATION** (our reading of verified facts), **ASSUMED** (convention adopted without evidence).

Detailed sub-reports: [integrity](audit/01_DATASET_INTEGRITY_REPORT.md) ·
[pairing](audit/02_VIBRATION_TEMPERATURE_PAIRING_REPORT.md) · [signal quality](audit/03_SIGNAL_QUALITY_REPORT.md) ·
[temperature](audit/04_TEMPERATURE_FEASIBILITY.md) · [physics](audit/05_PHYSICS_VALIDATION_REPORT.md) ·
[data dictionary](DATA_DICTIONARY.md) · decision: [DATA_FEASIBILITY_ASSESSMENT.md](DATA_FEASIBILITY_ASSESSMENT.md).

---

## 1. Inventory (VERIFIED)

- 98 files: 45 vibration `.mat`, 45 temperature/current `.tdms`, 5 acoustic `.mat` (0 Nm only), 3 zip archives.
  Every extracted file matches its archive entry (CRC32 + size); nothing missing, duplicated or unreadable.
- 15 conditions × 3 loads (0, 2, 4 Nm), **one recording per cell**. Bearing task: H, IR 0.3/1.0/3.0 mm, OR 0.3/1.0/3.0 mm
  → **21 recordings**; 24 misalignment/unbalance recordings form an out-of-scope challenge set.
- Durations: fault 60 s; healthy 300 s (0 Nm) and 120 s (2, 4 Nm); misalignment/unbalance 120 s. (Paper: healthy 120 s.)
- Manifest: `03_MANIFESTS/recordings.csv` (one row per recording, with SHA-256 of both modality files, measured fs,
  duration, start/end times, measured shaft speed, resolved temperature file, concurrency flag, session, acquisition order).

## 2. Vibration audit (VERIFIED unless noted)

- Schema: Test.Lab export `Signal.y_values.values` (N × 4, float64), fs = 25 600 Hz exactly, channels `Point1..4`
  (mapping to housing A/B × x/y **ASSUMED** from the paper's column order).
- No NaN/Inf, constant, duplicated or clipped channels; longest run of identical samples = 2; 1-s RMS CV ≤ 6.5 %.
- **Units unresolved:** metadata indicate stored SI (m/s²) with a display factor to g; the paper says g.
- Faulty recordings have 5–17× the healthy RMS; **severity is not monotonic** in RMS, kurtosis or evidence, and each
  condition's statistics are nearly constant across loads (installation signature).
- **Speed is not constant:** 49.88–50.19 Hz. Healthy and unbalance runs used a fixed VFD frequency (shaft slows to
  50.04 / 49.89 Hz at 2 / 4 Nm); IR, OR and misalignment runs were speed-corrected to 50.15–50.19 Hz.
  ⇒ **At 2 and 4 Nm, shaft speed alone separates Healthy from faulty recordings.**
- Figures: fig04 (waveforms), fig05 (PSD), fig11 (RMS/kurtosis), fig12 (speed).

## 3. Temperature audit (VERIFIED unless noted)

- Two K-type thermocouples, NI 9210, °C; logged at 25 608 Hz but **updated every 0.21 s (4.76 Hz)**; 0.01 °C steps.
  Mapping ai0/ai1 → housing A/B **ASSUMED**.
- Not in steady state: every recording warms at 0.17–1.13 °C/min.
- Fully confounded: H/IR/OR were recorded on three different days; loads were recorded in fixed 0→2→4 Nm order within
  warm-up blocks; on 25 Jun the start temperature tracks acquisition time (ρ = 0.95).
- Housing A − B is a per-condition fingerprint (η² by 7-way label 0.85, by fault group 0.05, by load 0.06).
- A parameter-free 1-NN on temperature features identifies conditions above chance under leave-one-load-out
  (7-way 0.43–0.57 with all features; n = 7 per fold) — a shortcut, not physics (**INTERPRETATION**).

## 4. Multimodal alignment (VERIFIED + INTERPRETATION)

- Two DAQ clocks; filename-paired start offsets cluster per day at +203.03 s (σ 0.09), +197.90 s (σ 0.90),
  +206.70 s (σ 0.11). Read as a clock offset with simultaneous acquisition (**INTERPRETATION**, physically unverifiable).
- Motor-current eccentricity sidebands give the shaft speed independently: gearbox ratio 2.071424 ± 0.000007 over 40
  pairs; concurrent pairs agree within 0.0007 Hz. This confirms recording identity, not sub-second timing.
- **Two TDMS files are swapped** (0 Nm IR 0.3 ↔ 1.0 mm) — proven by internal names, timing and speed; resolved in the
  manifest.
- **OR 0.3 mm (all loads): temperature/current logged ≈ 32 min after the vibration** — non-concurrent.
- Window-level alignment is not justified (0.21 s hold, ~0.01 °C change per window, ±1–3 s offset uncertainty).
  Only recording-level thermal context is defensible.

## 5. Physics validation (VERIFIED + INTERPRETATION)

- BPFI/BPFO formulas reproduce the paper exactly (fr = 3010/60 Hz). Paper BSF = 2 × standard BSF.
- **Pre-specified kinematic check (±1 %) agrees with labels in only 8/21 recordings.**
- Cause: every IR and OR recording has strong, sharp envelope harmonic families, but at −1.2 to −1.6 % (IR) and
  +0.9 to +2.5 % (OR) from the kinematic orders. BPFO_obs + BPFI_obs = 9.007 ≈ N, consistent with genuine race-defect
  frequencies and a ~2 % faster cage than predicted (e.g. a non-zero operating contact angle) — cause undetermined.
- 2×BPFI_obs coincides with 3×BPFO_kin (5.9 Hz apart): fixed templates above the 2nd harmonic are not class-specific.
- **Post-hoc label-free family search (±3.5 %, family contrast ≥ 6 dB, healthy-derived threshold):** expected family in
  100 % of 1-s windows of all 18 fault recordings; 1.1 % healthy and 1.7 % out-of-scope false calls (FIXED band);
  source-only thresholds transfer to the held-out load (fault windows 100 % correct; held-out 4 Nm healthy 83 % clean).
- Evidence strength is not graded by severity; the exact line position is a per-installation fingerprint.

## 6. Leakage prevention (VERIFIED by tests)

- Recording identifiers fixed before any windowing (`recording_id = L<load>_<condition>`).
- Protocol A (LOLO, primary): 3 folds; test = all 7 recordings of the held-out load; sources = other two loads;
  validation = last 20 % of each source recording with a 2-s purge gap (≥ longest window). Protocol B: 6 single-source
  permutations. Protocol C: within-recording temporal holdout (in-domain only). All in `03_MANIFESTS/splits/` with
  content hashes; `tests/test_manifests.py` re-validates them (no test recording in train/val; every recording tested
  exactly once in A; purge ≥ 2 s; hashes match). `source_only_stats` computes normalisation from train intervals only.
- **Explicit limitations of the split design:**
  - Validation shares recordings with training (not independent).
  - Each condition is one physical installation recorded at all three loads in one block, so LOLO tests transfer
    across load *within the same installation*, never across bearings.
  - There is no bearing-level replication; each LOLO test set has 7 recordings, so record-level metrics move in steps
    of 14 %.
  - Training time per class is imbalanced (Healthy 188–332 s vs 92 s per fault class per fold).

## 7. Confounds that any model will be able to exploit

| Confound | Affects | Evidence |
|---|---|---|
| Class ≡ acquisition day | H vs IR vs OR | H 25 Jun, IR 28 Jun, OR 30 Jun |
| Severity ≡ installation block | 7-way severity | one contiguous block per severity; per-block line offsets, RMS, A−B |
| Healthy speed not corrected | H vs faults at 2, 4 Nm | 0.13–0.30 Hz lower shaft speed |
| Load ≡ warm-up order | any temperature use | 0→2→4 Nm order; ρ = 0.95 with time |
| Temperature A − B | temperature models | η²(label7) = 0.85 |

## 8. Bottom line

The vibration data are clean, complete and contain strong, physically coherent bearing-fault evidence that survives
load changes, but only once fault orders are allowed to deviate from the textbook kinematics. Temperature is clean but
slow, non-stationary and confounded with day, warm-up and installation; it cannot support fault-related multimodal
claims. Severity labels are confounded with specimen/installation identity. See
[DATA_FEASIBILITY_ASSESSMENT.md](DATA_FEASIBILITY_ASSESSMENT.md) for the PASS/CONCERN/FAIL decision.
