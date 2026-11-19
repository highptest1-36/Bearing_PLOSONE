# PAVE-Net — Protocol amendment V2.4

Amends V2.3 (`6b4e4fdd…`, unchanged). Frozen 2026-11-19 before any Phase 3B classification, intervention or
selective-diagnosis result was produced. SHA-256 in `docs/PROTOCOL_FREEZE_V2.4.txt`.

## What was inspected before this amendment
A 1-epoch smoke run of `scripts/run_phase3b.py` on outer fold O1 (outputs in `results/phase3b/_smoke/`, never used for
results). Inspected **only label-free quantities**: calibrated 4-s regions (BPFO centre −0.02 %, half-width 1.74 %;
BPFI centre +0.05 %, half-width 1.27 %) and the fraction of measurements with valid families per setting ×
manufacturer. No class-wise statistic, no model output and no metric was inspected.

## Finding
With the V2.2/V2.3 region-based identifiability test (the whole calibrated search region, widened by the calibrated
half-width, must clear the guarded shaft harmonic by ≥ 1 bin), **0 % of PU measurements had a valid BPFO or BPFI family
at every setting and manufacturer, at 4 s as well as 1 s** (mean δ −0.05 … −3.7 bins). The training-calibrated
half-width (1.3–1.7 % ≈ 1.0–1.3 Hz at 1500 rpm) expresses the across-bearing uncertainty of the line position and is
comparable to the 1.4–1.8 Hz gap between the 6203 fault orders and 3·fr / 5·fr. The region-based test therefore
cannot distinguish a separable fault line from shaft-harmonic leakage, and masks all evidence.

## Amendment (identifiability of the *measured* line)
For family F and harmonic h, within the calibrated region (guard bins excluded) let f0 be the detected peak bin.
- δ_{F,h} = (|f0 − k·fr| − g_k)/Δf for the nearest shaft harmonic k (g_k = Δf + 3·k·σ_fr as before).
- **The harmonic is identifiable iff δ_{F,h} ≥ 1 and E(f0) is a local maximum (E(f0) ≥ E(f0 ± Δf)).**
  With a Hann window the shaft line's main lobe spans ±2 bins; δ ≥ 1 puts f0 outside it.
- valid_F = fundamental identifiable. Harmonic-level exclusion (overlap with other-family regions or guards) is unchanged.
- Reliability features `ident_BPFO`, `ident_BPFI` = clip(δ_{F,1}, −5, 20) of the measured line (0 → −5 if no peak).
- The region-based δ of V2.3 is still computed and reported (`delta_region_*`), not used for masking.
Everything else in V2.2/V2.3 (models, configurations, seeds, calibration, PEV, PE-SD, decision rules) is unchanged.
At 1 s (Δf = 1 Hz) a line needs ≥ 2 Hz separation, so 1-s PU concepts are still expected to be largely masked.
