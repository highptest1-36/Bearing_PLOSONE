# Physics validation report (NSK 6205 DDU, envelope analysis)

Code: `pave/signal/physics.py`, `pave/signal/envelope.py`, `pave/audit/vibration_audit.py`, `pave/audit/analysis.py`.
Tables: `results/audit/tables/physics_*.csv`, `frequency_collisions.csv`, `shaft_speed_and_fault_frequencies.csv`.
Figures: `fig06_envelope_spectra_*`, `fig07_evidence_heatmap_*`, `fig08_window_detection_*`, `fig13`–`fig15`.

## 1. Theoretical frequencies (unit-tested)

Geometry from Jung et al. 2023: N = 9, d = 7.90 mm, D = 38.5 mm, θ = 0°. In shaft orders: FTF 0.3974,
**BPFO 3.5766**, **BPFI 5.4234**, BSF 2.3342. At fr = 3010/60 = 50.1667 Hz: BPFO 179.43 Hz, BPFI 272.07 Hz, FTF 19.94 Hz,
exactly the paper's values (the paper's "50.17 Hz" is rounded; 50.17 Hz gives BPFI 272.09 Hz). The paper's BSF 234.19 Hz
is 2 × the standard BSF. Measured speeds range 49.88–50.19 Hz, so BPFI at the measured speed differs from the paper value
by up to −1.5 Hz; all analyses use the per-recording measured speed (1X/2X peak, all 4 channels agree within 0.0004 Hz).

## 2. Method (label-free)

Band-pass → Hilbert envelope → averaged envelope amplitude spectrum (whole record: 4-s segments, 0.25 Hz resolution;
windows: 1 s, 1 Hz). Two band choices, neither using labels: **FIXED** 1–10 kHz (pre-specified) and **SK** 2 kHz band at
the spectral-kurtosis maximum of each channel (selected centres 3.1–11 kHz). Statistic: peak in a tolerance band / median
of a ±20 Hz ring excluding shaft harmonics, compared with the same statistic at 24 control orders that avoid shaft
harmonics, half-orders, and the BPFO/BPFI/BSF harmonics ±2 sidebands. Synthetic unit tests confirm detection of impulse
trains, rejection of noise, and correct band selection.

## 3. Result A — pre-specified kinematic definition (±1 % around theory): mostly FAILS

Record-level call "detected" = excess ≥ 6 dB over the control median **and** above all 24 controls, housing A x or y:

| | Healthy | IR (9) | OR (9) |
|---|---|---|---|
| Expected call obtained | 3/3 none | 2/9 (IR 3.0 mm at 2, 4 Nm) | 3/9 (OR 0.3 mm, all loads) |

Overall 8/21 = 38 % agreement; the 24 out-of-scope recordings all give "none".

## 4. Why it fails — the fault lines are real but shifted (VERIFIED)

Envelope spectra of **every** IR and OR recording contain strong, sharp harmonic families (fundamental 43–56 dB above
a 40-Hz running-median floor, housing A x, whole record). Their positions, from the strongest line within ±3.5 % of theory (median over all 1-s windows, FIXED band):

| Condition | Observed order | Kinematic | Offset | IQR over windows |
|---|---|---|---|---|
| IR 0.3 mm | 5.360 | 5.4234 | −1.17 % | 0.0007 |
| IR 1.0 mm | 5.340 | 5.4234 | −1.55 % | 0.0011 |
| IR 3.0 mm | 5.360 | 5.4234 | −1.18 % | 0.0006 |
| OR 0.3 mm | 3.607 | 3.5766 | +0.85 % | 0.020 |
| OR 1.0 mm | 3.666 | 3.5766 | +2.50 % | 0.019 |
| OR 3.0 mm | 3.646 | 3.5766 | +1.95 % | 0.0018 |

- IR families show the textbook ±1×fr sidebands (4.35, 6.35 orders) and harmonics at 10.7 and 16.0 orders.
- OR families show harmonics at 7.2–7.3, 10.85–10.97, 14.5–14.6 and 18.1–18.3 orders.
- **Kinematic consistency check:** for any slip or contact angle, BPFO + BPFI = N·fr = 9 orders. Observed
  3.652 (OR 1.0/3.0 mean) + 5.355 (IR mean) = 9.007. The data are therefore consistent with genuine IR/OR defect
  frequencies whose cage ratio is ≈ 2 % *higher* than the kinematic prediction. This corresponds to an effective
  (d/D)·cosθ ≈ 0.189 instead of 0.205, e.g. a ≈ 23° operating contact angle under axial load, or a geometry deviation.
  Ordinary slip would shift frequencies the other way, so slip alone does not explain it. **This is an interpretation;
  the cause cannot be determined from the data.**
- The offset is specific to each condition (installation) and identical across its three loads (IQR ≤ 0.002 orders
  for most), so it acts as a fingerprint of the specimen/mounting.

## 5. Frequency collisions

| Target | Nearest component of the other family | Distance at paper speed | Within ±2 tolerances? |
|---|---|---|---|
| 3×BPFO (10.73) | 2×BPFI (10.85) | 5.9 Hz | **yes** |
| 2×BPFI (10.85) | 3×BPFO | 5.9 Hz | **yes** |
| BPFO, BPFI fundamentals | BPFI−2fr / BPFO+2fr | 7.7 Hz | no |

With the observed shifts, 2×BPFI_obs (10.70 orders) falls inside the ±1 % window of 3×BPFO_kin: in the strict analysis
every IR recording showed a large "3×BPFO" excess (+18 to +33 dB) and OR recordings a large "2×BPFI" excess. **Fixed
harmonic templates beyond the 2nd harmonic are not class-specific on this rig.** In the ±3.5 % BPFI band, OR recordings
also contain 2×BPFO_obs − 2fr (≈ 5.31 orders), so single-band evidence must be read as a *contrast* between families.

## 6. Result B — label-free family search (POST-HOC; defined after Result A was observed)

Per 1-s window: score = mean(SNR of strongest line in the ±3.5 % band, SNR of its 2nd harmonic), max over housing A x/y;
threshold = 99th percentile of healthy windows; call requires the winning family to lead the other by ≥ 6 dB.

| | FIXED band | SK band |
|---|---|---|
| Expected call, IR/OR windows (18 recordings) | 100 % in every recording | 95–100 % per recording |
| Healthy windows with a fault call | 1.1 % (all at 4 Nm) | 0.6 % |
| Out-of-scope windows (24 recordings, independent negatives) with a fault call | 1.7 % (mostly 4 Nm unbalance, IR-like) | 0.8 % |
| Source-only LOLO thresholds (healthy windows of the 2 source loads) → held-out load: fault windows correct | 100 % | 95–100 % |
| … held-out healthy windows with no fault call (0/2/4 Nm) | 100 / 100 / 83 % | 99 / 99 / 100 % |

Evidence strength (median family score, dB) does not decrease with load and is **not monotonic in severity**
(IR: 38–40 / 42–44 / 31–36 dB for 0.3 / 1.0 / 3.0 mm). Severity therefore cannot be read from evidence strength.

## 7. Where expected evidence is weak or absent

- With the pre-specified kinematic definition: absent for 13/18 fault recordings (Result A).
- With the family search: present in ≥ 95 % of windows of every fault recording; weakest for OR 1.0 mm in the SK band
  (95–97 %). Healthy 4 Nm windows are the main false-positive source (broadband envelope content rises with load,
  fig06), as is 4 Nm unbalance.
- No fault recording lacks evidence entirely. Evidence is *not* graded by severity.

## 8. Implications

1. Physics evidence concepts are **measurable and robust across loads** — but only if defined with measured speed and
   data-derived (or widened) fault orders; the textbook kinematic constants alone fail on this rig.
2. Evidence nearly determines the 3-way label at window level. A concept bottleneck on these concepts will be close to
   a deterministic rule for H/IR/OR; its value lies in faithfulness and auditing, not accuracy.
3. Evidence does not encode severity; 7-way discrimination must rely on other signal properties that, on this dataset,
   coincide with installation identity (RMS level, exact line position, resonance structure).
4. The ±3.5 % window and family-contrast rule were chosen after seeing Result A. Before any paper use they must be frozen
   and re-justified; any calibration of effective orders must be fit on source folds only.
