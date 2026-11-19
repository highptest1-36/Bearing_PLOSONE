# Vibration signal-quality report

Source table: `results/audit/tables/signal_quality.csv` (45 recordings × 4 channels = 180 rows).
Figures: `fig04_waveforms_{0,4}Nm.png`, `fig05_psd_0Nm.png`, `fig11_signal_quality_x_housing_{A,B}.png`, `fig12_shaft_speed.png`.

## 1. Integrity (VERIFIED, all 180 channels)

| Check | Result |
|---|---|
| Array shape / dtype | (N, 4) float64; N = 25 600 × duration exactly; header `number_of_values` = actual N |
| Sampling | increment 3.90625e-05 s ⇒ **25 600 Hz** in every file; time axis starts 0.4–38 µs after t=0 |
| NaN / Inf | 0 |
| Constant channels | 0 |
| Duplicated channels | 0 (pairwise identity test on the first 200 000 samples) |
| Clipping | none: no file has ≥ 2 consecutive samples at its extreme value |
| Longest run of identical samples | 2 samples (all channels) |
| Amplitude resolution | smallest step 1.2e-4 stored units (healthy RMS ≈ 1.0), i.e. negligible |
| Stationarity | 1-s RMS coefficient of variation 0.2–6.5 % (max: L4_Misalign_0p5 y_A); first-vs-last-5-s RMS drift within ±15 % |
| Shaft-speed stability within a run | σ of 2-s 2X tracking ≤ 0.010 Hz |

## 2. Units (UNRESOLVED, documented)

`y_values.quantity.label = 'g'`, `info = 'Data saved by Test.Lab in MKS'`, `unit_transformation.factor = 0.10197`
(=1/9.80665). In the Test.Lab export convention this means the stored numbers are SI (m/s²) and `g` is the display
unit. The paper states the unit is g. Healthy RMS (housing A x) is 1.00 / 1.65 / 2.50 stored units at 0 / 2 / 4 Nm;
the m/s² reading (0.10–0.25 g RMS) is the more physically plausible one. **Normalised models are unaffected; any
absolute amplitude threshold, ISO severity statement or cross-dataset comparison must state the assumption.**

## 3. Descriptive statistics by condition (housing A x; stored units)

| Condition | RMS 0 / 2 / 4 Nm | Kurtosis 0 / 2 / 4 Nm |
|---|---|---|
| Healthy | 1.00 / 1.65 / 2.50 | 3.05 / 2.87 / 2.54 |
| IR 0.3 mm | 9.35 / 10.24 / 10.95 | 11.15 / 9.86 / 9.05 |
| IR 1.0 mm | 13.26 / 13.47 / 13.68 | 4.24 / 4.25 / 4.18 |
| IR 3.0 mm | 7.37 / 7.67 / 7.77 | 4.17 / 4.03 / 4.00 |
| OR 0.3 mm | 5.24 / 5.49 / 6.09 | 4.18 / 4.09 / 3.70 |
| OR 1.0 mm | 17.13 / 17.39 / 17.33 | 4.23 / 4.14 / 4.09 |
| OR 3.0 mm | 8.97 / 9.44 / 9.83 | 3.30 / 3.26 / 3.38 |

Observations:
- Every faulty recording has 5–17× the healthy RMS: the fault classes are trivially separable from Healthy by energy.
- **Severity is not monotonic** in RMS, kurtosis or envelope evidence (1.0 mm > 3.0 mm for both IR and OR in RMS;
  0.3 mm is the most impulsive IR). Each condition's statistics are almost constant across loads, i.e. they behave as
  a per-installation signature. This is consistent with a separate specimen/mounting per severity and means the
  data do not physically validate an ordered "severity" variable.
- Highest kurtosis: L0_IR_0p3 y_A = 19.1.

## 4. Operating speed (VERIFIED; important confound)

| Group | VFD supply (Hz) | Shaft speed 0 / 2 / 4 Nm (Hz) |
|---|---|---|
| Healthy, Unbalance (25 Jun, no speed correction) | 48.497 at all loads | 50.156 / 50.04 / 49.88–49.89 |
| IR, OR, Misalignment (run names "RPMFix") | 48.50–48.82, raised with load | 50.15–50.19 at all loads |

At 2 and 4 Nm the Healthy recordings rotate 0.13–0.30 Hz slower than every faulty recording. A model can therefore
identify "Healthy" from shaft speed (or from the positions of high shaft harmonics, e.g. 13×fr shifts ≈ 4 Hz) rather
than from bearing condition. Mitigations for later phases: order-domain resampling with the measured speed, or a
speed-shortcut control experiment.
