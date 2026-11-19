# Vibration–temperature pairing and alignment report

Sources: `03_MANIFESTS/pairs.csv`, `results/audit/figures/fig02_acquisition_timeline.png`, `fig03_clock_offsets.png`.
Code: `pave/data/alignment.py`, `pave/audit/analysis.py::build_pairs`.

## 1. Two independent clocks (VERIFIED)

Vibration (Siemens SCADAS / Test.Lab) stores `absolute_time` with millisecond resolution; temperature/current
(NI cDAQ-9185 / FlexLogger) stores `wf_start_time` with microsecond resolution. Neither file references the other
system, no trigger channel exists in either file, and the paper does not describe synchronisation.

Pairing files by filename gives start-time offsets (vibration − TDMS) that cluster tightly **per acquisition day**:

| Session | Clock offset (robust median) | Robust σ | Pairs used / total |
|---|---|---|---|
| 25 Jun 2021 | +203.03 s | 0.09 s | 27 / 27 |
| 28 Jun 2021 | +197.90 s | 0.90 s | 7 / 9 (2 swapped files excluded) |
| 30 Jun 2021 | +206.70 s | 0.11 s | 6 / 9 (3 OR 0.3 mm excluded) |

A deliberate 3-minute start delay could not reproduce a ±0.1 s spread over 27 manually started runs, and the
offset changes by several seconds between days as unsynchronised PC clocks do. **Interpretation (not proof):** the
two systems recorded simultaneously and their clocks differed by ≈ 198–207 s. Under the alternative reading
(clocks correct, TDMS started 3 min earlier), no TDMS log would overlap any vibration recording at all.

## 2. Independent physical identity check (VERIFIED)

The motor current contains rotor-eccentricity sidebands at f_supply ± f_rotor,motor. The motor-shaft speed from the
current, multiplied by the gearbox ratio, must equal the output-shaft speed measured from the vibration spectrum if
both files come from the same run. Across the 40 label- and time-consistent pairs the ratio is
**2.071424 ± 0.000007** (paper: gearbox ×2.07). With this ratio, every time-consistent pair agrees on shaft speed
within **0.0007 Hz**, while the VFD supply frequency differs between runs (48.497–48.817 Hz), so speed is a sensitive
run identifier.

## 3. Pairing decisions (45 recordings)

| Status | n | Recordings | Evidence |
|---|---|---|---|
| `CONCURRENT_BY_NAME` | 40 | all others | clock-corrected start within 5 s (|residual| median 0.08 s, max 2.76 s), labels agree, speed agrees |
| `FILENAME_SWAP_RESOLVED` | 2 | L0_IR_0p3 ↔ `0Nm_BPFI_10.tdms`; L0_IR_1p0 ↔ `0Nm_BPFI_03.tdms` | (i) internal names `LogFile_0Nm_Inner_03mm` / `_1mm`; (ii) residual −0.10 / −0.65 s after swap vs +1595 / −1596 s before; (iii) speed mismatch 0.0002 / 0.0005 Hz after swap |
| `NON_CONCURRENT` | 3 | L0/L2/L4_OR_0p3 | TDMS starts ≈ 31.9 min after the vibration (clock-corrected); speed mismatch 0.005–0.007 Hz (7–10× the concurrent maximum); same VFD setpoint; no other TDMS overlaps these vibration files |

The original files are untouched; the corrected mapping lives only in `pairs.csv` / `recordings.csv` (`resolved_tdms`).

## 4. Is window-level time alignment justified?

**No, not as fine-grained synchronous fusion.**

1. Synchronisation rests on an *inferred* clock offset. Its within-day scatter (σ 0.1–0.9 s, single runs up to 2.8 s)
   is the uncertainty of any window-to-temperature mapping, and it cannot be verified physically:
   - slow RMS fluctuations of vibration and current are uncorrelated (|r| < 0.11 at all lags), and
   - shaft speed is too stable within a run (σ ≤ 0.01 Hz over 1-s blocks) to provide a common transient.
2. Temperature only changes every 0.21 s and the thermocouple/housing respond over seconds to minutes. A 1–2 s
   vibration window maps onto 5–10 temperature samples that differ by about one 0.01 °C step, so window-level pairing
   adds no information beyond the recording-level temperature trend.
3. For OR 0.3 mm no concurrent temperature exists at any load.

**Admissible use:** condition-level / recording-level context (mean level, slope over the recording, A−B), aligned
by the clock-offset model with ±3 s tolerance, with OR 0.3 mm marked missing/non-concurrent. Any claim of
"synchronous vibration–temperature fusion" is unsupported.

## 5. Unresolved uncertainties

- The clock-offset interpretation cannot be confirmed without the original DAQ configuration or a shared trigger.
- Why OR 0.3 mm was re-logged ≈ 32 min later is unknown (operator log not available).
- Channel-to-housing mapping for both temperature channels and all accelerometers is assumed from column order.
