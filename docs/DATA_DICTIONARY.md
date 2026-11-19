# Data dictionary — KAIST varying-load dataset (PAVE-Net Phase 1)

Status key: **VERIFIED** = read from the local files by `scripts/audit_data.py` (2026-11-18);
**PAPER** = stated in Jung et al. 2023 (Data in Brief, doi:10.1016/j.dib.2023.109049), not verifiable from files;
**ASSUMED** = project convention adopted because neither files nor paper settle it.

## 1. Raw files (read-only)

Root: `01_RAW_DATA/Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis/`

| Folder | Files | Format | Content | Status |
|---|---|---|---|---|
| `vibration/` | 45 `.mat` | MATLAB 5.0, Siemens Simcenter Test.Lab export | 4 accelerometer channels | VERIFIED |
| `current,temp/` | 45 `.tdms` | NI TDMS, FlexLogger 2021 R1.1 | 2 thermocouples + 1–3 motor-current phases | VERIFIED |
| `acoustic/` | 5 `.mat` | Test.Lab export | microphone, 0 Nm only (H, IR 0.3/1.0, OR 0.3/1.0) | VERIFIED (listing only; not audited, outside PAVE-Net scope) |
| `*.zip` | 3 | original archives | every extracted file's CRC32 and size match its archive entry | VERIFIED |

File naming: `<load>Nm_<condition>[_<severity>]` with `load ∈ {0,2,4}`;
`Normal`; `BPFI_{03,10,30}` = inner race 0.3/1.0/3.0 mm; `BPFO_{03,10,30}` = outer race 0.3/1.0/3.0 mm;
`Misalign_{01,03,05}` = 0.1/0.3/0.5 mm parallel misalignment; `Unbalance_{0583,1169,1751,2239,3318}mg`.
The 2 Nm unbalance **vibration** files are spelled `Unbalalnce` (typo in the distribution; handled by the parser).

## 2. Vibration `.mat` schema (VERIFIED)

| Path inside file | Type / value | Meaning |
|---|---|---|
| `Signal.y_values.values` | float64, shape (N, 4) | samples; N = 1 536 000 (60 s), 3 072 000 (120 s) or 7 680 000 (300 s, 0 Nm Healthy) |
| `Signal.x_values.increment` | 3.90625e-05 s | fs = 25 600 Hz exactly |
| `Signal.x_values.start_value` | 0.4–38 µs | sub-sample start offset of the time axis |
| `Signal.y_values.quantity.label` | `'g'` | display unit label |
| `Signal.y_values.quantity.unit_transformation.factor` | 0.101971621 (= 1/9.80665) | display = factor × stored |
| `Signal.y_values.quantity.info` | `'Data saved by Test.Lab in MKS'` | stored values are in SI base unit |
| `Signal.function_record.primary_channel.label` | `Point1..Point4` | channel identifiers |
| `...TL_export_properties_annotation.run_name` | e.g. `210625_0Nm_Inner_0.3mm` | operator run label (date prefix is NOT reliable) |
| `...TL_export_properties_annotation.absolute_time` | e.g. `2021-06-28 11:32:36 ms 799.212` | acquisition start, Test.Lab PC clock (UTC; local creation_time = +9 h) |

**Units.** The paper states the data are in g. The file metadata (MKS storage + a 1/9.80665 display factor to `g`)
indicates the stored numbers are **m/s²**. See `DATA_AUDIT_REPORT.md` §3.4. The audit stores raw values unchanged and
labels them "stored unit". Any absolute-amplitude claim must state which convention is used.

**Channel mapping (ASSUMED).** Files name channels `Point1..Point4`; the paper lists columns
`x_direction_housing_A, y_direction_housing_A, x_direction_housing_B, y_direction_housing_B`. We map
Point1→x_A, Point2→y_A, Point3→x_B, Point4→y_B by order. The fault-seeded bearing is in housing A (PAPER).

## 3. Temperature / current `.tdms` schema (VERIFIED)

Group `Log`, waveform channels with `wf_increment = 3.905e-05 s` (nominal 25 608.2 Hz, *not* exactly 25 600 Hz) and absolute `wf_start_time` (NI cDAQ clock, UTC).

| Raw channel | Canonical name | Device | Unit | Notes |
|---|---|---|---|---|
| `cDAQ9185-1F486B5Mod1/ai0` | `T_housing_A` (ASSUMED) | NI 9210 (miniTC), K-type, built-in CJC | °C | sample-and-hold: new value every 5 377 samples ≈ 0.21 s (**effective 4.76 Hz**) |
| `cDAQ9185-1F486B5Mod1/ai1` | `T_housing_B` (ASSUMED) | NI 9210 | °C | updates at the same instants as ai0 |
| `cDAQ9185-1F486B5Mod2/ai0` | `I_U` (ASSUMED) | NI 9775 | A | present in all files |
| `cDAQ9185-1F486B5Mod2/ai2`, `ai3` | `I_V`, `I_W` (ASSUMED) | NI 9775 | A | **empty (0 samples) in all 9 BPFO files** |

File property `name` (e.g. `LogFile_0Nm_Inner_1mm`) and `RunNumber` are FlexLogger run metadata and were used for
label cross-checking. The paper's DAQ-module statements are internally contradictory (NI 9211 vs NI 9775 for temperature);
the files show NI 9210 for temperature and NI 9775 for current.

## 4. Derived manifests (`03_MANIFESTS/`)

| File | One row per | Key columns |
|---|---|---|
| `file_inventory.csv` | raw file (98) | relative_path, size_bytes, sha256, crc32, archive_crc32, crc_match_archive |
| `provenance.json` | — | archive hashes, unextracted entries, duplicate-content groups, environment |
| `recordings.csv` | recording = condition × load (45) | recording_id (`L<load>_<condition_id>`), label7, label3, in_bearing_scope, vib path+sha256, fs, duration, start/end, measured shaft speed, resolved TDMS path+sha256, pairing_status, temp_concurrent, temp effective rate, session, acquisition order |
| `pairs.csv` | vibration recording (45) | filename-matched TDMS, internal labels, raw offset, session clock offset, residual, nearest-in-time TDMS, best speed-match TDMS, resolved TDMS, pairing_status, speed mismatch, VFD supply frequency |
| `splits/split_*.json` | split fold | assignments (recording_id, role, start_s, end_s, label, hashes), notes, content_sha256 |
| `splits/split_index.csv` | split fold | protocol, counts, seconds per role, content hash, leakage check |
| `splits/challenge_set_out_of_scope.csv` | out-of-scope recording (24) | misalignment / unbalance recordings, never for tuning |
| `splits/lolo_class_seconds.csv` | fold × role | seconds of signal per class |

**Labels.** `label7 ∈ {H, IR_0p3, IR_1p0, IR_3p0, OR_0p3, OR_1p0, OR_3p0}`; `label3 ∈ {H, IR, OR}`;
misalignment/unbalance have no label7/label3 (`in_bearing_scope = False`).

**pairing_status.** `CONCURRENT_BY_NAME` — filename-matched TDMS starts within 5 s of the vibration after per-session
clock-offset correction and its internal label agrees. `FILENAME_SWAP_RESOLVED` — the filename-matched TDMS is from
another run; the TDMS that is concurrent in time *and* label-consistent is used instead. `NON_CONCURRENT` — no TDMS log
overlaps the vibration recording; the filename-matched log is kept as condition-level context only.

## 5. Audit tables (`results/audit/tables/`)

| File | Content |
|---|---|
| `signal_quality.csv` | per recording × channel: n, NaN/Inf, mean, std, RMS, min/max, peak, p2p, crest factor, kurtosis (Gaussian = 3), skewness, longest constant run, quantisation step, clipping indicators, 1-s segment RMS CV/drift, segment kurtosis median/IQR |
| `temperature_per_file.csv` | per TDMS × channel: update period, effective rate, min/max/mean/std, first/last 5 s means, rise, slope (°C/min), curvature, residual noise, A−B |
| `temperature_per_recording.csv` | per recording (resolved pairing): TA/TB means, start values, slopes, A−B, acquisition time |
| `physics_evidence_record_level.csv` | per recording × channel × band × target: SNR, peak frequency ratio, control-frequency median/q95/max, excess dB, robust z, fraction of controls ≥ target, tolerance sensitivity |
| `physics_evidence_window_level.csv` | per recording × channel × band × target: fraction of 1-s windows with evidence, excess quantiles |
| `physics_calls_record_level.csv` | per recording: label-free physics call (none / OR-like / IR-like / both) vs expectation |
| `frequency_collisions.csv` | proximity of BPFO/BPFI harmonics to shaft harmonics and to the other family's components |
| `shaft_speed_and_fault_frequencies.csv` | measured shaft frequency and BPFO/BPFI at measured vs paper speed |
| `duration_by_condition_load.csv` | seconds per condition × load |

## 6. Physical constants (PAPER; re-derived in `pave/signal/physics.py`)

NSK 6205 DDU: N = 9 balls, d = 7.90 mm, D = 38.5 mm, contact angle 0°. Order multipliers (×fr):
FTF 0.3974, BPFO 3.5766, BPFI 5.4234, BSF 2.3342 (standard). The paper's BSF (234.19 Hz) equals 2×BSF.
The paper's BPFI/BPFO values correspond to fr = 3010/60 = 50.1667 Hz. Measured shaft speed differs by load
(see `shaft_speed_and_fault_frequencies.csv`); all evidence uses the **measured** per-recording speed.
