# Dataset integrity and inventory report

Generated from `03_MANIFESTS/file_inventory.csv`, `03_MANIFESTS/provenance.json`, `03_MANIFESTS/recordings.csv`
(`python scripts/audit_data.py --stage inventory|analyze`, 2026-11-18). Raw data were opened read-only.

## 1. What is on disk (VERIFIED)

| Item | Count | Size | Notes |
|---|---|---|---|
| `vibration/*.mat` | 45 | 3.69 GB | 15 conditions × 3 loads |
| `current,temp/*.tdms` | 45 | 4.39 GB | 15 conditions × 3 loads |
| `acoustic/*.mat` | 5 | 0.12 GB | 0 Nm only: H, IR 0.3/1.0, OR 0.3/1.0 (matches paper Table 2; not used) |
| `*.zip` archives | 3 | 4.25 GB | vibration (45 entries), current,temp (45), acoustic (5) |
| **Total** | **98 files** | | |

- **Archive provenance:** every extracted file's CRC32 and byte size equal the corresponding zip entry; no archive entry
  is missing from the extracted folders; no extracted file is absent from its archive. Archive SHA-256:
  vibration.zip `df73fa80…4659f`, current,temp.zip `7caa2215…d6`, acoustic.zip `5dd13719…62f` (full hashes in `provenance.json`).
- **Duplicates:** no two data files share content (SHA-256).
- **Corruption:** all 90 vibration/TDMS files parsed; no NaN or Inf in any vibration or temperature channel;
  no constant or clipped vibration channel (§3 of the signal-quality report).
- **Dataset version:** the local copy contains no version file. The blueprint states Mendeley V6
  (doi:10.17632/ztmf3m7h5x.6); this is recorded as an *unverified* attribution in `recordings.csv`.
- **Acquisition dates** (zip entry timestamps and in-file clocks): 25 Jun 2021 (H, misalignment, unbalance),
  28 Jun 2021 (all IR), 30 Jun 2021 (all OR); acoustic exported 6 Jul 2021.

## 2. Conditions, loads and independent units

| Group | Severities | Loads | Recordings | Duration each |
|---|---|---|---|---|
| Healthy | — | 0, 2, 4 Nm | 3 | 300 s (0 Nm), 120 s (2, 4 Nm) |
| Inner race (IR) | 0.3, 1.0, 3.0 mm | 0, 2, 4 Nm | 9 | 60 s |
| Outer race (OR) | 0.3, 1.0, 3.0 mm | 0, 2, 4 Nm | 9 | 60 s |
| Misalignment (out of scope) | 0.1, 0.3, 0.5 mm | 0, 2, 4 Nm | 9 | 120 s |
| Unbalance (out of scope) | 583, 1169, 1751, 2239, 3318 mg | 0, 2, 4 Nm | 15 | 120 s |

- **Bearing task:** 21 recordings = 7 conditions × 3 loads, exactly one recording per cell (expected and verified).
  The paper states 120 s healthy / 60 s faulty; the 0 Nm healthy file is actually **300 s** (VERIFIED).
- **Independent physical units:** the paper does not say whether bearings were re-used. The acquisition timeline
  shows each fault severity recorded as one contiguous block of its three loads (≈ 7–14 min), with a pause between
  severities (temperature resets, see temperature report), consistent with **one bearing installation per severity**.
  The effective number of independent fault specimens is therefore at most 6 (+1 healthy), and each is observed only
  under one mounting. There is **no bearing-level replication**: no condition has two independent specimens.
- Windows cut from these recordings are correlated sub-samples and are not independent observations.

## 3. Naming and labelling anomalies (VERIFIED)

| Issue | Files | Impact | Handling |
|---|---|---|---|
| Typo `Unbalalnce` | 5 vibration files at 2 Nm | none if parser tolerant | parser alias; recording_id normalised |
| **TDMS filename swap** | `0Nm_BPFI_03.tdms` contains the 1.0 mm run (`LogFile_0Nm_Inner_1mm`) and `0Nm_BPFI_10.tdms` contains the 0.3 mm run | temperature/current of two IR recordings mislabelled | resolved in `pairs.csv` on three independent lines of evidence (pairing report) |
| OR 0.3 mm TDMS not concurrent | `{0,2,4}Nm_BPFO_03.tdms` | temperature/current recorded ≈ 32 min after the vibration | flagged `temp_concurrent = False` |
| Missing current phases | all 9 `*BPFO*.tdms`: channels ai2, ai3 have 0 samples | only one current phase for OR | documented; current not used in PAVE-Net |
| Unreliable run-name dates | e.g. 2 Nm IR 0.3 mm run_name `210625_…` recorded 28 Jun | none | timestamps used instead |
| Internal unbalance masses | FlexLogger names `Unbalance_0.583`, `0.1169`; vibration `Unbal_584mg` vs filename 583 mg | label check needs tolerance | 1 % severity tolerance; parsed to mg |

## 4. Paper-vs-file discrepancies

| Topic | Paper | Files | Status |
|---|---|---|---|
| Vibration channel names | `x_direction_housing_A` … | `Point1..Point4` | mapping ASSUMED by order |
| Vibration unit | g | Test.Lab "saved in MKS" + factor 1/9.80665 to the `g` label ⇒ stored m/s² | **unresolved**; m/s² reading also gives the more plausible healthy level (0.10 g RMS vs 1.0 g RMS) |
| Temperature module | NI 9211 or NI 9775 (paper contradicts itself) | NI 9210 (miniTC), K-type | file metadata used |
| Temperature sampling | 25.6 kHz | logged at 25 608.2 Hz, **sample-and-hold, new value every 0.21 s (4.76 Hz)** | file metadata + data used |
| Healthy duration | 120 s | 300 s at 0 Nm | file used |
| Shaft speed | 3010 rpm at all loads | 49.88–50.19 Hz measured; depends on load **and** on whether the VFD was re-tuned (see physics report) | measured speed used |
| BPFI/BPFO values | 272.07 / 179.43 Hz for fr = 50.17 Hz | values reproduce with fr = 3010/60 = 50.1667 Hz exactly (50.17 Hz gives 272.09) | unit-tested |
| BSF | 234.19 Hz | equals 2 × standard BSF (117.10 Hz) | documented; ball faults not in scope |
| Synchronisation | "synchronized" speed data mentioned; no method | two independent clocks, offset 198–207 s | see pairing report |
