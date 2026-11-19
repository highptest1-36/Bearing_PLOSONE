# PAVE-Net — Data feasibility assessment and scientific decision

Basis: [DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md) and sub-reports in `docs/audit/` (2026-11-18). No model has been
trained. Ratings judge whether *this dataset* can support each claim as it would be written in a PLOS ONE paper.

## Decision summary

| Objective | Rating | One-line reason |
|---|---|---|
| A. Vibration-only physics-informed diagnosis | **PASS** (3-way) / **CONCERN** (7-way) | Strong, load-robust race-defect evidence in every fault window; severity is confounded with installation |
| B. Vibration–temperature multimodal diagnosis | **FAIL** | Temperature is slow, warming, non-concurrent for OR 0.3 mm, and fully confounded with day/warm-up/installation |
| C. Physics-evidence concept learning | **CONCERN** | Concepts are measurable, but only with measured speed and data-derived fault orders; they nearly equal the 3-way label |
| D. Physics-based explanation verification | **CONCERN** | Sharp lines make targeted interventions feasible; harmonic collisions and installation-specific shifts require care |
| E. Cross-load reliability assessment | **CONCERN** | Only 3 loads × 1 recording per condition, same installation across loads, Healthy speed shortcut |

**Recommendation: MODIFY** (details in §6). The original architecture's central claim — vibration–temperature fusion
— is not supported. The physics-audit, concept and selective-diagnosis ideas remain viable on vibration alone if
claims are narrowed.

---

## A. Vibration-only physics-informed diagnosis — PASS (3-way H/IR/OR), CONCERN (7-way)

**Supports:**
- Clean data: no NaN/Inf, clipping or dropouts; 25.6 kHz verified.
- A label-free family rule finds the expected IR/OR family in 100 % of 1-s windows of all 18 fault recordings, with
  ~1 % false calls on healthy and out-of-scope windows.
- Thresholds fit on two source loads transfer to the third.
- BPFO_obs + BPFI_obs = 9.007 ≈ N, a kinematic consistency check that does not depend on labels.

**Concerns:**
- Kinematic constants alone fail (8/21 agreement). The geometry must be supplemented with measured speed and
  effective fault orders (−1.2 to −1.6 % IR, +0.9 to +2.5 % OR).
- H vs faults is also separable by RMS (5–17×), by day, and at 2/4 Nm by shaft speed, so physics is not the only route
  to high accuracy.
- 7-way severity: evidence and RMS are not monotonic in defect size, and each severity has its own line position, RMS
  level and A−B temperature. A 7-way classifier can succeed by recognising installations; "severity generalisation"
  cannot be claimed.

## B. Vibration–temperature multimodal diagnosis — FAIL

- Temperature is a 4.76 Hz sample-and-hold signal (0.01 °C steps) from housing thermocouples, warming at
  0.17–1.13 °C/min throughout every recording.
- Synchronisation is inferred from a per-day clock offset (σ 0.1–0.9 s, single runs up to 2.8 s) and cannot be verified
  physically. Two TDMS files are swapped (resolved), and OR 0.3 mm temperature is ≈ 32 min non-concurrent at all loads.
- Class ≡ recording day; load ≡ warm-up order; A−B ≡ installation (η² = 0.85 for the 7-way label, 0.05 for fault type).
  A parameter-free temperature 1-NN already identifies conditions above chance under LOLO.
- Consequently, **any** temperature-driven accuracy gain would be uninterpretable: it cannot be separated from session
  and installation shortcuts with the available data (no repeated sessions, no re-mounting, no counterbalanced load
  order). Window-level fusion is physically meaningless at this temporal resolution.
- Blueprint gate 3 applies: stop making multimodal claims.

## C. Physics-evidence concept learning — CONCERN

**Supports:**
- Fault-family evidence, harmonic support, IR shaft sidebands, envelope impulsiveness, RMS/kurtosis and signal quality
  are computable per 1-s window, label-free, and stable across loads.

**Concerns:**
1. Concept definitions at kinematic frequencies are wrong for this rig. Effective orders must be estimated on source
   folds only, or a widened search window frozen before testing (the ±3.5 % rule here was chosen post hoc).
2. 3rd-harmonic concepts are ambiguous (2×BPFI_obs ≈ 3×BPFO_kin); single-band concepts need family contrast.
3. Concepts are nearly deterministic for H/IR/OR (≈ 100 % / ~1 % FP). A bottleneck will be accurate but may be close to
   a rule; concept-intervention results risk being trivial. Severity is not encoded by any physical concept.
4. Concept targets and the classifier read the same vibration signal (no independent sensor), so concept "fidelity" is
   self-consistency, not ground truth.
5. Temperature concepts are excluded (see B).

## D. Physics-based explanation verification — CONCERN

**Supports:**
- Fault lines are narrow (IQR ≤ 0.002 orders for most conditions) and 30–50 dB above the floor. Band-limited raw-signal
  interventions with matched-energy control bands are technically feasible, and control frequencies avoiding shaft
  harmonics can be defined (24 used here).

**Concerns:**
- The intervention bands must target the observed (installation-specific) lines plus IR sidebands, not kinematic values.
- Removing a line that is also an installation fingerprint cannot distinguish "uses fault physics" from "uses specimen
  identity".
- Harmonic collisions complicate wrong-class control bands.
- Severity explanations have no physical ground truth.

## E. Cross-load reliability assessment — CONCERN

**Supports:**
- Leakage-safe LOLO / permutation / within-recording splits are pre-registered, hashed and unit-tested. Source-only
  physics thresholds transfer across loads.

**Concerns:**
- 3 load domains, 7 test recordings per fold (record-level metrics in 14 % steps), no bearing replication.
- All loads of a condition come from one installation recorded within ~10 min, so LOLO measures load transfer within a
  specimen, not reliability on new bearings.
- Healthy speed differs from all fault runs at 2/4 Nm.
- With near-perfect 3-way physics separability, the risk–coverage analysis (PE-SD) may have very few errors to defer in
  3-way; in 7-way, errors will be dominated by installation confounds. Thresholds tuned on same-recording validation
  are not independent.

---

## 6. Recommendation: MODIFY

Proceeding with the original PAVE-Net (vibration–temperature fusion as headline) is **not** recommended; a NO-GO is not
warranted because the vibration physics evidence is genuinely strong. Proposed modifications, each requiring the
author's decision:

1. **Drop temperature from the primary architecture and title.** Keep it only as (i) a temperature-only / concat
   ablation reported as confounded, and (ii) an explicit shortcut analysis (A−B, day, warm-up). Retitle along the lines
   of "Physics-audited, concept-based and selective vibration diagnosis …".
2. **Primary task 3-way H/IR/OR; 7-way severity as secondary**, with the installation confound stated in the abstract and
   limitations. Do not claim severity generalisation.
3. **Physics concepts from measured speed + effective fault orders**, estimated per LOLO fold from source data only (or
   the frozen ±3.5 % family search), using family contrast and fundamentals/2nd harmonics. Freeze definitions before
   any test-fold inspection and record the post-hoc origin of the widened window.
4. **Neutralise the speed shortcut:** order-domain (angular) resampling with the measured speed, plus a speed-only
   control model to show the shortcut's size.
5. **Make confounds part of the reliability study:** report a shortcut-control set (temperature-only, speed-only,
   RMS-only baselines) beside PAVE-Net; evaluate PE-SD on 7-way and on noise-corrupted 3-way where errors exist.
6. **External validity:** across-bearing claims require a second dataset with independent specimens (e.g. Paderborn
   with multiple bearings per damage type), vibration only, geometry-specific concepts. Without it, restrict claims to
   load transfer on one rig.
7. Report the OR 0.3 mm non-concurrency, the TDMS swap, the unit ambiguity and the assumed channel mappings in Methods.

**Questions for the author before Phase 2:**
(a) Accept dropping the thermal branch from the main claim and retitling?
(b) 3-way primary with 7-way secondary, or keep 7-way primary with confound disclosure?
(c) Add an external multi-specimen vibration dataset (which one)?
(d) Contact the KAIST authors to confirm clock synchronisation, units, channel mapping and bearing re-use?
