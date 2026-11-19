# Temperature feasibility analysis

Sources: `results/audit/tables/temperature_per_file.csv`, `temperature_per_recording.csv`,
`temperature_shortcut_probe_1nn.csv`, `audit_summary.json` (`temp_eta2`, `temp_vs_time`);
figures `fig09_temperature_traces_T_housing_{A,B}.png`, `fig10_temperature_by_class_load.png`.

## 1. What the temperature channels actually are (VERIFIED)

- Two K-type thermocouples on an NI 9210 (miniTC) module, built-in cold-junction compensation, logged by FlexLogger.
- Logged at 25 608.2 Hz but **sample-and-hold**: a new value every 5 377 logged samples = 0.21 s ⇒ **4.76 Hz effective**,
  287 distinct readings per 60-s file. Both channels update at the same instants.
- Step between readings ≈ 0.010 °C (median); residual noise after a quadratic trend 0.010–0.067 °C.
- No NaN/Inf; plausible range 25.0–33.4 °C across the 45 files (in-scope record means 26.3–32.7 °C housing A).
- The thermocouple sits on the housing, so the bearing-to-sensor thermal path has a time constant of seconds to
  minutes (not measured here). The 4.76 Hz rate therefore over-samples the physically meaningful bandwidth.

## 2. Temperature is not in steady state

Every recording is still warming: slope 0.17–1.13 °C/min (housing A) and 0.24–1.06 °C/min (housing B) across the
21 bearing recordings; rise within a file −0.2 to +2.1 °C. The rise within one 60-s recording (≈ 0.2–1.1 °C) is of the
same size as many between-class differences. In a source-only temporal validation (last 20 % of each recording)
the validation temperatures are systematically higher than training temperatures.

## 3. Temperature is dominated by acquisition history

- **Class = session:** Healthy was recorded only on 25 Jun, IR only on 28 Jun, OR only on 30 Jun. Any H/IR/OR
  temperature contrast is indistinguishable from day, ambient temperature, machine history and bearing installation.
- **Load = order within block:** each severity was recorded as a 0 → 2 → 4 Nm block a few minutes apart while the
  rig warmed. Housing-A start temperature (°C):

  | Condition | 0 Nm | 2 Nm | 4 Nm |
  |---|---|---|---|
  | Healthy | 25.06 | 27.83 | 27.06 |
  | IR 0.3 | 26.93 | 27.44 | 28.58 |
  | IR 1.0 | 27.65 | 28.45 | 30.07 |
  | IR 3.0 | 27.00 | 29.38 | 30.81 |
  | OR 0.3 | 29.02* | 31.40* | 32.56* |
  | OR 1.0 | 28.70 | 29.52 | 31.44 |
  | OR 3.0 | 27.60 | 30.31 | 31.60 |

  (*not concurrent with the vibration.) The apparent "load effect" coincides with elapsed time; on 25 Jun the start
  temperature rises monotonically with time of acquisition (Spearman ρ = 0.95, n = 27).
- Descriptive variance shares (η², 21 bearing recordings): housing-A mean — label7 0.55, fault group 0.45, load 0.38,
  session 0.45 (fault group and session are the same partition). These cannot be separated.

## 4. Housing A − B difference is an installation fingerprint

A − B is nearly constant within each condition across its three loads (e.g. IR 0.3: −1.55 to −1.47 °C; IR 3.0:
+0.24 to +0.97 °C; OR 3.0: −1.65 to −0.69 °C) and differs between conditions: η²(label7) = 0.85, but η²(fault group)
= 0.05 and η²(load) = 0.06. It identifies *which installation/block* a recording came from, not the fault type.
Plausible causes include thermocouple contact/mounting changes between bearing swaps. No physical model links
A − B to defect size.

## 5. Shortcut probe (diagnostic, not a model result)

Parameter-free record-level 1-NN with source-only standardisation, leave-one-load-out, n = 7 test recordings per fold:

| Features | 7-way acc (0/2/4 Nm held out) | 3-way acc |
|---|---|---|
| A − B | 0.14 / 0.29 / 0.14 | 0.43 / 0.57 / 0.57 |
| TA, TB | 0.29 / 0.14 / 0.14 | 0.29 / 0.29 / 0.43 |
| TA, TB, A − B, slopes | 0.57 / 0.43 / 0.43 | 0.71 / 0.71 / 0.43 |

Chance 7-way = 0.14; majority 3-way = 0.43. With 7 test recordings per fold, one recording = 0.14, so these
numbers are extremely noisy. They show temperature can partially *identify* conditions; because of §3–4, that
information is attributable to session/installation/warm-up confounds rather than to bearing-fault physics.

## 6. Does temperature add information beyond operating-load differences?

It adds information, but of the wrong kind. Temperature differences between classes are fully confounded with
recording day, warm-up time and bearing installation, and the data contain no repetition that could separate them
(no class recorded on two days, no load order varied, no specimen re-mounted). Nothing in this dataset allows a
fault-related thermal effect to be identified. The 30–60 s seeded-defect runs are also far shorter than the thermal
time constants over which friction heating from a local defect would be expected to appear.

## 7. Verdict for multimodal use

- Window-level fusion: **not justified** (0.21 s hold, ~0.01 °C per window, clock-offset-inferred alignment, OR 0.3 mm
  non-concurrent).
- Recording-level thermal context: technically available for 18/21 bearing recordings, but any accuracy gain from it
  would be indistinguishable from a shortcut. If kept, it should only appear as (a) an ablation reported as a negative
  or confounded result and (b) an explicit shortcut demonstration (e.g. A − B alone, temperature-only model).
- Blueprint gate 3 ("temperature improves at least one objective beyond noise, else switch to vibration-only") cannot
  be satisfied *validly* on this dataset even if a gain appears, because a gain could not be attributed to physics.
