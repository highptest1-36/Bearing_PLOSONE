# PAVE-Net — Research Protocol V2.2 (Phase 3B development + external validation plan)

**Amendment of V2 (`60d0c66a…`) and V2.1 (`15343677…`); both remain unchanged and valid for Phases 2 and 3A.**
Version 2.2, written and frozen on 2026-11-19 **before** any Phase 3B model, concept-V3 computation on real data,
intervention or selective-diagnosis experiment. SHA-256 recorded in `docs/PROTOCOL_FREEZE_V2.2.txt`.
Results already inspected at freeze time: all Phase 1–3A outputs (KAIST, Paderborn PU-BD5/A2R), the Ottawa dataset
paper and the Ottawa file listing/variable schema (no Ottawa signal, speed or label-dependent statistic was computed).

## 1. Motivation of the amendment
| Change | Reason (evidence) |
|---|---|
| Physics concepts V3 (drop `BPFO_sb`, identifiability indicators, validity masks, 4-s variant) | Phase 3A: `BPFO_sb` reversed (bearing AUROC 0.24); BPFI undefined in 97.5 % of FAG 900-rpm windows (shaft-harmonic collision); NaN imputation hid missing evidence |
| Fixed, pre-declared hyper-parameters; no nested selection | Phase 3A: V2.1 §6 fallback triggered (inner-fold spread 0.18–0.64 for the same configuration) |
| Inner loop retained only for calibration, PE-SD fitting, architecture-inclusion decision | only use of development data that does not touch outer-test bearings |
| Candidate B (physics-anchored PAVE-Net) prioritised; Candidate A (concept bottleneck) as comparator | Phase 3A: concept models ≥ CNN; CNN over-confident (T up to 20) and insensitive to fault-line removal |
| External validation on University of Ottawa UORED-VAFCLS | Paderborn has been inspected; a locked external dataset is required for any confirmatory claim |

## 2. Research questions and hypotheses
- **RQ1** Physics concepts give more stable bearing-level evidence than unrestricted learned features.
  *H1:* B1-V3 bearing macro-F1 ≥ B3 on PU-BD5; concepts are less predictive of nuisance metadata than CNN embeddings.
- **RQ2** Evidence-reliability-aware fusion helps when fault signatures are weak.
  *H2:* PAVE-Net (A4) improves weak-evidence bearings (KA15, KA22, KA30, KI14, KI16, KI21) without degrading
  strong-evidence bearings, relative to the physics-only model (B4/A2).
- **RQ3** Controlled interventions verify dependence on physical evidence.
  *H3:* targeted fault-family suppression changes p(true class) more than matched controls for physics-based models and
  for PAVE-Net; the CNN shows weaker targeted-minus-control effects.
- **RQ4** Physics-aware selective diagnosis lowers accepted-decision error at matched coverage.
  *H4:* S4 has lower bearing-level selective risk than S1/S2 at coverage 12/15 and 9/15 and lower AURC.
- **RQ5** Improvements transfer to an independent dataset without target-label adaptation (Ottawa).
All hypotheses may be rejected; negative results are reported.

## 3. Tasks, units and data
- **Primary task:** 3-class Healthy (H) / Inner race (IR) / Outer race (OR).
- **Experimental unit:** the physical bearing (PU-BD5: 15 bearings). Windows/measurements are descriptive only.
- **Development data:** PU-BD5 (Lessmeier 2016 Table 10 rows R1–R5), outer folds O1–O5 exactly as V2.1 §4.
  Same exclusions as V2.1 §5. KAIST is not used in Phase 3B.
- **Additional development bearings for probes only:** the 12 artificial bearings, K006, KI17 (never trained on).
- **External data:** Ottawa UORED-VAFCLS (user-stated version 5, doi 10.17632/y2px5tg92h.5); governed by a separate
  external protocol frozen after the Phase 3B gate (§12).

## 4. Physics concepts V3
Inputs: raw vibration window x (duration T), sampling rate fs, measured shaft frequency fr, speed uncertainty σ_fr
(within-window std of the speed channel, Hz; 0 if unavailable), bearing geometry (N, d, D, θ).
Envelope spectrum E(f): band-pass (Butterworth order 4, zero-phase; PU/Ottawa 1–20 kHz) → |Hilbert| − mean → Hann
amplitude spectrum, resolution Δf = 1/T (T = 1 s or 4 s; 0–1000 Hz kept).
- Kinematic order o_F (F ∈ {BPFO, BPFI}); training-calibrated relative centre r_F and half-width w_F exactly as V2
  (`calibrate_regions`, train bearings only, but with the V3 shaft guard).
- Region for harmonic h: centre c_{F,h} = h·o_F·fr·(1 + r_F); half-width hw_{F,h} = max(h·w_F·o_F·fr, Δf) + 3·h·o_F·σ_fr.
- **Shaft guard** around k·fr: g_k = Δf + 3·k·σ_fr (V2 used a fixed ±1 Hz).
- **Identifiability distance (bins):** δ_F = (min_k |c_{F,1} − k·fr| − hw_{F,1} − g_k)/Δf; **valid_F = 1 if δ_F ≥ 1**.
- **Harmonic-level exclusion:** harmonic h of family F is excluded if its region overlaps a shaft guard or a region
  (h' = 1..3, plus ±fr sidebands of BPFI) of the other family; `overlap_F` = number of excluded harmonics.
- SNR(c, hw) = 20·log10(max E in [c ± hw] / median E in a ±20 Hz ring), ring excluding shaft guards, the other
  region and ±2 bins around the peak.
- **Concepts (8):** `BPFO_snr1`, `BPFO_harm` (mean SNR over valid harmonics), `BPFO_nharm` (#valid harmonics ≥ 6 dB),
  `BPFI_snr1`, `BPFI_harm`, `BPFI_nharm`, `BPFI_sb` (mean SNR at f0 ± fr, sidebands outside guards), `contrast`
  = `BPFO_harm` − `BPFI_harm`.
- **Masks (8):** family concepts masked by valid_F (and finite); `BPFI_sb` additionally by sideband validity;
  `contrast` by valid_BPFO ∧ valid_BPFI.
- **Deterministic representation:** z_k = m_k·(c_k − μ_k)/σ_k with μ, σ from valid TRAINING entries; masked → 0;
  model input = [z, m] (16 values). NaN never reaches a model.
- **Reliability features q (7, label-free):** clip(δ_BPFO, −5, 20), clip(δ_BPFI, −5, 20), mean(m), f_max = max of valid
  family `harm` values (0 if none), |contrast|·m_contrast, max(`nharm`), total excluded harmonics.
- **V2 concepts** (`pave/features/physics.py`) are kept unchanged for B1-V2 and ablations.
- **4-s variant:** the same definitions on each full measurement (first 4 s; Δf = 0.25 Hz). PU measurement = one file.

## 5. Models (all fixed; no hyper-parameter search)
Common: AdamW, cosine schedule over the stated epochs, class-weighted cross-entropy, mixed precision for CNNs, seeds
42/43/44 set **before** model construction; primary prediction = mean probability of the 3 seeds; per-seed reported.

| ID | Model | Input | Configuration |
|---|---|---|---|
| B0 | physics rule V3 | q, contrast | faulty if f_max ≥ τ (τ maximises train H-vs-faulty balanced accuracy); class OR if `contrast`>0 when valid, else the larger valid family; IR otherwise |
| B1-V2 | P1 (Phase 2/3A) | 9 V2 concepts (median-imputed) | StandardScaler + LR, C = 1, balanced |
| B1-V3-1s | LR | [z, m] 1-s | LR, C = 1, balanced |
| B1-V3-4s | LR | [z, m] 4-s measurement | LR, C = 1, balanced |
| B2-V3 | MLP | [z, m] 1-s | 2×64 ReLU, dropout 0.1, lr 1e-3, wd 1e-2, batch 256, 20 epochs |
| B3 | CNN-1D raw | 1-s, 32 kHz, z-scored (train) | Phase 2 architecture (≈0.23 M), lr 1e-3, wd 1e-2, batch 64, 25 epochs |
| B4 (=A1) | physics head, no latent | [z, m] | linear head, AdamW lr 1e-3, wd 1e-2, batch 64, 25 epochs |
| A2 | physics head + reliability | [z, m, q] | as B4 |
| A3 | physics + bounded latent, no gate | [z, m] + raw | g ≡ 0 below |
| **A4 PAVE-Net** | physics-anchored fusion | [z, m], q, raw | full model below |
| A5 | A4 + PE-SD | — | §8 |
| A6 Candidate A (PECB) | CNN → ĉ → linear | raw | λ_c = 1, below |

**PAVE-Net (A4).** ℓ_phys = W_p[z, m] + b_p; h = E_v(x) (CNN-1D of B3 without its head, 128-d); ℓ_lat = W_l h;
g = σ(uᵀ q̃ + b_u) with q̃ the train-standardised q; **ℓ = ℓ_phys + γ (1 − g) · B · tanh(ℓ_lat / B)**, γ = 1, B = 3
(each latent logit bounded in ±3). Loss: CE(ℓ, y) + β·mean(1 − g), β = 0.05 (prior favouring physics). Physics-only
mode: γ = 0. A3: same with g ≡ 0 (latent always on, still bounded).
**Candidate A.** ĉ = W_c h (8 predicted standardised concepts); ℓ = W_y ĉ + b_y (classifier sees ĉ only);
loss CE + λ_c · Σ_k m_k Huber(ĉ_k − z_k)/Σ_k m_k, λ_c = 1.

Hyper-parameter justification: B3/CNN values are the Phase 2 defaults (Phase 3A default B3 ≥ nested B3); B2 width/wd
selected 5/5 times in Phase 3A inner loops (epochs median 15 → 20); LR C = 1 gave identical bearing results to nested C.

## 6. Training, calibration and resources
- Outer fold Ok: train on the 12 outer-train bearings, predict the 3 outer-test bearings.
- **Inner loop (seed 42 only)**: for each outer fold, 4 inner folds (V2.1) produce out-of-fold (OOF) probabilities for
  all probabilistic models. Used ONLY for: temperature scaling, fitting S4 (§8), the architecture-inclusion rule (§9).
- Temperature T > 0 per model per outer fold, NLL-minimised on pooled inner OOF windows.
- Checkpoints `results/phase3b/checkpoints/<model>/<fold>/s<seed>.pt` with SHA-256 manifest; existing files are never
  overwritten (resume = skip). Logged: config, seed, epochs, durations, peak GPU memory, library/CUDA/GPU versions.
- Budget ≤ 4 GPU-hours on RTX 3060 Laptop 6 GB.

## 7. Physics-evidence verification (PEV)
Envelope-domain interventions (`pave/explain/interventions.py`, Phase 3A): targeted suppression of the fold-calibrated
family regions (h = 1–3; ±fr sidebands for BPFI; guards and other-family overlaps removed), matched-width control,
matched-energy control; strengths 1.0 and 0.5. Sample: 5 measurements × 4 settings × 15 bearings (first window).
Models: B1-V3-1s, A2, A6, A4 (physics and latent logits separately), B3. Outcomes per bearing: Δp(true), Δlogit(true),
prediction flips, target/other-family concept change, distortion (in-band SNR, clipping, out-of-band change, energy ratio).
Primary contrast: targeted − matched-energy control on Δp(true) for the bearing's own family (faulty bearings, n = 10),
bootstrap 95 % CI over bearings. Terminology: *dependence* (model output changes under intervention) ≠ *faithfulness*
(explanation names the evidence the model depends on) ≠ *physical plausibility* ≠ *causality* (not claimed).

## 8. Physics-evidence-aware selective diagnosis (PE-SD)
Window scores aggregated by bearing mean. Higher score = more confident.
- **S1** max probability (after temperature scaling) of the bearing-mean probability.
- **S2** negative entropy of the bearing-mean probability.
- **S3** evidence clarity (prediction-independent): mean over windows of mean(m)·|f_max − τ|.
- **S4** physics-evidence-aware: logistic regression (C = 0.1) on window features [S1_window, κ, mean(m)], fitted on
  inner OOF windows to predict correctness; κ = evidence–prediction compatibility (dB): ŷ = H → τ − f_max;
  ŷ = OR → `contrast`; ŷ = IR → −`contrast` (if `contrast` masked: ± larger valid family `harm` − τ).
- Base classifiers: B1-V3-1s and A4 (and the final recommended model).
- Metrics: bearing-level risk–coverage, AURC, selective risk and accepted errors at coverage 12/15 and 9/15 (matched),
  error-detection AUROC, rejected bearings by class; window-level curves descriptive.

## 9. Decision rules
- **Architecture inclusion (development data only).** The latent branch is retained in the final recommended model only
  if, on pooled inner OOF (5 outer × 12 bearings = 60 bearing predictions), A4 improves bearing macro-F1 over A2 by
  ≥ 0.05 **and** window NLL is not worse. Otherwise the final recommended model is physics-only.
- **Physics-only final model:** B1-V3 with the context length (1 s or 4 s) having the higher pooled inner-OOF bearing
  macro-F1 (tie → 1 s).
- Outer-test results are reported for all models and never used for selection.
- **Phase 3B gate (GO):** all tests pass; no leakage assertion fails; all configurations fixed as above; final model and
  external evaluation fully specified. Otherwise STOP before external validation.

## 10. Statistics
Primary endpoint: bearing-level macro-F1 (15 bearings; 3-seed ensemble). Comparators: B1-V2 (author-designated
benchmark) and B1-V3-1s. Stratified bootstrap over bearings (10 000) for CIs; paired bootstrap differences; exact
McNemar on bearing correctness. No inference from windows. Secondary: balanced accuracy, per-class precision/recall,
fault sensitivity, healthy false-alarm rate, IR-vs-OR accuracy on faulty bearings, window NLL, ECE (15 bins), Brier,
AURC, intervention effects, latency (batch 1, GPU/CPU), parameters, peak memory.

## 11. Shortcut analyses
Embeddings (B3 penultimate, A4 latent h, A6 ĉ) and V3 concept vectors extracted from fold models for bearings outside
that fold's training set (outer-test bearings + 14 non-BD5 bearings). Probes: leave-one-bearing-out LR (C = 1) on
bearing-mean representations for manufacturer (IBU vs other), acquisition year, origin (artificial vs real), and
window-level probes for operating condition (4 settings) and speed (900 vs 1500 rpm). Report bearing-level AUROC.
Predictability ≠ use by the classifier. Stratum analysis: IBU (K001–K005 vs KA22).

## 12. External validation (planned; details frozen later)
After a GO gate, `docs/EXTERNAL_VALIDATION_PROTOCOL_V1.md` is frozen with checkpoint hashes, before any Ottawa
prediction. Final models = the predefined models retrained on all 15 PU-BD5 bearings (seeds 42–44) with physics regions
calibrated on all 15 bearings; temperature and S4 parameters fitted on pooled PU outer-fold test predictions.
Ottawa is evaluated once; no target-label adaptation.

## 13. Reporting
Per-bearing predictions, confusion matrices, per-fold and per-seed results, CIs, paired differences, negative results,
all deviations with timestamps in `docs/DECISION_LOG.md`.
