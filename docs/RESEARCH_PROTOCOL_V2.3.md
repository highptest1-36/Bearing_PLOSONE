# PAVE-Net — Protocol amendment V2.3

Amends V2.2 (`078d726b…7509`, unchanged). Frozen 2026-11-19 before any Phase 3B real-data computation.
SHA-256 in `docs/PROTOCOL_FREEZE_V2.3.txt`.

## What was inspected before this amendment
Only (i) the V3 synthetic unit tests (`tests/test_physics_v3.py`) and (ii) an analytic identifiability table computed
from PU bearing geometry (fact sheets) and nominal speeds, using the frozen V3 definitions with default region
half-width 0.5 % and zero speed uncertainty. No real vibration signal, concept value, label or model output of Phase 3B.

## Finding that motivates the amendment
Identifiability distance δ (bins) of the fundamental family region from the nearest guarded shaft harmonic:

| Geometry | Speed | BPFO 1 s | BPFO 4 s | BPFI 1 s | BPFI 4 s |
|---|---|---|---|---|---|
| FAG (D = 28.55 mm) | 900 rpm | −1.19 | +1.26 | −1.19 | **+0.77** |
| FAG | 1500 rpm | −0.64 | +2.90 | −0.64 | +1.96 |
| IBU/MTK (D = 29.05 mm) | 900 rpm | −0.94 | +2.23 | −0.94 | +1.76 |
| IBU/MTK | 1500 rpm | −0.24 | +4.52 | −0.24 | +3.59 |

For the 6203 geometry BPFO ≈ 3.05·fr and BPFI ≈ 4.93–4.95·fr lie 0.8–1.8 Hz from 3·fr and 5·fr. With Δf = 1 Hz (1-s
windows) no family is identifiable under the V3 criterion (δ ≥ 1) at any PU setting; with Δf = 0.25 Hz (4 s) all are,
except FAG BPFI at 900 rpm. Under V2.2 the physics branch of every 1-s fusion model would therefore be fully masked.

## Amendment
1. **Physics context = 4-s measurement.** For B0, B4/A1, A2, A3, A4 (PAVE-Net), A6 (concept targets), PE-SD (S3, S4, κ)
   and PEV, the physics concepts, masks and reliability features of a 1-s window are those of the 4-s measurement that
   contains it (V3, Δf = 0.25 Hz; regions calibrated on training measurements). The CNN input remains the 1-s window.
   Input durations are reported per model (CNN: 1 s; physics: 4 s).
2. **B1-V3-1s** is run exactly as specified in V2.2 (expected to be largely masked); it documents the effect of context
   length. Context effect = B1-V3-4s vs B1-V3-1s; architecture effect = A2/A4 vs B1-V3-4s (same physics input).
3. **PEV** interventions are applied to the first 4 s of each sampled measurement; physics models recompute 4-s
   concepts, CNN-based models average the four 1-s windows of the intervened signal.
4. **Physics-only final model** (V2.2 §9) is B1-V3-4s unless B1-V3-1s has higher pooled inner-OOF bearing macro-F1.
5. **Synthetic tests** adjusted to the physically correct expectation (1-s regions are not identifiable at 6203/1500 rpm);
   test signals for detection use 4-s records. Nothing else in V2.2 changes.
