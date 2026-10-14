# CAPA-Fi — Member 2 Implementation Plan
**Permutation-Invariant Multi-User Modeling (Spatial-Temporal Backbone, M-Slot Classifier, Hungarian Matching, Rotation SSL)**

Owner: Ashlin Joe (Member 2)
Repo: `github.com/asherwoods39/capa-fi`
Status: Pre-implementation plan — no model code exists yet in `src/models/`

---

## 0. Resolved Project Decisions (ground truth for this plan)

| Item | Resolution | Source |
|---|---|---|
| Model input tensor contract | `[B, 3, 3000, 30]` (channels, time, subcarriers) | Team decision |
| Temporal downsampling schedule | `3000 → 1500 → 750 → 375` across 3 CNN blocks | Team decision |
| CNN → BiLSTM handoff shape | `[B, 128, 375, 3]` → reshaped to `[B, 375, 384]` | Team decision |
| Rotation SSL in source training | Included, with a **separate** `lambda_rot_source` weight | Team decision |
| `lambda_rot_source` numerical value | **Not set** — added as a required config field, value is an open project decision | Team decision |
| `adaptation.lambda_rot: 0.5` | Governs Member 3's adaptation phase only — **not** reused for source training | Team decision |
| 180° rotation transform (the augmentation itself) | Member 1's responsibility (data preprocessing) | `README.md` milestone table |
| `docs/` architecture spec files referenced in README | Do not exist in the repo — treat as not authoritative | Repo inspection |

---

## 1. Architecture

```
CSI [B, 3, 3000, 30]
 │
 ▼
Augmentation (jitter / scale / slice-shuffle / magnitude-warp — NOT the rotation transform)
 │
 ▼
CNN2D Block 1 → 32 channels, time 3000 → 1500
 │
 ▼
CNN2D Block 2 → 64 channels, time 1500 → 750
 │
 ▼
CNN2D Block 3 → 128 channels, time 750 → 375
 │
 ▼
[B, 128, 375, 3]  →  reshape  →  [B, 375, 384]
 │
 ▼
BiLSTM (hidden=128, layers=2, bidirectional)
 │
 ▼
z = [B, 256]
 ├─────────────────────────────┐
 ▼                              ▼
MultiSlotLinear              BinaryRotationMLP
 │                              │
 ▼                              ▼
[B, 2, 7]                     [B, 2]
 │
 ▼
Hungarian matching (scipy, non-differentiable)
 │
 ▼
Matched cross-entropy (differentiable, on matched pairs only)
```

## 2. Verified Tensor Contract

| Stage | Shape | Status |
|---|---|---|
| Model input | `[B, 3, 3000, 30]` | Resolved |
| CNN Block 1 output | `[B, 32, 1500, 15]` | Resolved |
| CNN Block 2 output | `[B, 64, 750, 7]` | Resolved |
| CNN Block 3 output | `[B, 128, 375, 3]` | Resolved |
| Reshaped for BiLSTM | `[B, 375, 384]` | Resolved |
| BiLSTM output (z) | `[B, 256]` | Fixed by config (128 × 2 directions) |
| Classifier output | `[B, 2, 7]` | Fixed by config (M=2, K+1=7) |
| Rotation head output | `[B, 2]` | Fixed by config |

Subcarrier pooling is fixed at `30 → 15 → 7 → 3` using 2×2 pooling in each CNN block.

## 3. File-by-File Plan

### `src/models/backbone.py`
- 3 Conv2D blocks (32 → 64 → 128 channels), each: conv → batchnorm → ReLU → pool.
- Each block uses 2×2 pooling, downsampling both time and subcarriers. Time: 3000→1500→750→375; subcarriers: 30→15→7→3.
- After Block 3: reshape `[B, 128, 375, 3]` → `[B, 375, 384]`.
- Feed into a 2-layer bidirectional LSTM (hidden=128).
- Output `z = [B, 256]` via mean-pooling across the LSTM's time outputs (default choice; document if changed).

### `src/models/classifier.py`
- `MultiSlotLinear`: single linear layer, `256 → 2×7`, reshaped to `[B, 2, 7]`.
- No dependency on backbone internals — must be freezable independently (`model.classifier.requires_grad_(False)`).

### `src/models/rotation_head.py`
- `BinaryRotationMLP`: Linear(256→64) → activation → Linear(64→2).
- Takes `z` directly; independent of classifier.
- Consumes rotation labels produced upstream by Member 1's preprocessing — does not generate the rotation transform itself.

### `src/models/matcher.py`
- Builds a `2×2` cost matrix per sample: cost = negative log-probability of the ground-truth class for each (predicted slot, target slot) pairing.
- Ground-truth slot tensor is `target_slots: [B, 2]`, with class indices `0..6` including No-Person.
- Detach + convert to NumPy before calling `scipy.optimize.linear_sum_assignment` (common integration bug if skipped).
- Returns assignment indices only — no gradient flows through this file.

### `src/models/losses.py`
- Reorders predictions (or ground truth — pick one direction, stay consistent) per the matcher's assignment.
- Computes cross-entropy on matched pairs only.
- This is where permutation invariance is actually enforced and tested.

### `src/models/train_source.py`
- Pipeline: load labeled source batch → augment classification inputs → receive original/180° pair from Member 1 → backbone → {classifier, rotation head} in parallel → Hungarian match → matched CE + rotation CE (weighted by `lambda_rot_source`) → backprop.
- Rotation SSL interface: `x_original [B,3,3000,30]` has rotation label 0; `x_rotated [B,3,3000,30]` has rotation label 1. Member 2 does not generate the 180° transform.
- Optimizer: `AdamW`; scheduler: `CosineAnnealingLR` (per `configs/default_config.yaml`).

### `src/utils/augmentation.py`
- Does not exist yet in the repo — new file.
- Implements: jittering, scaling, slice shuffling, magnitude/window warping.
- Explicitly NOT the rotation transform (Member 1's responsibility).
- Keep import-only, no config side effects, so it's reusable by other members if needed.

## 4. Config Changes Required

Add to `configs/default_config.yaml` under `source_training:`

```yaml
source_training:
  # existing fields unchanged
  lambda_rot_source: null   # REQUIRED — value is an open project decision, do not hard-code in code
```

Do not default this to a numeric value inside the code. `train_source.py` should fail loudly (raise an error) if `lambda_rot_source` is unset, rather than silently defaulting.

## 5. Training Objective

$$L_{\text{source}} = L_{\text{matched-CE}} + \lambda_{\text{rot\_source}} \cdot L_{\text{rotation}}$$

- `lambda_rot_source`: new, separate from `adaptation.lambda_rot: 0.5`. Value is an open team decision — not set in this plan.
- `adaptation.lambda_rot` is Member 3's parameter and is not reused here.

## 6. Testing Plan

Each test gates the next — do not proceed to the next numbered test until the current one passes.

1. **Backbone** — `[B,3,3000,30] → [B,256]`; verify shape, forward pass, backward pass, no NaNs. Also verify the intermediate time-axis shapes at each CNN block (1500, 750, 375) match the agreed schedule exactly.
2. **Classifier** — `[B,256] → [B,2,7]`.
3. **Rotation head** — `[B,256] → [B,2]`.
4. **Hungarian matching** — hand-constructed cost matrix with a known optimal assignment; verify the matcher returns it exactly.
5. **Permutation invariance** — feed the same correct answer through the loss in two different slot orders (e.g. `[Walk,Sit]` vs `[Sit,Walk]`); use `torch.testing.assert_close(loss1, loss2)`.
6. **Augmentation** — output shape unchanged, no invalid values, labels remain semantically unchanged.
7. **End-to-end forward pass** — CSI → augmentation → backbone → classifier/rotation head → matching → loss, no crashes.
8. **Tiny overfit** — 5–10 samples, verify the model can drive loss near zero before attempting full training.

## 7. Implementation Order

```
Stage 0 — Confirm data/model interface            [RESOLVED — contract is [B,3,3000,30]]
   ↓
Stage 1 — augmentation.py
   ↓
Stage 2 — backbone.py (CNN + BiLSTM, with 3000→1500→750→375 downsampling)
   ↓
Stage 3 — classifier.py (M-slot head)
   ↓
Stage 4 — matcher.py (Hungarian)
   ↓
Stage 5 — losses.py (matched cross-entropy)
   ↓
Stage 6 — rotation_head.py
   ↓
Stage 7 — train_source.py (combined loss, requires lambda_rot_source set in config)
   ↓
Stage 8 — Unit tests (§6, items 1–7)
   ↓
Stage 9 — Tiny overfit test (§6, item 8)
   ↓
Stage 10 — Full source training
   ↓
Stage 11 — Integration handoff to Member 3
```

## 8. Member 2 → Member 3 Interface

```
Input:              target CSI [B, 3, 3000, 30]
Backbone output:     z = [B, 256]
Classifier output:   [B, 2, 7]     (frozen during adaptation — config: freeze_classifier: true)
Rotation output:     [B, 2]

Frozen during adaptation:    classifier
Trainable during adaptation: backbone
Trainable during adaptation:       rotation head
```

## 9. Checkpoint Contents

`checkpoints/source_model.pt` must contain backbone, classifier, and rotation-head weights as **separately-addressable state dict keys** (not one flat blob), so Member 3 can load and freeze/adapt components independently without modifying Member 2's code.

```python
# Illustrative structure only — not implementation
{
  "backbone": backbone.state_dict(),
  "classifier": classifier.state_dict(),
  "rotation_head": rotation_head.state_dict(),
  "config": {...}  # architecture hyperparameters, for reconstruction before loading weights
}
```

## 10. Open Items Requiring Team Confirmation

- [ ] `lambda_rot_source` numerical value.
- [ ] BiLSTM-to-`z` pooling method (mean-pooling assumed as default; confirm or override).

## 11. Explicitly Out of Scope for Member 2

- Member 1's preprocessing, CSI parsing, phase sanitization, DWT denoising, rotation *transform* generation.
- Member 3's adaptation algorithm, Information Maximization, confidence weighting, category-aware filtering mask.
- Member 4's evaluation/benchmarking suite.

## 12. Definition of Done

All items in §6 pass, including the numerical-close permutation-invariance check. Tiny-overfit test drives loss near zero. Full source training completes and produces `checkpoints/source_model.pt` per §9. Member 3 can load the checkpoint, freeze the classifier, and run a forward pass on target CSI with zero changes to Member 2's code.
