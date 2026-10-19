# CAPA-Fi — Architecture, Workflow & Methodology

> **CAPA-Fi** (**C**ategory-**A**ware **P**artial **A**daptation **Fi**ltering) is a source-free test-time domain adaptation system for multi-user Wi-Fi Human Activity Recognition (HAR) using CSI (Channel State Information) signals from the WiMANS benchmark dataset.

---

## Table of Contents

1. [Research Problem](#1-research-problem)
2. [System Overview](#2-system-overview)
3. [Module Architecture](#3-module-architecture)
   - [Module 1 — Data Pipeline](#module-1--data-pipeline)
   - [Module 2 — Model Architecture](#module-2--model-architecture)
   - [Module 3 — Adaptation Engine](#module-3--adaptation-engine)
4. [End-to-End Workflow Pipeline](#4-end-to-end-workflow-pipeline)
5. [Tensor Contracts (Data Flow)](#5-tensor-contracts-data-flow)
6. [Training Objectives](#6-training-objectives)
7. [CAPA-Fi Novel Contributions](#7-capa-fi-novel-contributions)
8. [Configuration Reference](#8-configuration-reference)
9. [Directory Layout](#9-directory-layout)

---

## 1. Research Problem

### Domain Shift in Wi-Fi HAR

Deploying HAR models trained in one environment (source domain — Room 1, 5 GHz) to a different environment (target domain — Room 2, 2.4 GHz) degrades performance due to:

- **Frequency band differences**: 2.4 GHz vs. 5 GHz multipath profiles are distinct
- **Environmental geometry**: Different room layouts alter CSI propagation paths
- **Partial label shift**: Target domain may have fewer activity classes than source

### Partial-Set Adaptation

In **partial domain adaptation**, the target label space `Y_t ⊆ Y_s`. For example:

| Protocol      | Source Classes       | Target Classes        |
|---------------|----------------------|-----------------------|
| `closed_set`  | {1,2,3,4,5,6}       | {1,2,3,4,5,6}        |
| `partial_75`  | {1,2,3,4,5,6}       | {1,2,3,4} — 4 of 6   |
| `partial_50`  | {1,2,3,4,5,6}       | {1,2,3} — 3 of 6     |

Standard adaptation methods suffer **negative transfer** — absent source classes corrupt the target update. CAPA-Fi explicitly suppresses these via a learned category mask `γ_k`.

### Multi-User Complexity

A single CSI snapshot may contain **up to M=6 concurrent users**, each performing a different activity. The model must output a **set** of predictions — not a single label — which requires permutation-invariant training via the **Hungarian algorithm**.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            CAPA-Fi Pipeline                             │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────────┐    ┌───────────────────┐  │
│  │  MODULE 1    │    │      MODULE 2         │    │     MODULE 3      │  │
│  │  Data Engine │───▶│  Model Architecture  │───▶│ Adaptation Engine │  │
│  │  (Abey J.P.) │    │  (Ashlin Joe)        │    │ (Athishta P.A.)  │  │
│  └──────────────┘    └──────────────────────┘    └───────────────────┘  │
│                                                                         │
│  WiMANS CSI .npy ──▶ [B,3,3000,30] ──▶ HARModel ──▶ Adapted HARModel  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Architecture

### Module 1 — Data Pipeline

**File**: `src/dataset.py`

Loads WiMANS `.npy` CSI files referenced by `splits/source_split.csv` and `splits/target_split.csv`.

#### Processing Steps

```
Raw .npy  [T, Tx, Rx, S]
    │
    ├─ 1. Pad / truncate T → 3000
    ├─ 2. Reshape to [T, C=Tx*Rx, S] → permute → [C, T, S]
    ├─ 3. Clip to C=3 channels (or zero-pad)
    └─ 4. Per-sample z-score normalization (mean/std over T×S per channel)
    
Output: x [3, 3000, 30],  target_slots [6]  (dtype=torch.long)
```

#### Activity-to-Class Mapping

| Class | Activity  |
|-------|-----------|
| 0     | NO_PERSON |
| 1     | Walk      |
| 2     | Sit       |
| 3     | Stand     |
| 4     | Fall      |
| 5     | Wave      |
| 6     | Pick-up   |

#### Multi-Slot Target Encoding

Up to 6 concurrent users. Active users fill the first `n` slots; remaining slots padded with `0` (NO_PERSON). Example:

```
2 users [Walk, Sit] → target_slots = [1, 2, 0, 0, 0, 0]
```

---

### Module 2 — Model Architecture

#### 2a. Backbone: `SpatialTemporalBackbone` (`src/models/backbone.py`)

Encodes a raw CSI sample into a fixed-size latent vector `z ∈ ℝ^D`.

```
Input: [B, C=3, T=3000, S=30]
  │
  ├─ CNN Block 1: Conv2D(3→32)   + BN + ReLU + MaxPool2D(2,2)  → [B, 32, 1500, 15]
  ├─ CNN Block 2: Conv2D(32→64)  + BN + ReLU + MaxPool2D(2,2)  → [B, 64,  750,  7]
  ├─ CNN Block 3: Conv2D(64→128) + BN + ReLU + MaxPool2D(2,2)  → [B, 128, 375,  3]
  │
  ├─ Reshape: [B, 128, 375, 3] → [B, 375, 384]  (C*S = 128*3)
  ├─ BiLSTM (hidden=128, layers=2, bidirectional) → [B, 375, 256]
  └─ Mean-pool over time → z: [B, 256]
```

#### 2b. Classifier: `MultiSlotLinear` (`src/models/classifier.py`)

Maps latent embedding to M concurrent activity predictions.

```
z [B, 256]
  └─ Linear(256, M*(K+1)) → reshape → logits [B, M=6, K+1=7]
```

**Frozen during adaptation** (`requires_grad = False`). No softmax — raw logits only.

#### 2c. Rotation Head: `BinaryRotationMLP` (`src/models/rotation_head.py`)

Self-supervised auxiliary head for 180° rotation detection.

```
z [B, 256]
  └─ Linear(256, 64) → ReLU → Linear(64, 2) → rot_logits [B, 2]
```

**Trainable during adaptation**. Labels: `0` = original, `1` = 180°-inverted.

#### 2d. Hungarian Matcher: `HungarianMatcher` (`src/models/matcher.py`)

Finds the optimal (minimum-cost) bipartite assignment between M predicted slots and M ground-truth slots.

```
Cost matrix C[i, j] = -log P(target_class_j | predicted_slot_i)

Solves: argmin_π Σ_i C[i, π(i)]   via scipy.optimize.linear_sum_assignment
```

> **Non-differentiable** — runs under `torch.no_grad()`. Only assignment indices returned.

#### 2e. Unified Model: `HARModel` (`src/models/har_model.py`)

Wires backbone + classifier + rotation head into a single `nn.Module`.

```python
model.forward(x)              # → (logits [B, M, K+1], z [B, D])
model.forward_rotation(z)     # → rot_logits [B, 2]
model.freeze_classifier()     # sets classifier.requires_grad = False
model.unfreeze_backbone()     # sets backbone + rot_head.requires_grad = True
model.save_checkpoint(path)   # saves {backbone, classifier, rotation_head} separately
model.load_checkpoint(path)   # restores all three components independently
```

Aliases: `model.slot_head ≡ model.classifier`, `model.rot_head ≡ model.rotation_head`

#### 2f. Source Training: `train_source.py` (`src/models/train_source.py`)

End-to-end supervised source training with rotation SSL.

```
Per batch:
  1. Load (x [B,3,3000,30], target_slots [B,6]) from source DataLoader
  2. Apply CSIAugmentor (jitter, scaling, magnitude warp, optional others)
  3. Create rotation pair: x_original + x_freq_inv → x_combined [2B, 3, 3000, 30]
     (frequency inversion = flip subcarrier axis dims=[-1] ONLY — see note below)
  4. backbone(x_combined) → z_combined [2B, 256]
  5. classifier(z_combined[:B]) → logits [B, 6, 7]
  6. rotation_head(z_combined) → rot_logits [2B, 2]
  7. Hungarian matching: assign predicted slots to target slots
  8. Loss = L_matched_CE + λ_rot_source × L_rotation
  9. clip_grad_norm(max=1.0) → AdamW step (dual LR: backbone 1e-3, rot_head 1e-2)
```

> **Rotation Axis Note:** Only the **subcarrier axis** (`dims=[-1]`) is inverted, not the time axis.
> Flipping the time axis in a BiLSTM + global temporal mean-pool architecture produces embeddings
> that are nearly identical (forward pass ≈ backward pass after pooling), causing the rotation head
> to output uniform logits (ℓ=0.693 ≈ ln 2, random guessing). Subcarrier inversion reverses OFDM
> frequency-bin ordering (low↔high) without affecting temporal structure, yielding maximally
> distinct embeddings that allow the rotation head to learn a real discriminative signal.

#### 2g. Augmentation: `CSIAugmentor` (`src/utils/augmentation.py`)

Composable pipeline of CSI-specific augmentations (all shape-preserving `[B,C,T,S]`):

| Augmentation      | Description                                     |
|-------------------|-------------------------------------------------|
| `jitter`          | Additive Gaussian noise                         |
| `scaling`         | Per-channel multiplicative random factor        |
| `slice_shuffle`   | Random permutation of temporal segments         |
| `magnitude_warp`  | Smooth amplitude distortion via spline          |
| `window_warp`     | Time-axis stretch/compress of random segment    |

> 180° temporal inversion is **not** in the augmentor — it is performed inline in training.

---

### Module 3 — Adaptation Engine

#### 3a. Confidence Masking (`src/adaptation/confidence_masking.py`)

**Non-Background Sample Confidence Weight `w_i`**

Measures how confident the model is about human activity classes ($1..K$), explicitly excluding Class 0 (NO_PERSON) to prevent background collapse:

```
w_{i,m} = max_{k \ge 1} p_{i,m,k}                   (max non-background probability)
w_i     = (1/M) Σ_m w_{i,m}                         (mean across slots)
```

**Occupancy factor `p_occ`**

Estimates room occupancy based on background (No-Person) probability:

```
p_occ_i = 1 − (1/M) Σ_m P(class=0 | slot m, sample i)
```

**Category Mask Engine `γ_k`**

EMA-tracked estimate of which activity categories are present in the target domain, calibrated with $\tau_{\text{conf}} = 0.15$:

```
Confident mask:      I_{i,m} = [w_{i,m} ≥ τ_conf]
Class occurrences:   counts_k = bincount(pred_k | I = True)
Normalized freq:     freq_k   = counts_k / (Σ I + ε)
EMA smoothing:       μ_k      ← β · μ_k + (1-β) · freq_k   (for k = 0..K)
Dynamic filtering:   γ_0      = 1.0  (Class 0 / NO_PERSON always retained)
                     γ_k      = I[μ_k ≥ 0.10 · max_{j≥1}(μ_j)]  (for k = 1..K)
```

#### 3b. Loss Engine (`src/adaptation/loss_engine.py`)

**Category-Masked Conditional Entropy Loss `L_ent`** (minimised → sharp predictions on present classes)

```
p̃_{i,m,k} = (p_{i,m,k} · γ_k) / (Σ_j p_{i,m,j} · γ_j + ε)
L_ent     = − (1/BM) Σ_i Σ_m Σ_k  p̃_{i,m,k} · log(p̃_{i,m,k} + ε)
```

**Marginal Diversity Loss `L_div`** (maximised → prevents collapsed uniform predictions)

```
p̂_k   = mean_{i,m} p_{i,m,k}                        (batch marginal distribution)
L_div  = p̄_occ · Σ_k  p̂_k · log(p̂_k + ε)            (negative entropy)
```

**Rotation SSL Loss `L_rot`**

```
L_rot = CrossEntropy(rot_logits, rot_targets)       (0° vs. 180° subcarrier inversion)
```

**Total CAPA-Fi Target Adaptation Objective**

```
L = λ_ent · L_ent  +  λ_div · L_div  +  λ_rot · L_rot
```


Parameters learned: **backbone `f_θ`** and **rotation head `h_ψ`**.
Parameters frozen: **classifier `g_θ`** — weights must be bitwise invariant after adaptation.

#### 3c. Adaptation Trainer (`src/adaptation/trainer.py`)

`TargetAdaptationTrainer` / `CAPAFiAdaptationEngine`:

```
Per batch (unlabeled target CSI):
  1. x_target [B, 3, T, S] → model forward → (logits, z)
  2. x_target_freq_inv = flip(x_target, dims=[-1]) → model forward → z_freq_inv
     (subcarrier-only inversion — avoids BiLSTM temporal symmetry trap)
  3. Compute w_i, γ_k, p_occ via CategoryMaskEngine
  4. Compute L_total via CAPAFiLossEngine
  5. optimizer.zero_grad(); L_total.backward()
  6. clip_grad_norm(trainable params, max=1.0); optimizer.step()
  7. Accumulate telemetry (entropy, diversity, rotation, confidence, active classes, occupancy)

Per epoch:
  → Returns EpochMetrics (float + dict) with all telemetry averaged
  → After all epochs: save checkpoint + JSON history
```

---

### Module 4 — Evaluation & Benchmarking

#### 4a. Evaluation Metrics (`src/evaluation/metrics.py`)

Multi-user Wi-Fi HAR outputs an unordered set of $M=6$ slots. Evaluation must be permutation-invariant via Hungarian bipartite matching before calculating metrics.

1. **Permutation-Invariant Alignment**:
   ```
   Cost(m, j) = -log P(slot m predicts target class j)
   π* = argmin_π Σ_m Cost(m, π(m))
   ```
2. **Slot-Wise Accuracy**:
   $$\text{Acc}_{\text{slot}} = \frac{1}{BM} \sum_{i=1}^B \sum_{m=1}^M \mathbb{I}(\hat{y}_{i,m} = y_{i,m})$$
   Also computed on active slots only ($\text{Acc}_{\text{active}}$ where $y_{i,m} \neq 0$).
3. **Multi-User Occupancy MAE**:
   $$\text{MAE}_{\text{occ}} = \frac{1}{B} \sum_{i=1}^B \left| \sum_{m=1}^M \mathbb{I}(\hat{y}_{i,m} \neq 0) - \sum_{m=1}^M \mathbb{I}(y_{i,m} \neq 0) \right|$$
4. **Macro F1-Score**: Class-balanced unweighted harmonic mean across activity categories.
5. **Exact Match Ratio**: Proportion of samples where all $M=6$ slots match simultaneously.

#### 4b. Benchmarking & Visualizations (`src/evaluation/benchmark.py`)

Evaluates models under partial category shift protocols:
- `closed_set`: Full overlap $\mathcal{Y}_t = \{1, 2, 3, 4, 5, 6\}$
- `partial_75`: 75% overlap $\mathcal{Y}_t = \{1, 2, 3, 4\}$ (Classes 5, 6 absent)
- `partial_50`: 50% overlap $\mathcal{Y}_t = \{1, 2, 3\}$ (Classes 4, 5, 6 absent)

Artifacts generated in `results/`:
- **Confusion Matrix**: Row-normalized activity classification heatmap (`results/confusion_matrix_{protocol}.png`)
- **2D t-SNE Projections**: Manifold visualization of latent embeddings $z$ (`results/tsne_{protocol}.png`)
- **Loss Curves**: Multi-epoch convergence tracking Entropy, Diversity, and Confidence (`results/adaptation_loss_curves.png`)
- **LaTeX Tables**: Automated publication-grade comparison tables (`results/benchmark_table_{protocol}.tex`)

---

## 4. End-to-End Workflow Pipeline

```
Phase 1: SOURCE TRAINING
─────────────────────────
WiMANS source .npy files
    │
    ▼
src/dataset.py ──▶ MultiUserCSIDatasetWithPreprocessing
    │                   x: [B, 3, 3000, 30]
    │                   target_slots: [B, 6]
    ▼
src/models/train_source.py
    │   ├─ CSIAugmentor (jitter, scale, warp)
    │   ├─ create_rotation_pair → x_combined [2B, 3, 3000, 30]
    │   ├─ HARModel forward → logits [B, 6, 7], rot_logits [2B, 2]
    │   ├─ HungarianMatcher → assignments
    │   ├─ SourceTrainingLoss = L_matched_CE + λ × L_rot
    │   └─ AdamW + CosineAnnealing scheduler
    ▼
checkpoints/source_model.pt

Phase 2: TARGET ADAPTATION
────────────────────────────
WiMANS target .npy files (UNLABELED)
    │
    ▼
src/adaptation/run_adaptation.py
    │   ├─ Load source_model.pt → HARModel
    │   ├─ Freeze classifier g_θ
    │   ├─ Build target DataLoader
    │   ├─ CAPAFiAdaptationEngine.adapt(target_loader, epochs=30)
    │   │       Per batch:
    │   │         compute_weights_and_masks → w_i, γ_k, p_occ
    │   │         CAPAFiLossEngine → L_ent, L_div, L_rot
    │   │         AdamW backward on backbone + rot_head only
    │   ├─ Verify classifier frozen (bitwise)
    │   └─ Save checkpoints/adapted_model.pt + results/adaptation_history.json
    ▼
checkpoints/adapted_model.pt

Phase 3: EVALUATION & BENCHMARKING (Member 4)
─────────────────────────────────────────────
checkpoints/adapted_model.pt
    │
    ▼
src/evaluation/benchmark.py (--protocol [closed_set | partial_75 | partial_50])
    │   ├─ match_slots_hungarian → optimal bipartite permutation alignment
    │   ├─ evaluate_predictions → Slot-Acc, Active-Acc, Macro-F1, Occ-MAE, Exact-Match
    │   ├─ format_terminal_table → renders comparative table directly in terminal
    │   ├─ plot_benchmark_summary → unified 2x2 subplot figure in results/benchmark_{protocol}.png
    │   └─ (optional) export_latex_table → results/benchmark_table_{protocol}.tex (--export-latex)
    ▼
results/benchmark_{protocol}.json & results/benchmark_{protocol}.png
```

---

## 5. Tensor Contracts (Data Flow)

| Tensor          | Shape            | Description                                    |
|-----------------|------------------|------------------------------------------------|
| `x`             | `[B, 3, 3000, 30]`  | CSI input: channels × time × subcarriers    |
| `z`             | `[B, 256]`       | Backbone latent embedding                      |
| `logits`        | `[B, 6, 7]`      | Multi-slot class logits (raw, pre-softmax)     |
| `target_slots`  | `[B, 6]`         | Ground-truth class indices ∈ {0..6}            |
| `rot_logits`    | `[B, 2]` or `[2B, 2]` | Binary rotation logits                    |
| `rot_labels`    | `[2B]`           | 0=original, 1=180°-inverted                   |
| `w_i`           | `[B]`            | Sample confidence weights ∈ [0,1]              |
| `γ_k`           | `[6]`            | Category presence mask ∈ [0,1]                 |
| `p_occ`         | `[B]`            | Room occupancy factor ∈ [0,1]                  |
| `assignments`   | `List[(row, col)]` | Hungarian matching result (no gradient)      |

---

## 6. Training Objectives

### Source Training

```
L_source = L_matched_CE(logits, target_slots, assignments)
         + λ_rot_source × L_CE(rot_logits, rot_labels)
```

- `L_matched_CE`: permutation-invariant cross-entropy via Hungarian assignment
- `λ_rot_source = 0.5` (see `configs/default_config.yaml`)
- **Rotation strategy**: subcarrier-axis frequency inversion (`dims=[-1]`), not temporal flip
- **Optimizer**: dual-group AdamW `{backbone: lr=1e-3, rotation_head: lr=1e-2}` + gradient clipping (`max_norm=1.0`)

#### Why Subcarrier Inversion (Not Temporal Flip)

The backbone routes CSI through a **Bidirectional LSTM** then **global temporal mean-pooling**:
```
BiLSTM:   forward h→, backward h←   at each timestep t
z = mean_pool over t of [h→_t || h←_t]
```
When the time axis is flipped (`dims=[-2]`), the BiLSTM’s forward pass traces the original backward pass:
```
h→_t(x_rev) ≈ h←_t(x)   ⇒   z(x_rev) ≈ concat_swap(z(x))
```
After mean-pooling, `z(x_rev)` is near-identical in magnitude to `z(x)`, giving the rotation head zero discriminative signal — rotation loss plateaus at `ln(2) ≈ 0.693`.

Subcarrier inversion (`dims=[-1]`) reverses the OFDM frequency-bin ordering (30 bins: low↔high). The CNN spatial blocks process this as a genuinely different spectral pattern; the resulting embedding `z_freq_inv` is maximally distinct from `z`, allowing the rotation head to learn a real binary classifier.

### Target Adaptation (CAPA-Fi)

```
L_adapt = λ_ent · L_ent  −  λ_div · L_div  +  λ_rot · L_rot

where:
    L_ent = conditional entropy (confidence encouragement)
    L_div = masked marginal diversity (collapse prevention)
    L_rot = 180° rotation SSL (geometric consistency)
```

Default weights: `λ_ent=1.0, λ_div=1.0, λ_rot=0.5`

---

## 7. CAPA-Fi Novel Contributions

### 1. Dynamic Category Presence Mask (`γ_k`)

Suppresses gradient from absent source classes in the target domain using an EMA-tracked sigmoid gate. This directly mitigates **negative transfer** — the core failure mode of partial-set adaptation.

### 2. Occupancy-Gated Diversity Loss

The diversity penalty is scaled by `p̄_occ`, preventing the loss from penalising empty-room predictions where no activity should be detected.

### 3. Sample Confidence Weighting (`w_i`)

Uncertainty-aware pseudo-label weighting ensures only confident samples drive the EMA statistics update, making the system robust to highly uncertain batches.

### 4. Rotation SSL on Unlabeled Target Data

Subcarrier-axis frequency inversion (`flip(x, dims=[-1])`) is used as a self-supervised signal during both source training and target adaptation — no labels required. The rotation head is updated jointly with the backbone.

> **Critical design choice**: Temporal-axis flip (`dims=[-2]`) is *not* used because the BiLSTM + global mean-pool architecture creates near-identical embeddings for time-reversed inputs, reducing the task to random guessing. Subcarrier inversion produces a genuinely distinct spectral pattern that the CNN processes differently, enabling a real discriminative signal.

### 5. Permutation-Invariant Multi-User Prediction

The Hungarian matcher and matched cross-entropy loss make the system fully invariant to the ordering of concurrent users in the slot representation.

---

## 8. Configuration Reference

`configs/default_config.yaml` governs all modules:

| Section              | Key                          | Default  | Purpose                                  |
|----------------------|------------------------------|----------|------------------------------------------|
| `system`             | `seed`                       | 42       | Reproducibility                          |
| `system`             | `device`                     | `cuda`   | Compute device                           |
| `data`               | `window_size_T`              | 3000     | CSI temporal window length               |
| `data`               | `subcarriers_S`              | 30       | OFDM subcarrier count                    |
| `data`               | `channels_C`                 | 3        | Antenna channel pairs                    |
| `model`              | `max_users_M`                | 6        | Concurrent user slots                    |
| `model`              | `num_activity_classes_K`     | 6        | Activity classes (excl. NO_PERSON)       |
| `model`              | `latent_dim_D`               | 256      | Backbone output dimension                |
| `model`              | `temporal_hidden_dim`        | 128      | BiLSTM hidden units per direction        |
| `source_training`    | `lambda_rot_source`          | 0.5      | Source rotation SSL weight               |
| `source_training`    | `learning_rate`              | 1e-3     | Backbone + classifier AdamW LR           |
| `source_training`    | `learning_rate_rot_head`     | 1e-2     | Rotation head AdamW LR (10x backbone)    |
| `source_training`    | `grad_clip_norm`             | 1.0      | Max gradient L2 norm (0 = disabled)      |
| `adaptation`         | `method`                     | `capa_fi`| Adaptation variant                       |
| `adaptation`         | `lambda_ent`                 | 1.0      | Entropy weight                           |
| `adaptation`         | `lambda_div`                 | 1.0      | Diversity weight                         |
| `adaptation`         | `lambda_rot`                 | 0.5      | Freq-inversion SSL weight                |
| `adaptation`         | `grad_clip_norm`             | 1.0      | Max gradient L2 norm for adapt. step     |
| `adaptation`         | `tau_conf`                   | 0.7      | Confidence threshold for EMA update      |
| `adaptation`         | `ema_decay_beta`             | 0.9      | Category mask EMA smoothing factor       |
| `adaptation`         | `learning_rate_backbone`     | 1e-4     | Target adaptation LR                     |

---

## 9. Directory Layout

```
CAPA-Fi/
├── configs/
│   └── default_config.yaml         # All system hyperparameters
├── checkpoints/
│   └── adapted_model.pt            # Saved adapted model weights
├── results/
│   └── adaptation_history.json     # Per-epoch convergence telemetry
├── splits/
│   ├── source_split.csv            # Source domain annotation CSV
│   └── target_split.csv            # Target domain annotation CSV
├── docs/
│   ├── ARCHITECTURE.md             # This file
│   └── LEARNING_GUIDE.md           # Implementation walkthrough
├── src/
│   ├── dataset.py                  # MultiUserCSIDatasetWithPreprocessing
│   ├── models/
│   │   ├── backbone.py             # SpatialTemporalBackbone (CNN+BiLSTM)
│   │   ├── classifier.py           # MultiSlotLinear (frozen during adapt.)
│   │   ├── rotation_head.py        # BinaryRotationMLP (SSL head)
│   │   ├── har_model.py            # HARModel (unified wrapper)
│   │   ├── matcher.py              # HungarianMatcher (non-differentiable)
│   │   ├── losses.py               # MatchedCrossEntropyLoss, SourceTrainingLoss
│   │   └── train_source.py         # Source domain pre-training script
│   ├── adaptation/
│   │   ├── confidence_masking.py   # w_i, γ_k, p_occ computation
│   │   ├── loss_engine.py          # CAPAFiLossEngine
│   │   ├── trainer.py              # TargetAdaptationTrainer / CAPAFiAdaptationEngine
│   │   ├── mock_inputs.py          # MockHARModel for testing without real data
│   ├── evaluation/
│   │   ├── __init__.py             # Evaluation package interface
│   │   ├── metrics.py              # Slot accuracy, occupancy MAE, Macro F1, exact match
│   │   └── benchmark.py            # Partial shift benchmark runner & LaTeX exporter
│   └── utils/
│       └── augmentation.py         # CSIAugmentor and individual augmentation ops
├── tests/
│   └── test_capa_fi.py             # Consolidated project test suite
├── requirements.txt
├── README.md
└── .gitignore
```
