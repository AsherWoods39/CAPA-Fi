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
   - [Module 4 — Evaluation & Benchmarking](#module-4--evaluation--benchmarking)
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

- **Frequency band differences**: 2.4 GHz vs. 5 GHz multipath propagation profiles are structurally distinct
- **Environmental geometry**: Different room layouts alter CSI propagation paths and reflections
- **Partial label shift**: Target domain may have fewer activity classes than source (Y_t ⊂ Y_s)
- **Multi-user concurrency**: Up to M=6 users producing superimposed reflections simultaneously

### Partial-Set Adaptation

In **partial domain adaptation**, the target label space Y_t ⊆ Y_s:

| Protocol | Source Classes | Target Classes |
|---|---|---|
| `closed_set` | {1, 2, 3, 4, 5, 6} | {1, 2, 3, 4, 5, 6} |
| `partial_75` | {1, 2, 3, 4, 5, 6} | {1, 2, 3, 4} — Wave & Pick-up absent |
| `partial_50` | {1, 2, 3, 4, 5, 6} | {1, 2, 3} — Fall, Wave & Pick-up absent |

Standard adaptation methods (SHOT, IM) suffer **negative transfer** under partial-set conditions — absent source classes corrupt the diversity loss, forcing the model to spread predictions across classes that don't exist in the target environment. CAPA-Fi explicitly suppresses this via a learned category mask `γ_k`.

### Multi-User Complexity

A single CSI snapshot may contain **up to M=6 concurrent users**, each performing a different activity. The model must output a **set** of predictions — not a single label — which requires:
- **Permutation-invariant training**: Hungarian matching before cross-entropy
- **Multi-slot output head**: M=6 parallel predictions over K+1=7 classes
- **Occupancy gating**: Distinguishing empty rooms from occupied ones

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
│  └──────────────┘    └──────────────────────┘    └────────┬──────────┘  │
│                                                           │             │
│  WiMANS .npy ──▶ [B,3,3000,30] ──▶ HARModel ──▶ Adapted HARModel     │
│                                                           │             │
│                                         ┌─────────────────▼───────────┐ │
│                                         │       MODULE 4              │ │
│                                         │    Evaluation & Benchmark   │ │
│                                         │    (Fathima Bushara M.C.)   │ │
│                                         └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Architecture

### Module 1 — Data Pipeline

**File**: `src/dataset.py`  
**Utilities**: `src/utils/download_data.py`, `src/utils/augmentation.py`

Loads WiMANS `.npy` CSI files referenced by `splits/source_split.csv` and `splits/target_split.csv`.

#### 1a. Data Download (`src/utils/download_data.py`)

```
download_wimans_dataset()
    ├─ Check if .npy files already present in data/raw/WiMANS/
    ├─ kagglehub.dataset_download("shuokanghuang/wimans")  [with auto-resume, max 50 retries]
    ├─ Recursively discover .npy files in downloaded directory
    ├─ NTFS hard-link to data/raw/WiMANS/ (0 extra disk space on Windows)
    └─ Optional: cleanup ~/.cache/kagglehub/ archive (~50 GB reclaimed)
```

#### 1b. Dataset Preprocessing (`MultiUserCSIDatasetWithPreprocessing`)

```
Raw .npy  [Time, Tx, Rx, Subcarriers]
    │
    ├─ Step 1: Pad / truncate Time axis → fixed_length = 3000
    ├─ Step 2: Reshape [T, Tx, Rx, S] → [T, Channels=Tx*Rx, S] → permute → [C, T, S]
    ├─ Step 3: Clip or zero-pad to C = 3 channels
    └─ Step 4: Per-sample z-score normalisation: (x - mean) / (std + 1e-5)
                (mean/std computed over T×S axes per channel independently)

Output: x [3, 3000, 30],  target_slots [6]  (dtype=torch.long)
```

If no `.npy` file is found and `strict=False` (default), a synthetic random tensor is returned. If `strict=True`, a `FileNotFoundError` is raised.

#### 1c. Activity-to-Class Mapping

| Class | Activity | Aliases |
|---|---|---|
| 0 | NO_PERSON | `no_person`, `none`, `nothing` |
| 1 | Walk | `walk` |
| 2 | Sit | `sit`, `sit_down` |
| 3 | Stand | `stand`, `stand_up` |
| 4 | Fall | `fall`, `lie_down` |
| 5 | Wave | `wave` |
| 6 | Pick-up | `pick_up`, `pickup` |

#### 1d. Multi-Slot Target Encoding (`extract_target_slots`)

Up to 6 concurrent users. Active users fill the first `n` slots; remaining slots padded with `0` (NO_PERSON):

```
0 users: target_slots = [0, 0, 0, 0, 0, 0]
1 user (Walk):           [1, 0, 0, 0, 0, 0]
2 users (Walk, Sit):     [1, 2, 0, 0, 0, 0]
```

#### 1e. CSI Augmentation (`CSIAugmentor`)

All augmentations are shape-preserving `[B, C, T, S]`:

| Augmentation | Default | Description |
|---|---|---|
| `jitter` | ✓ enabled | Additive Gaussian noise (σ=0.05) |
| `scaling` | ✓ enabled | Per-channel random multiplicative factor (σ=0.1) |
| `magnitude_warp` | ✓ enabled | Smooth spline amplitude distortion (σ=0.2, knots=4) |
| `slice_shuffle` | ✗ disabled | Random permutation of temporal segments |
| `window_warp` | ✗ disabled | Time-axis stretch/compress of random window |

Applied with probability `augment_prob=0.8` per batch.

> 180° frequency inversion is **not** part of `CSIAugmentor` — it is created inline in the training loop via `torch.flip(x, dims=[-1])`.

---

### Module 2 — Model Architecture

**Files**: `src/models/backbone.py`, `src/models/classifier.py`, `src/models/rotation_head.py`, `src/models/har_model.py`, `src/models/matcher.py`, `src/models/losses.py`, `src/models/train_source.py`

#### 2a. Backbone: `SpatialTemporalBackbone` (`src/models/backbone.py`)

Encodes a raw CSI sample into a fixed-size latent vector `z ∈ ℝ^D`.

```
Input: [B, C=3, T=3000, S=30]
  │
  ├─ CNN Block 1: Conv2D(3→32,  k=3, pad=1) + BN + ReLU + MaxPool2D(2,2) → [B, 32,  1500, 15]
  ├─ CNN Block 2: Conv2D(32→64, k=3, pad=1) + BN + ReLU + MaxPool2D(2,2) → [B, 64,   750,  7]
  ├─ CNN Block 3: Conv2D(64→128,k=3, pad=1) + BN + ReLU + MaxPool2D(2,2) → [B, 128,  375,  3]
  │
  ├─ Reshape: [B, 128, 375, 3] → permute → [B, 375, 128] → reshape → [B, 375, 384]  (C*S = 128*3)
  ├─ BiLSTM(input=384, hidden=128, layers=2, bidirectional=True, dropout=0.2) → [B, 375, 256]
  └─ Global mean-pool over time axis → z: [B, 256]
```

Note: The LSTM `input_size` is hardcoded to 384 (= 128 channels × 3 remaining subcarriers after 3 rounds of MaxPool2D(2,2) on S=30).

#### 2b. Classifier: `MultiSlotLinear` (`src/models/classifier.py`)

Maps latent embedding to M concurrent activity predictions.

```
z [B, 256]
  └─ Linear(256, M*(K+1) = 6*7 = 42) → reshape → logits [B, M=6, K+1=7]
```

**Frozen during adaptation** (`requires_grad = False`). Returns raw logits — no softmax applied. Cross-entropy loss applies log-softmax internally.

#### 2c. Rotation Head: `BinaryRotationMLP` (`src/models/rotation_head.py`)

Self-supervised auxiliary head for 180° rotation detection (subcarrier-axis inversion).

```
z [B, 256]
  └─ Linear(256, 64) → ReLU → Linear(64, 2) → rot_logits [B, 2]
```

**Trainable during both source training and adaptation.** Labels: `0` = original, `1` = freq-inverted.

#### 2d. Hungarian Matcher: `HungarianMatcher` (`src/models/matcher.py`)

Finds the minimum-cost bipartite assignment between M predicted slots and M ground-truth slots.

```
For each sample b in batch:
    cost_matrix[b, i, j] = -log_softmax(logits[b, i])[target_slots[b, j]]
    (row_ind, col_ind) = scipy.optimize.linear_sum_assignment(cost_matrix[b])
```

> **Non-differentiable** — runs under `@torch.no_grad()`. Predictions are `.detach()`-ed before cost computation. Only assignment indices are returned; no gradient flows through this module.

#### 2e. Unified Model: `HARModel` (`src/models/har_model.py`)

Wires backbone + classifier + rotation head into a single `nn.Module`. Provides checkpoint I/O with separately-addressable component state dicts.

```python
# Forward passes
logits, z   = model.forward(x)           # x [B,3,3000,30] → logits [B,6,7], z [B,256]
rot_logits  = model.forward_rotation(z)  # z [B,256] → rot_logits [B,2]

# Gradient control
model.freeze_classifier()   # classifier.requires_grad = False
model.unfreeze_backbone()   # backbone + rotation_head.requires_grad = True

# Checkpoint I/O
model.save_checkpoint(path, epoch, optimizer, config)
# Saves: {'backbone': ..., 'classifier': ..., 'rotation_head': ..., 'epoch': ..., 'optimizer': ..., 'config': ...}

model.load_checkpoint(path)
# Loads backbone, classifier, rotation_head state dicts independently

# Class method constructor
model = HARModel.from_config(config_dict)
```

Aliases: `model.slot_head ≡ model.classifier`, `model.rot_head ≡ model.rotation_head`

#### 2f. Source Losses (`src/models/losses.py`)

**`MatchedCrossEntropyLoss`**: Permutation-invariant cross-entropy using Hungarian matching.

```
For each sample b in batch:
    For each matched (pred_slot, gt_slot) pair:
        loss += CrossEntropy(logits[b, pred_slot, :], target_slots[b, gt_slot])
return total_loss / (B * M)
```

**`SourceTrainingLoss`**: Combined source training objective.

```
L_source = L_matched_CE(logits, target_slots) + λ_rot_source × CrossEntropy(rot_logits, rot_labels)
```

`λ_rot_source` must be explicitly set in `configs/default_config.yaml` — no default fallback (raises `ValueError` if missing).

#### 2g. Source Training: `train_source.py` (`src/models/train_source.py`)

Full source domain pre-training pipeline.

**Per-batch pipeline:**

```
1. Load (x [B,3,3000,30], target_slots [B,6]) from DataLoader
2. Apply CSIAugmentor (jitter, scaling, magnitude_warp)
3. create_rotation_pair(x_aug):
   - x_rotated = flip(x_aug, dims=[-1])    ← subcarrier-axis inversion ONLY
   - x_combined = cat([x_aug, x_rotated])  → [2B, 3, 3000, 30]
   - rot_labels = [0...0, 1...1]           → [2B]
4. logits_combined, z_combined = model(x_combined)  → [2B,6,7], [2B,256]
5. logits = logits_combined[:B]                      → [B,6,7]
6. rot_logits = model.forward_rotation(z_combined)  → [2B,2]
7. losses = SourceTrainingLoss(logits, target_slots, rot_logits, rot_labels)
8. losses["total"].backward()
9. clip_grad_norm_(model.parameters(), max_norm=1.0)
10. optimizer.step()
```

**Dual-LR AdamW optimizer:**

```python
optimizer = AdamW([
    {"params": backbone_and_classifier_params, "lr": 1e-3,  "weight_decay": 1e-4},
    {"params": rotation_head_params,           "lr": 1e-2,  "weight_decay": 1e-4},
])
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

Saves `checkpoints/source_model.pt` on best loss improvement, `checkpoints/source_model_final.pt` after all epochs.

---

### Module 3 — Adaptation Engine

**Files**: `src/adaptation/confidence_masking.py`, `src/adaptation/loss_engine.py`, `src/adaptation/trainer.py`, `src/adaptation/run_adaptation.py`

#### 3a. Confidence Masking (`src/adaptation/confidence_masking.py`)

**`compute_probs(logits)`**: Converts raw logits to clamped probabilities.

```
probs = softmax(logits, dim=-1)              [B, M, K+1]
probs = clamp(probs, min=1e-7, max=1-1e-7)
```

**`compute_sample_confidence(probs, tau_conf)`**: Per-sample entropy-based confidence weight.

```
H(p_{i,m}) = -Σ_k p_{i,m,k} · log(p_{i,m,k})       per-slot entropy
w_{i,m}    = 1 - H(p_{i,m}) / log(K+1)              normalised to [0,1]
w_i        = mean_m(w_{i,m})                          mean over M slots → [B]
```

High w_i = low entropy = confident prediction.

**`compute_occupancy(probs)`**: Estimated room occupancy per sample.

```
p_bg  = probs[:, :, 0]               Class 0 (NO_PERSON) probability  [B, M]
p_occ = 1 - mean_m(p_bg)                                              [B]
```

p_occ ≈ 0 → empty room; p_occ ≈ 1 → fully occupied.

**`CategoryMaskEngine` (stateful `nn.Module`)**:

EMA-tracked estimate of which activity categories are present in the target domain.

```
Per batch update (update_mask):
    1. probs = softmax(logits)
    2. act_probs = probs[:, :, 1:]        exclude Class 0
    3. max_act_probs, pred_acts = act_probs.max(dim=-1)
    4. confident_mask = max_act_probs >= tau_conf   (τ_conf = 0.15)
    5. counts = bincount(pred_acts[confident_mask], minlength=num_classes)
    6. freqs = counts / (sum(confident_mask) + ε)
    7. μ_k ← β · μ_k + (1-β) · freqs_k       (EMA, β=0.90)
    8. γ_0 = 1.0                              Class 0 always retained
       γ_k = sigmoid((μ_k - τ_thresh) / τ_temp)  for k=1..K
             (soft continuous gating prevents permanent class extinction)

Temperature annealing (step_epoch):
    τ_temp cosine-annealed from τ_temp_start (0.04) → τ_temp_end (0.015) over epochs
    (sharpening class selection as predictions cluster)
```

**`compute_weights_and_masks(logits, mask_engine, tau_conf)`**: Main entry point returning `(w_i, γ_k, p_occ)`.

#### 3b. Loss Engine (`src/adaptation/loss_engine.py`)

**`CAPAFiLossEngine`**: Combined CAPA-Fi adaptation objective (inherits `nn.Module`).

**1. Category-Masked Conditional Entropy Loss `L_ent`** (minimised → sharp predictions on present classes):

```
p̃_{i,m,k} = (p_{i,m,k} · γ_k) / (Σ_j p_{i,m,j} · γ_j + ε)    masked, renormalised
slot_entropy = -Σ_k p̃_{i,m,k} · log(p̃_{i,m,k} + ε)
L_ent = mean_{i,m}(slot_entropy)
```

**2. Active-Category Marginal Diversity Loss `L_div`** (maximised → prevents class collapse):

```
avg_masked = mean_{i,m}(p̃_{i,m,k})            batch marginal over active classes
active_avg = avg_masked[active_mask] / sum(avg_masked[active_mask] + ε)

inv_freq_k = 1 / (active_avg_k + δ=0.05)        inverse-frequency weighting
w_k = inv_freq_k / (mean(inv_freq_k) + ε)       normalised class weights

L_div = Σ_{k ∈ active} w_k · active_avg_k · log(active_avg_k + ε)
```

If `occupancy_weight` is provided: `L_div = p̄_occ · L_div`

**3. Rotation SSL Loss `L_rot`**:

```
rot_targets = cat([zeros(B), ones(B)])          [2B]: 0=original, 1=freq-inverted
L_rot = CrossEntropy(rot_logits [2B, 2], rot_targets [2B])
```

**Total CAPA-Fi Objective:**

```
L_adapt = λ_ent · L_ent  +  λ_div · L_div  +  λ_rot · L_rot
```

Default weights: `λ_ent=1.0, λ_div=0.25, λ_rot=0.30`

Returns a dict with both short-form and long-form keys for backward compatibility:
`{loss, loss_ent, loss_div, loss_rot, loss_total, loss_entropy, loss_diversity, loss_rotation, mean_confidence, active_classes_count, occupancy_factor}`

#### 3c. Adaptation Trainer (`src/adaptation/trainer.py`)

`TargetAdaptationTrainer` / `CAPAFiAdaptationEngine` (alias):

**Constructor**: Freezes `slot_head` (classifier), unfreezes `backbone` and `rot_head`. Builds AdamW optimizer on trainable params only. Initialises `CategoryMaskEngine` and `CAPAFiLossEngine`.

**`adapt_epoch(target_loader)`**: Single adaptation epoch over unlabeled target data.

```
For each batch (x_target):
    1. x_target_180 = flip(x_target, dims=[-1])   ← subcarrier inversion only
    2. logits, z = model(x_target)
    3. rot_logits_0 = model.forward_rotation(z)
    4. _, z_180 = model(x_target_180)
    5. rot_logits_180 = model.forward_rotation(z_180)
    6. γ_k, mean_conf = mask_engine.update_mask(logits)
    7. w_i = compute_sample_confidence(compute_probs(logits), tau_conf=0.15)
    8. p_occ = compute_occupancy(compute_probs(logits))
    9. rot_logits = cat([rot_logits_0, rot_logits_180])   [2B, 2]
       rot_targets = cat([zeros(B), ones(B)])              [2B]
    10. loss_dict = loss_engine(logits, rot_logits, rot_targets, γ_k, p_occ)
    11. loss_dict["loss_total"].backward()
    12. clip_grad_norm_(trainable_params, max_norm=1.0)
    13. optimizer.step()
    14. Accumulate telemetry (loss, entropy, diversity, rotation, confidence, active classes, occupancy)

Returns: EpochMetrics (float subclass + dict)
```

**`adapt(target_loader, epochs, save_path, results_path, verbose)`**: Multi-epoch loop with CosineAnnealingLR scheduler, checkpoint saving, and JSON telemetry export.

**`verify_classifier_frozen(state_before)`**: Verifies g_θ weights are bitwise identical to pre-adaptation snapshot.

**Ablation toggles** (controlled by config):
- `enable_category_filtering=False` → `γ_k = ones(...)` (ablates the category mask)
- `enable_occupancy_weighting=False` → `p_occ = ones(...)` (ablates occupancy gate)

#### 3d. Adaptation Runner (`src/adaptation/run_adaptation.py`)

CLI entry point for the full adaptation pipeline:

```
1. Load configs/default_config.yaml
2. Build/load target DataLoader
   - Prefers splits/target_split.csv + real .npy files
   - Falls back to 64 synthetic CSI samples if data not available
3. Load/create model
   - Loads checkpoints/source_model.pt if available
   - Falls back to initialized MockHARModel (matching API contract)
4. Instantiate CAPAFiAdaptationEngine(model, config)
5. Snapshot classifier state before adaptation
6. trainer.adapt(target_loader, epochs=5, save_path, results_path, verbose=True)
7. verify_classifier_frozen(state_before)   [assert invariant]
8. Print final telemetry summary
```

---

### Module 4 — Evaluation & Benchmarking

**Files**: `src/evaluation/metrics.py`, `src/evaluation/benchmark.py`

#### 4a. Evaluation Metrics (`src/evaluation/metrics.py`)

Multi-user output requires permutation-invariant evaluation via Hungarian matching before any metric computation.

**`match_slots_hungarian(logits, target_slots)`**:

```
For each sample b:
    cost_matrix[m, j] = -log(probs[b, m, target_slots[b, j]] + ε)
    (row_ind, col_ind) = linear_sum_assignment(cost_matrix)
    matched_preds[b, col_ind] = argmax(logits[b, row_ind], dim=-1)
Returns: matched_preds [B, M],  matched_probs [B, M, K+1]
```

**Metric functions** (all computed on Hungarian-matched predictions):

| Function | Formula | Description |
|---|---|---|
| `compute_slot_wise_accuracy` | `mean(ŷ == y)` | All-slot accuracy (incl. empty) |
| `compute_slot_wise_accuracy(ignore_empty=True)` | `mean(ŷ == y where y≠0)` | Active-user-only accuracy |
| `compute_occupancy_mae` | `mean(|count(ŷ≠0) - count(y≠0)|)` | Multi-user count error |
| `compute_macro_f1` | sklearn `f1_score(average='macro')` | Class-balanced F1 |
| `compute_exact_match_ratio` | `mean(all M slots correct)` | Strictest per-sample metric |

**`evaluate_predictions(logits, target_slots, num_classes=7)`**: Runs full pipeline and returns all metrics as a dict.

#### 4b. Benchmarking & Visualisations (`src/evaluation/benchmark.py`)

**`run_benchmark(config, protocol, model_checkpoint, num_eval_samples, export_latex)`**:

```
1. Build HARModel and load checkpoint (adapted_model.pt)
2. Ingest evaluation data:
   - Prefers real WiMANS samples from splits/target_split.csv filtered to protocol classes
   - Falls back to 64 synthetic samples if data unavailable
3. Forward pass → logits, latents
4. evaluate_predictions() → all metrics
5. Save results/benchmark_{protocol}.json
6. Build comparison dict against 5 literature baselines
7. format_terminal_table() → print to stdout
8. plot_benchmark_summary() → unified 2×2 subplot .png
9. plot_confusion_matrix() → standalone confusion matrix .png
10. [optional] export_latex_table() → .tex file
```

**Visualisation outputs:**

| File | Content |
|---|---|
| `results/benchmark_{protocol}.png` | 2×2 subplot: confusion matrix + paradigm comparison + occupancy MAE + convergence curves |
| `results/confusion_matrix_{protocol}.png` | Standalone row-normalised confusion matrix heatmap |
| `results/adaptation_loss_curves.png` | Multi-epoch entropy/diversity/total loss + confidence w_i |
| `results/benchmark_table_{protocol}.tex` | LaTeX comparison table (`--export-latex`) |

**t-SNE visualisation** (`plot_tsne_embeddings`): 2D projection of backbone embeddings z [N, 256] using `sklearn.manifold.TSNE(n_components=2, perplexity=min(30, N//4))`.

---

## 4. End-to-End Workflow Pipeline

```
═══════════════════════════════════════════════════════════════════════════
PHASE 0: DATA DOWNLOAD
─────────────────────────────────────────────────────────────────────────
python -m src.utils.download_data
    └── data/raw/WiMANS/*.npy  (49.3 GB, auto-resumed on network drops)

═══════════════════════════════════════════════════════════════════════════
PHASE 1: SOURCE TRAINING
─────────────────────────────────────────────────────────────────────────
splits/source_split.csv  +  data/raw/WiMANS/*.npy
    │
    ▼
MultiUserCSIDatasetWithPreprocessing
    x: [B, 3, 3000, 30],  target_slots: [B, 6]
    │
    ▼
src/models/train_source.py
    ├─ CSIAugmentor (jitter, scale, magnitude_warp, prob=0.8)
    ├─ create_rotation_pair → x_combined [2B, 3, 3000, 30]  (subcarrier flip)
    ├─ HARModel forward → logits [B, 6, 7],  rot_logits [2B, 2]
    ├─ HungarianMatcher → assignments
    ├─ SourceTrainingLoss = L_matched_CE + λ_rot_source × L_rotation
    ├─ clip_grad_norm + AdamW (dual LR: backbone 1e-3, rot_head 1e-2)
    └─ CosineAnnealingLR scheduler
    │
    ▼
checkpoints/source_model.pt  (best loss)
checkpoints/source_model_final.pt  (final epoch)

═══════════════════════════════════════════════════════════════════════════
PHASE 2: TARGET ADAPTATION
─────────────────────────────────────────────────────────────────────────
checkpoints/source_model.pt  +  splits/target_split.csv (UNLABELED)
    │
    ▼
src/adaptation/run_adaptation.py
    ├─ Load model → freeze g_θ (classifier), unfreeze f_θ + h_ψ
    ├─ AdamW on trainable params only (lr=2e-5)
    ├─ Per epoch: adapt_epoch(target_loader)
    │       Per batch:
    │         x_180 = flip(x, dims=[-1])
    │         logits, z = model(x);  rot_0 = model.forward_rotation(z)
    │         _, z_180 = model(x_180);  rot_180 = model.forward_rotation(z_180)
    │         γ_k, mean_conf = mask_engine.update_mask(logits)  [EMA]
    │         w_i = compute_sample_confidence(probs)
    │         p_occ = compute_occupancy(probs)
    │         L_total = λ_ent·L_ent + λ_div·L_div + λ_rot·L_rot
    │         L_total.backward(); clip_grad_norm; optimizer.step()
    ├─ verify_classifier_frozen(state_before)  [assert]
    └─ CosineAnnealingLR scheduler
    │
    ▼
checkpoints/adapted_model.pt
results/adaptation_history.json   (per-epoch telemetry)

═══════════════════════════════════════════════════════════════════════════
PHASE 3: EVALUATION & BENCHMARKING
─────────────────────────────────────────────────────────────────────────
checkpoints/adapted_model.pt
    │
    ▼
src/evaluation/benchmark.py  (--protocol partial_75 / partial_50 / closed_set)
    ├─ match_slots_hungarian → optimal bipartite alignment
    ├─ evaluate_predictions → Slot-Acc, Active-Acc, Macro-F1, Occ-MAE, Exact-Match
    ├─ format_terminal_table → comparative table (SHOT, WiFi-SFGADR, SAN,
    │                          Partial-SFUDA, MU-SHOT-Fi, CAPA-Fi) printed to terminal
    ├─ plot_benchmark_summary → results/benchmark_{protocol}.png  (2×2 subplots)
    ├─ plot_confusion_matrix → results/confusion_matrix_{protocol}.png
    └─ [optional] export_latex_table → results/benchmark_table_{protocol}.tex
    │
    ▼
results/benchmark_{protocol}.json
results/benchmark_{protocol}.png
```

---

## 5. Tensor Contracts (Data Flow)

| Tensor | Shape | Description |
|---|---|---|
| `x` | `[B, 3, 3000, 30]` | CSI input: channels × time × subcarriers |
| `x_combined` | `[2B, 3, 3000, 30]` | Rotation pair: original + freq-inverted |
| `z` | `[B, 256]` | Backbone latent embedding |
| `logits` | `[B, 6, 7]` | Multi-slot class logits (raw, pre-softmax) |
| `probs` | `[B, 6, 7]` | Softmax probabilities, clamped [1e-7, 1-1e-7] |
| `target_slots` | `[B, 6]` | Ground-truth class indices ∈ {0..6} |
| `rot_logits` | `[2B, 2]` | Binary rotation logits (original + inverted) |
| `rot_labels` | `[2B]` | 0=original, 1=freq-inverted |
| `w_i` | `[B]` | Sample confidence weights ∈ [0, 1] |
| `γ_k` | `[7]` | Category presence mask ∈ [0, 1], γ_0=1.0 always |
| `p_occ` | `[B]` | Room occupancy factor ∈ [0, 1] |
| `μ_k` | `[7]` | EMA running class presence statistics (buffer in CategoryMaskEngine) |

---

## 6. Training Objectives

### Source Training

```
L_source = L_matched_CE(logits [B,6,7], target_slots [B,6], Hungarian_assignments)
         + λ_rot_source × CrossEntropy(rot_logits [2B,2], rot_labels [2B])
```

- `L_matched_CE`: Hungarian-matched permutation-invariant cross-entropy, averaged over B×M pairs
- `λ_rot_source = 0.3` (config: `source_training.lambda_rot_source`)
- **Rotation strategy**: subcarrier-axis frequency inversion (`dims=[-1]`), NOT temporal flip (`dims=[-2]`)
- **Dual-LR AdamW**: backbone+classifier at 1e-3, rotation_head at 1e-2 (10×)
- **Scheduler**: CosineAnnealingLR(T_max=50)
- **Gradient clip**: max_norm=1.0

#### Critical Design Note: Why Subcarrier Inversion, Not Temporal Flip

The backbone routes CSI through a **Bidirectional LSTM** then **global temporal mean-pooling**:

```
BiLSTM:  forward h→_t, backward h←_t  at each timestep t
z = (1/T) Σ_t [h→_t || h←_t]
```

When the time axis is flipped (`dims=[-2]`), the BiLSTM's forward pass traces the original backward pass:

```
h→_t(x_reversed) ≈ h←_t(x_original)
z(x_reversed) ≈ feature-permuted version of z(x_original)
```

After mean-pooling over T=375 steps, `z(x_reversed)` is near-identical in magnitude to `z(x_original)`. The rotation head receives two embeddings of identical norm → rotation loss plateaus at `ln(2) ≈ 0.693` (random guessing).

**Subcarrier inversion** (`dims=[-1]`) reverses the OFDM frequency-bin ordering (bins 0..29 → 29..0). The 3 CNN blocks process this as a genuinely different spectral pattern. The resulting embedding `z_freq_inv` is maximally distinct from `z`, providing a real binary classification signal to the rotation head.

### Target Adaptation (CAPA-Fi)

```
L_adapt = λ_ent · L_ent  +  λ_div · L_div  +  λ_rot · L_rot
```

Default weights (from `configs/default_config.yaml`):
- `λ_ent = 1.0` (entropy minimisation)
- `λ_div = 0.25` (diversity maximisation — moderated to prevent forced uniform predictions)
- `λ_rot = 0.30` (rotation SSL)

Parameters learned: **backbone f_θ** + **rotation head h_ψ**
Parameters frozen: **classifier g_θ** — bitwise invariant, verified via `verify_classifier_frozen()` assertion.

---

## 7. CAPA-Fi Novel Contributions

### 1. Dynamic Category Presence Mask (γ_k)

The core novel component. Per-class soft sigmoid gate updated via EMA over confident target predictions. Suppresses the diversity penalty on absent source categories, eliminating the primary failure mode of partial-set SFUDA (negative transfer).

Unlike hard thresholding (Partial-SFUDA Unlearning), the sigmoid gating is continuous and temperature-annealed — preventing permanent extinction of minority classes that may be statistically underrepresented early in adaptation.

### 2. Occupancy-Gated Diversity Loss

Scales the marginal diversity penalty by estimated room occupancy `p̄_occ`. In fully empty rooms, no diversity pressure is applied — preventing the loss from penalising all-NO_PERSON predictions where no activity should be detected.

### 3. Sample Confidence Weighting (w_i)

Entropy-based, uncertainty-aware weighting of pseudo-label quality. Only confident samples (low per-slot entropy) drive the EMA statistics update in `CategoryMaskEngine`, making the system robust to highly uncertain batches typical of early adaptation.

### 4. Rotation SSL on Unlabeled Target Data

Subcarrier-axis frequency inversion provides a label-free geometric consistency signal during target adaptation — no labels required. The rotation head is updated jointly with the backbone, maintaining the self-supervised auxiliary task throughout adaptation.

### 5. Permutation-Invariant Multi-User Prediction

Hungarian Bipartite Matching is used consistently across:
- Source training (MatchedCrossEntropyLoss + HungarianMatcher)
- Target evaluation (match_slots_hungarian in metrics.py)

The system is fully invariant to the ordering of concurrent users in the M=6 slot representation.

---

## 8. Configuration Reference

`configs/default_config.yaml` governs all modules. Sections:

| Section | Key | Value | Purpose |
|---|---|---|---|
| `system` | `seed` | 42 | Global RNG seed |
| `system` | `device` | `cuda` | Compute device |
| `system` | `num_workers` | 4 | DataLoader workers |
| `data` | `window_size_T` | 3000 | CSI temporal window length |
| `data` | `subcarriers_S` | 30 | OFDM subcarrier count |
| `data` | `channels_C` | 3 | Antenna channel pairs |
| `data` | `sampling_rate_hz` | 1000 | CSI packet frequency |
| `data` | `protocol` | `partial_75` | Active partial-shift protocol |
| `data` | `source_classes` | [1..6] | Source activity class IDs |
| `data` | `target_classes_partial_75` | [1,2,3,4] | Target classes: Wave & Pick-up absent |
| `data` | `target_classes_partial_50` | [1,2,3] | Target classes: Fall, Wave, Pick-up absent |
| `model` | `max_users_M` | 6 | Concurrent user slots |
| `model` | `num_activity_classes_K` | 6 | Activity classes (excl. NO_PERSON) |
| `model` | `total_classes` | 7 | K+1 (incl. class 0: NO_PERSON) |
| `model` | `spatial_channels` | [32, 64, 128] | CNN channel progression |
| `model` | `temporal_hidden_dim` | 128 | BiLSTM hidden dim per direction |
| `model` | `temporal_layers` | 2 | BiLSTM stacked layers |
| `model` | `latent_dim_D` | 256 | Backbone embedding dimension |
| `model` | `dropout` | 0.2 | CNN + BiLSTM dropout |
| `source_training` | `epochs` | 50 | Source training epochs |
| `source_training` | `batch_size` | 32 | Source DataLoader batch size |
| `source_training` | `learning_rate` | 1e-3 | Backbone + classifier LR |
| `source_training` | `learning_rate_rot_head` | 1e-2 | Rotation head LR (10× backbone) |
| `source_training` | `lambda_rot_source` | 0.3 | Source rotation SSL weight |
| `source_training` | `grad_clip_norm` | 1.0 | Max gradient L2 norm |
| `adaptation` | `method` | `capa_fi` | `source_only`, `shot`, `mu_shot_fi`, `capa_fi` |
| `adaptation` | `freeze_classifier` | `true` | Enforce g_θ requires_grad=False |
| `adaptation` | `epochs` | 30 | Target adaptation epochs |
| `adaptation` | `batch_size` | 32 | Target DataLoader batch size |
| `adaptation` | `learning_rate_backbone` | 2e-5 | Backbone adaptation LR |
| `adaptation` | `lambda_ent` | 1.0 | Entropy weight |
| `adaptation` | `lambda_div` | 0.25 | Diversity weight |
| `adaptation` | `lambda_rot` | 0.30 | Rotation SSL weight |
| `adaptation` | `grad_clip_norm` | 1.0 | Max gradient L2 norm |
| `adaptation` | `tau_conf` | 0.15 | Confidence threshold for EMA |
| `adaptation` | `tau_thresh` | 0.05 | Category presence cutoff |
| `adaptation` | `tau_temp` | 0.04 | Sigmoid temperature (annealed) |
| `adaptation` | `ema_decay_beta` | 0.90 | EMA smoothing factor for μ_k |
| `evaluation` | `metrics` | [slot_acc, occ_mae, macro_f1, exact_match] | Evaluation metrics list |
| `evaluation` | `output_dir` | `results/` | Evaluation output directory |

---

## 9. Directory Layout

```
CAPA-Fi/
├── configs/
│   └── default_config.yaml         # All system hyperparameters
├── checkpoints/
│   ├── source_model.pt             # Best source training checkpoint
│   ├── source_model_final.pt       # Final source training checkpoint
│   └── adapted_model.pt            # Output of run_adaptation.py
├── results/
│   ├── benchmark_{protocol}.json   # Raw evaluation metrics
│   ├── benchmark_{protocol}.png    # 2×2 evaluation subplot figure
│   └── confusion_matrix_{protocol}.png
├── splits/
│   ├── source_split.csv            # Source domain annotation CSV
│   └── target_split.csv            # Target domain annotation CSV
├── data/
│   └── raw/WiMANS/                 # WiMANS .npy files
├── docs/
│   ├── ARCHITECTURE.md             # This file
│   └── LEARNING_GUIDE.md           # Module-by-module learning walkthrough
├── src/
│   ├── dataset.py                  # MultiUserCSIDatasetWithPreprocessing
│   ├── models/
│   │   ├── backbone.py             # SpatialTemporalBackbone (CNN2D + BiLSTM)
│   │   ├── classifier.py           # MultiSlotLinear (frozen during adaptation)
│   │   ├── rotation_head.py        # BinaryRotationMLP (SSL head)
│   │   ├── har_model.py            # HARModel (unified wrapper + checkpoint I/O)
│   │   ├── matcher.py              # HungarianMatcher (no gradient)
│   │   ├── losses.py               # MatchedCrossEntropyLoss, SourceTrainingLoss
│   │   └── train_source.py         # Source pre-training script
│   ├── adaptation/
│   │   ├── confidence_masking.py   # compute_probs, w_i, γ_k, p_occ, CategoryMaskEngine
│   │   ├── loss_engine.py          # CAPAFiLossEngine
│   │   ├── trainer.py              # TargetAdaptationTrainer / CAPAFiAdaptationEngine
│   │   ├── run_adaptation.py       # Adaptation runner CLI script
│   │   └── mock_inputs.py          # MockHARModel for test/offline use
│   ├── evaluation/
│   │   ├── metrics.py              # Hungarian matching + all evaluation metrics
│   │   └── benchmark.py            # Benchmark runner, visualisers, LaTeX exporter
│   └── utils/
│       ├── augmentation.py         # CSIAugmentor + individual augmentation functions
│       └── download_data.py        # WiMANS auto-resume downloader
├── tests/
│   └── test_capa_fi.py             # Full test suite (unit + integration)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 10. Checkpoint-Based Execution Pipeline

For fast iteration and CI pipelines, any phase can be resumed from an existing checkpoint without re-running prior phases.

### Phase Checkpoints and Artifacts

| Phase | Command | Reads | Writes |
|---|---|---|---|
| Data Download | `python -m src.utils.download_data` | Kaggle API | `data/raw/WiMANS/*.npy` |
| Source Training | `python -m src.models.train_source` | `splits/source_split.csv` + `.npy` | `checkpoints/source_model.pt`, `source_model_final.pt` |
| Target Adaptation | `python -m src.adaptation.run_adaptation` | `source_model.pt` + `splits/target_split.csv` + `.npy` | `checkpoints/adapted_model.pt`, `results/adaptation_history.json` |
| Benchmark | `python -m src.evaluation.benchmark` | `adapted_model.pt` + `splits/target_split.csv` + `.npy` | `results/benchmark_{protocol}.json`, `.png` |

### Skip-to-Adaptation (Source Checkpoint Available)

```bash
# Requires: checkpoints/source_model.pt
python -m src.adaptation.run_adaptation
python -m src.evaluation.benchmark --protocol partial_75
```

### Skip-to-Benchmark (Adapted Checkpoint Available)

```bash
# Requires: checkpoints/adapted_model.pt
python -m src.evaluation.benchmark \
    --protocol partial_75 \
    --checkpoint checkpoints/adapted_model.pt \
    --samples 64
```

### Smoke Test Pipeline (No Real Data, No GPU)

All three stages automatically fall back to synthetic CSI tensors when `.npy` files are absent. This allows end-to-end pipeline validation in under 2 minutes on CPU:

```bash
# 1. Source training — 2 batches per epoch, 1 epoch (synthetic fallback)
python -m src.models.train_source \
    --config configs/default_config.yaml \
    --epochs 1 \
    --batch_size 4 \
    --limit_batches 2

# 2. Target adaptation — 5 epochs (synthetic fallback)
python -m src.adaptation.run_adaptation

# 3. Benchmark — 32 synthetic evaluation samples
python -m src.evaluation.benchmark \
    --protocol partial_75 \
    --samples 32
```

> **Important**: Synthetic fallback results are statistically random and carry no meaningful performance information. Use only for pipeline validation.

### Checkpoint File Formats

**`checkpoints/source_model.pt`** (saved by `HARModel.save_checkpoint()`):

```python
{
    "backbone":      OrderedDict,   # SpatialTemporalBackbone state_dict
    "classifier":    OrderedDict,   # MultiSlotLinear state_dict
    "rotation_head": OrderedDict,   # BinaryRotationMLP state_dict
    "epoch":         int,           # Epoch number when saved
    "optimizer":     OrderedDict,   # AdamW optimizer state (optional)
    "config":        dict,          # YAML config at training time
}
```

**`checkpoints/adapted_model.pt`** (saved by `TargetAdaptationTrainer.save_checkpoint()`):

```python
# Flat state_dict (torch.save(model.state_dict()))
# Keys: backbone.cnn_blocks.0.conv.weight, ... classifier.fc.weight, ...
```

### Loading a Source Checkpoint Manually

```python
from src.models.har_model import HARModel
import yaml

with open("configs/default_config.yaml") as f:
    config = yaml.safe_load(f)

model = HARModel.from_config(config)
ckpt = model.load_checkpoint("checkpoints/source_model.pt")
print(f"Loaded epoch {ckpt['epoch']}")

# Prepare for adaptation: freeze classifier, unfreeze backbone + rotation head
model.freeze_classifier()
model.unfreeze_backbone()
```

### Loading an Adapted Checkpoint for Evaluation

```python
import torch
from src.models.har_model import HARModel
from src.evaluation.metrics import evaluate_predictions

model = HARModel(in_channels=3, num_slots=6, total_classes=7)
state = torch.load("checkpoints/adapted_model.pt", map_location="cpu", weights_only=False)

# Handle both flat and component-keyed checkpoint formats
if "backbone" in state and isinstance(state["backbone"], dict):
    model.load_checkpoint("checkpoints/adapted_model.pt")
else:
    model.load_state_dict(state, strict=False)

model.eval()
```
