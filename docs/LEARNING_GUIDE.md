# CAPA-Fi — Learning Guide

> A complete walkthrough of the CAPA-Fi codebase — what each file does, how
> the components connect, and how to run everything from scratch.

---

## Table of Contents

1. [What is CAPA-Fi?](#1-what-is-capa-fi)
2. [Key Concepts to Understand First](#2-key-concepts-to-understand-first)
3. [Codebase Walkthrough — Module by Module](#3-codebase-walkthrough--module-by-module)
   - [Data Pipeline](#31-data-pipeline--srcdatasetpy)
   - [Augmentation](#32-augmentation--srcutilsaugmentationpy)
   - [Backbone](#33-backbone--srcmodelsbackbonepy)
   - [Classifier Head](#34-classifier-head--srcmodelsclassifierpy)
   - [Rotation Head](#35-rotation-head--srcmodelsrotation_headpy)
   - [Hungarian Matcher](#36-hungarian-matcher--srcmodelsmatcherpy)
   - [Losses](#37-losses--srcmodelslossspy)
   - [HARModel](#38-unified-model--srcmodelshar_modelpy)
   - [Source Training](#39-source-training--srcmodelstrain_sourcepy)
   - [Confidence Masking](#310-confidence-masking--srcadaptationconfidence_maskingpy)
   - [Loss Engine](#311-loss-engine--srcadaptationloss_enginepy)
   - [Trainer](#312-adaptation-trainer--srcadaptationtrainerpy)
   - [Run Script](#313-run-script--srcadaptationrun_adaptationpy)
   - [Evaluation Metrics](#314-evaluation-metrics--srcevaluationmetricspy)
   - [Benchmarking & Visualizations](#315-benchmarking--visualizations--srcevaluationbenchmarkpy)
4. [Order of Execution & How to Run the Project](#4-order-of-execution--how-to-run-the-project)
   - [Phase 1: Setup & Pre-Flight Validation](#41-phase-1-setup--pre-flight-validation)
   - [Phase 2: Source Domain Pre-Training](#42-phase-2-source-domain-pre-training)
   - [Phase 3: Source-Free Target Adaptation](#43-phase-3-source-free-target-adaptation)
   - [Phase 4: Target Evaluation & Benchmarking](#44-phase-4-target-evaluation--benchmarking)
5. [How to Run the Tests](#5-how-to-run-the-tests)
6. [Mental Model: One Forward Pass](#6-mental-model-one-forward-pass)
7. [Common Questions](#7-common-questions)
8. [Performance Fixes Applied](#8-performance-fixes-applied)
   - [Rotation SSL Stagnation Fix](#81-rotation-ssl-stagnation-fix-rot-loss-stuck-at-0693)
   - [Dual Learning Rate for Rotation Head](#82-dual-learning-rate-for-rotation-head)
   - [Gradient Clipping](#83-gradient-clipping)
   - [Convergence Monitoring Checklist](#84-convergence-monitoring-checklist)

---

## 1. What is CAPA-Fi?

CAPA-Fi is a deep learning system that:

1. **Trains** a Human Activity Recognition (HAR) model on Wi-Fi CSI data from a **source environment** (e.g. Room 1 at 5 GHz)
2. **Adapts** that model — **without any labels** — to work in a **target environment** (e.g. Room 2 at 2.4 GHz)
3. Handles **multiple people** in the room simultaneously (up to 6)
4. Handles **partial class shifts** where the target room has fewer activities than the source

The system reads Wi-Fi **CSI (Channel State Information)** signals — data your router broadcasts 1000 times per second that subtly changes as people move around. By processing these signals through a neural network, we can identify who is doing what.

---

## 2. Key Concepts to Understand First

### CSI Tensor Shape: `[B, C, T, S]`

Every raw CSI input has shape `[B, 3, 3000, 30]`:
- `B` = batch size (e.g. 32 samples)
- `C = 3` = antenna channels (from 1 transmitter × 3 receivers)
- `T = 3000` = time steps (3 seconds at 1000 Hz)
- `S = 30` = OFDM subcarriers (frequency channels)

### Multi-Slot Prediction

The model doesn't output one activity — it outputs **M=6 slots**, one per possible user:
- `logits: [B, 6, 7]` — 6 user slots, 7 possible classes each (6 activities + NO_PERSON)
- `target_slots: [B, 6]` — ground truth, with unused slots set to `0` (NO_PERSON)

### Permutation Invariance

If Room A has [Walk, Sit, empty, empty, empty, empty], the model should give the same loss whether we encode that as `[1, 2, 0, 0, 0, 0]` or `[2, 1, 0, 0, 0, 0]` or any other ordering. The **Hungarian matcher** solves this by finding the optimal slot-to-ground-truth assignment before computing the loss.

### Source-Free Adaptation

During target adaptation, **no labels are available**. We use:
- **Entropy minimization** — push the model to be more certain
- **Diversity maximization** — prevent it from collapsing to one prediction
- **Rotation SSL** — a self-supervised trick: flip the signal 180° and train the model to detect it

---

## 3. Codebase Walkthrough — Module by Module

### 3.1 Data Pipeline — `src/dataset.py`

**What it does**: Loads WiMANS `.npy` CSI files and converts them into tensors.

**Key class**: `MultiUserCSIDatasetWithPreprocessing`

```python
dataset = MultiUserCSIDatasetWithPreprocessing(
    annotation_file="splits/source_split.csv",
    data_dir="data/processed/WiMANS",
    fixed_length=3000,
    num_slots=6,
)
x, target_slots = dataset[0]
# x: [3, 3000, 30]  target_slots: [6]
```

**What it reads**: The CSV files have columns like `label`, `user_1_activity`, `user_2_activity`, ..., `user_6_activity`.

**Preprocessing steps**:
1. Load `.npy` file → shape `[T, Tx, Rx, S]`
2. Pad or truncate time to exactly 3000 steps
3. Flatten antennas: `[T, Tx, Rx, S] → [T, C, S]` where `C = Tx × Rx` (truncated/padded to `C=3`)
4. Transpose to `[C, T, S]`
5. Normalize: subtract mean, divide by std (zero-mean unit-variance per sample)

**Data Resolution & Downloader (`src/utils/download_data.py`)**:
- WiMANS raw CSI files (~49.3 GB) can be downloaded via:
  ```bash
  python -m src.utils.download_data --retries 50 --cleanup-cache
  ```
- **Auto-Resume**: Uses HTTP range requests to resume from existing progress if the connection drops.
- **Zero-Copy NTFS Hard-Linking**: Populates `data/raw/WiMANS` using hard links on Windows without duplicating 50 GB.
- **Directory Discovery**: `MultiUserCSIDatasetWithPreprocessing` automatically checks both `data/raw/WiMANS` and `data/processed/WiMANS`.
- **Strict Mode (`strict=True`)**: Raises `FileNotFoundError` if `.npy` files are missing during real training, preventing silent training on random numbers.
- **Synthetic fallback (`strict=False`)**: Used during unit testing (`tests/test_capa_fi.py`) so tests run offline in ~20 seconds without downloading 50 GB.

---

### 3.2 Augmentation — `src/utils/augmentation.py`

**What it does**: Applies random transformations to CSI tensors during source training to improve generalization.

**Key functions** (all take `[B, C, T, S]` → `[B, C, T, S]`):

| Function | What it does |
|----------|--------------|
| `jitter` | Adds small Gaussian noise |
| `scaling` | Multiplies by a random scale factor per channel |
| `slice_shuffle` | Randomly reorders time segments |
| `magnitude_warp` | Applies a smooth random amplitude curve |
| `window_warp` | Stretches or compresses a random time window |

**Key class**: `CSIAugmentor` — a configurable pipeline of the above:

```python
aug = CSIAugmentor(augment_prob=0.8, enable_jitter=True, enable_scaling=True)
x_aug = aug(x)  # same shape as x
```

> The 180° flip (used for rotation SSL) is **not** here — it's done inline during training.

---

### 3.3 Backbone — `src/models/backbone.py`

**What it does**: Converts raw CSI tensor `[B, 3, 3000, 30]` into a compact feature vector `z [B, 256]`.

**Architecture** (two stages):

**Stage 1 — Spatial CNN**: Three `CNN2DBlock` layers, each doing `Conv2D → BatchNorm → ReLU → MaxPool2D(2,2)`.

```
[B, 3, 3000, 30] → [B, 32, 1500, 15] → [B, 64, 750, 7] → [B, 128, 375, 3]
```

**Stage 2 — Temporal BiLSTM**: Reshape the CNN output and feed to a 2-layer bidirectional LSTM.

```
[B, 128, 375, 3] → reshape → [B, 375, 384] → BiLSTM → [B, 375, 256] → mean → [B, 256]
```

**Why these choices?**
- CNN captures local spatial patterns (subcarrier × time correlations)
- BiLSTM captures long-range temporal dependencies (motion sequences)
- Mean pooling aggregates all time steps into one fixed-size embedding

---

### 3.4 Classifier Head — `src/models/classifier.py`

**What it does**: Maps embedding `z [B, 256]` to multi-slot predictions `[B, 6, 7]`.

```python
class MultiSlotLinear(nn.Module):
    def forward(self, z):
        out = self.fc(z)           # [B, 42]  (fc: 256 → 6*7)
        return out.view(B, 6, 7)   # [B, 6, 7]
```

Simple linear layer — no softmax (raw logits returned). **Frozen during adaptation.**

---

### 3.5 Rotation Head — `src/models/rotation_head.py`

**What it does**: Given embedding `z [B, 256]`, predicts whether the input was flipped 180° or not.

```python
class BinaryRotationMLP(nn.Module):
    # Linear(256, 64) → ReLU → Linear(64, 2)
    # Output: [B, 2] logits for (original, rotated)
```

**Trainable during adaptation.** This is the self-supervised signal that helps the backbone adapt without labels.

---

### 3.6 Hungarian Matcher — `src/models/matcher.py`

**What it does**: Given model predictions and ground-truth labels (both as M=6 slots), finds the optimal slot-to-slot assignment.

**Why it's needed**: If the model predicts [Walk, Sit, ...] and the ground truth is [Sit, Walk, ...], the loss should be low (correct prediction, wrong order). The matcher figures out that prediction slot 0 matches ground-truth slot 1 and vice versa.

**How it works**:
1. Build cost matrix: `cost[i, j] = -log P(gt_class_j | predicted_slot_i)`
2. Call `scipy.optimize.linear_sum_assignment(cost_matrix)` → returns row/column indices
3. Use those indices to pair predictions with ground truth before computing cross-entropy

**Critical**: Runs under `torch.no_grad()` — no gradients flow through the matcher.

---

### 3.7 Losses — `src/models/losses.py`

**`MatchedCrossEntropyLoss`**: Uses the Hungarian matcher to compute permutation-invariant cross-entropy.

```python
loss_fn = MatchedCrossEntropyLoss()
loss = loss_fn(logits, target_slots)  # logits: [B,6,7], target: [B,6]
```

**`SourceTrainingLoss`**: Combines matched CE with rotation loss for source training.

```python
loss_fn = SourceTrainingLoss(lambda_rot_source=0.5)
losses = loss_fn(logits, target_slots, rot_logits, rot_labels)
# Returns: {"total": ..., "matched_ce": ..., "rotation": ...}
```

---

### 3.8 Unified Model — `src/models/har_model.py`

**What it does**: Combines backbone + classifier + rotation head into one `nn.Module` with a clean API.

```python
model = HARModel()

# Forward pass
logits, z = model(x)                  # x: [B,3,3000,30]
rot_logits = model.forward_rotation(z)

# Adaptation preparation
model.freeze_classifier()             # g_θ frozen
model.unfreeze_backbone()             # f_θ + h_ψ trainable

# Checkpointing
model.save_checkpoint("checkpoints/model.pt", epoch=5)
ckpt = model.load_checkpoint("checkpoints/model.pt")
```

Provides aliases for compatibility: `model.slot_head` = `model.classifier`, `model.rot_head` = `model.rotation_head`.

---

### 3.9 Source Training — `src/models/train_source.py`

**What it does**: Runs supervised training on labeled source data.

**Entry point**:
```bash
python -m src.models.train_source --config configs/default_config.yaml
```

**Per-epoch logic**:
1. Load batch `(x, target_slots)`
2. Augment: `x_aug = CSIAugmentor()(x)`
3. Create rotation pair: `x_combined = cat([x_aug, flip(x_aug)])` → shape `[2B, 3, 3000, 30]`
4. Forward: `logits_combined, z_combined = model(x_combined)`
5. Split: `logits = logits_combined[:B]` (original samples only)
6. Rotation head: `rot_logits = model.forward_rotation(z_combined)` (all 2B)
7. Loss = `L_matched_CE + 0.5 × L_rotation`
8. Backward + AdamW + CosineAnnealing scheduler step

Saves best checkpoint to `checkpoints/source_model.pt`.

---

### 3.10 Confidence Masking — `src/adaptation/confidence_masking.py`

**What it does**: Computes non-background sample confidence and maintains the dynamic Category Filtering Mask $\gamma_k$.

**`compute_probs(logits)`** → softmax probabilities `[B, M, K+1]`

**`CategoryMaskEngine.update_mask(logits)`** → `(γ, mean_confidence)`:
- **Non-Background Confidence**: Computes confidence $w_{i,m} = \max_{k \ge 1} p_{i,m,k}$ specifically across human activity classes ($1..6$), excluding Class 0 (NO_PERSON). This prevents unconfident target slots from falsely reporting high confidence due to background collapse.
- **Calibrated Gating ($\tau_{\text{conf}} = 0.15$)**: Samples with non-background confidence exceeding $0.15$ contribute to class occurrence statistics.
- **EMA Tracking**: Updates running prevalence $\mu_k \leftarrow \beta \mu_k + (1-\beta) \text{freq}_k$ ($K+1=7$ classes).
- **Dynamic Mask ($\gamma_k$)**: Class 0 is always retained ($\gamma_0 = 1.0$). Activity classes are filtered via $\gamma_k = \mathbb{I}(\mu_k \ge 0.10 \cdot \max_{j \ge 1} \mu_j)$, effectively zeroing out absent classes under partial domain shift.

**`compute_occupancy(probs)`** → `p_occ [B]`:
- High `p_occ` = room is occupied ($1 - \text{mean}(p_{\text{NO\_PERSON}})$).

---

### 3.11 Loss Engine — `src/adaptation/loss_engine.py`

**What it does**: Evaluates the unified CAPA-Fi unsupervised adaptation objective:

```python
engine = CAPAFiLossEngine(lambda_ent=1.0, lambda_div=0.25, lambda_rot=0.30)
loss_dict = engine(
    logits=logits,
    rot_logits=rot_logits,
    rot_targets=rot_targets,
    category_mask=gamma_k,
    occupancy_weight=p_occ,
)
```

**Loss Components**:
1. **Category-Masked Conditional Entropy ($L_{\text{ent}}$)**: Masks out absent source categories ($\gamma_k = 0$) and re-normalizes slot probability distributions before computing entropy. This prevents negative transfer from forcing predictions into absent classes.
2. **Marginal Diversity ($L_{\text{div}}$)**: Maximizes marginal entropy of batch-averaged predictions across all classes, gated by multi-user occupancy $p_{\text{occ}}$.
3. **Rotation Self-Supervised Loss ($L_{\text{rot}}$)**: Auxiliary cross-entropy on subcarrier frequency inversion pretext task ($0^\circ$ vs $180^\circ$).

**Total Loss Formulation**:
$$\mathcal{L} = \lambda_{\text{ent}} \mathcal{L}_{\text{ent}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}} + \lambda_{\text{rot}} \mathcal{L}_{\text{rot}}$$

---

### 3.12 Adaptation Trainer — `src/adaptation/trainer.py`

**What it does**: The orchestration engine for target adaptation.

```python
trainer = TargetAdaptationTrainer(model=model, config=cfg)

# Run one epoch
epoch_loss = trainer.adapt_epoch(target_loader)

# Run full adaptation
history = trainer.adapt(
    target_loader, epochs=30,
    save_path="checkpoints/adapted_model.pt",
    results_path="results/adaptation_history.json"
)
```

**Guarantees enforced on init**:
- `model.slot_head.requires_grad = False` (classifier frozen)
- `model.backbone.requires_grad = True`
- `model.rot_head.requires_grad = True`
- Optimizer only sees trainable parameters

**`EpochMetrics`**: A `float` subclass that also exposes a dictionary of per-component losses via `epoch_loss["loss_entropy"]` etc.

**`CAPAFiAdaptationEngine`**: Alias for `TargetAdaptationTrainer` (same class).

---

### 3.13 Run Script — `src/adaptation/run_adaptation.py`

**What it does**: CLI entry point for the full adaptation pipeline.

```bash
python -m src.adaptation.run_adaptation
```

Steps:
1. Load `configs/default_config.yaml`
2. Try to load `splits/target_split.csv` + real `.npy` files (fall back to synthetic if missing)
3. Try to load `checkpoints/source_model.pt` (fall back to `MockHARModel` if missing)
4. Run `CAPAFiAdaptationEngine.adapt(epochs=30)`
5. Verify classifier is bitwise frozen
6. Save `checkpoints/adapted_model.pt` + `results/adaptation_history.json`

---

### 3.14 Evaluation Metrics — `src/evaluation/metrics.py`

**What it does**: Computes multi-user activity metrics using Hungarian slot alignment.

Because the model outputs an unordered set of predictions across $M=6$ slots, standard accuracy would falsely penalize correct predictions placed in different slot orders. This module first solves optimal bipartite matching, then computes:

1. **`match_slots_hungarian(logits, target_slots)`**: Permutes predicted slots to match ground truth order using minimum negative log-likelihood cost.
2. **`compute_slot_wise_accuracy(pred, target)`**: Classification accuracy across all $B \times M$ slots (or active slots only).
3. **`compute_occupancy_mae(pred, target)`**: Mean Absolute Error between predicted number of concurrent people vs. ground truth.
4. **`compute_macro_f1(pred, target)`**: Unweighted class-balanced F1 score.
5. **`compute_exact_match_ratio(pred, target)`**: Strict sample-level accuracy where all $M$ slots must be correct.
6. **`evaluate_predictions(logits, target_slots)`**: Convenient wrapper returning all five metrics in a clean dictionary.

---

### 3.15 Benchmarking & Visualizations — `src/evaluation/benchmark.py`

**What it does**: Evaluates adapted models against baselines under partial category shifts and generates publication figures and tables.

```bash
python -m src.evaluation.benchmark --protocol partial_75 --samples 64
```

**Protocols Supported**:
- `closed_set`: All 6 classes active in source and target.
- `partial_75`: Target has 4 of 6 classes (`Wave` and `Pick-up` absent).
- `partial_50`: Target has 3 of 6 classes (`Fall`, `Wave`, `Pick-up` absent).

**Visualizations & Outputs Saved in `results/`**:
- `benchmark_{protocol}.png`: Consolidated publication-ready figure containing 4 subplots (Confusion matrix heatmap, Paradigm comparison bar chart, Occupancy MAE comparison, Convergence loss curves & support distribution).
- `benchmark_{protocol}.json`: Machine-readable telemetry metrics (Slot Acc, Active Acc, Macro F1, Occ MAE, Exact Match).
- Comparison table is printed directly to the terminal stdout for immediate inspection without generating auxiliary LaTeX files.

---

## 4. Order of Execution & How to Run the Project

The CAPA-Fi lifecycle executes in four sequential phases:

```text
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   PHASE 1: PRE-FLIGHT   │     │ PHASE 2: SOURCE TRAIN   │     │ PHASE 3: TARGET ADAPT   │     │   PHASE 4: EVALUATION   │
│ Environment & Unit Test │ ──► │ Supervised Pre-Training │ ──► │ Source-Free Adaptation  │ ──► │ Benchmark & Performance │
│ pytest tests/           │     │ Hungarian Match + SSL   │     │ CAPA-Fi Masking (γ_k)   │     │ Metrics & Visualizers   │
└─────────────────────────┘     └────────────┬────────────┘     └────────────┬────────────┘     └─────────────────────────┘
                                             │                               │
                                             ▼                               ▼
                                  checkpoints/source_model.pt     checkpoints/adapted_model.pt
                                                                  results/adaptation_history.json
```

---

### 4.1 Phase 1: Setup & Pre-Flight Validation

Before executing training or adaptation, verify that the environment is sound and all 67 tensor contracts are satisfied.

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Full Test Suite**:
   ```bash
   pytest tests/test_capa_fi.py -v
   ```
   * **Verification Criteria**: All 67 tests must pass (runs in ~18 seconds). This ensures data loaders, CSI augmentation pipelines, 2D-CNN + BiLSTM backbones, Hungarian matchers, and confidence masking engines are operating within specification.

---

### 4.2 Phase 2: Source Domain Pre-Training

Trains the feature encoder $f_\theta$, multi-slot classifier $g_\theta$, and auxiliary rotation head $h_\psi$ on the labeled Source Domain (Room 1, 5 GHz).

* **Input Data**: `splits/source_split.csv` and `data/processed/WiMANS/*.npy` (falls back to synthetic data if `.npy` files are not on disk).
* **Supervision**: Hungarian bipartite matching resolves unordered user permutations; 180° rotation SSL reinforces spatial feature invariance.
* **Output Artifacts**: `checkpoints/source_model.pt` and `checkpoints/source_model_final.pt`.

#### Execution Commands:

1. **Full Pre-Training (Recommended for GPU / CUDA)**:
   ```bash
   python -m src.models.train_source --config configs/default_config.yaml
   ```
   * Trains for 50 epochs over 9,030 source samples with batch size 32.

2. **Fast Local CPU Smoke Test (~1 minute)**:
   ```bash
   python -m src.models.train_source --batch_size 4 --limit_batches 2 --epochs 1
   ```
   * Use this to quickly verify end-to-end model checkpoint generation without running full multi-hour CPU training.

---

### 4.3 Phase 3: Source-Free Target Adaptation (CAPA-Fi)

Adapts the pre-trained encoder to the unseen Target Domain (Room 2, 2.4 GHz) **without requiring any target ground-truth labels** and **under partial category shift** ($\mathcal{Y}_t \subset \mathcal{Y}_s$).

* **Input Data**: `splits/target_split.csv` and target `.npy` streams.
* **Input Model**: `checkpoints/source_model.pt` (automatically initializes architectural mock if pre-training was skipped).
* **Adaptation Logic**:
  1. **Freezes** classifier $g_\theta$ (`requires_grad = False`).
  2. Evaluates conditional entropy $\mathcal{L}_{ent}$ and multi-user occupancy $p_{occ}$.
  3. Computes the **Category-Aware Dynamic Filtering Mask ($\gamma_k$)** to suppress diversity loss on absent source categories, stopping negative transfer.
  4. Optimizes backbone $f_\theta$ via AdamW.
* **Output Artifacts**: `checkpoints/adapted_model.pt` and `results/adaptation_history.json`.

#### Execution Commands:

1. **Full Adaptation Pipeline**:
   ```bash
   python -m src.adaptation.run_adaptation --config configs/default_config.yaml
   ```
2. **Fast Test with Mock Inputs**:
   ```bash
   python -m src.adaptation.run_adaptation --mock --epochs 3 --batch_size 2
   ```

* **Verification Gate**: The run automatically asserts that the classifier head weights remain **bitwise identical** before and after adaptation (`[PASS] Classifier weights are bitwise identical`).

---

### 4.4 Phase 4: Target Evaluation & Benchmarking

Evaluates model performance across three partial domain shift regimes:
1. `closed_set`: Full class overlap $\mathcal{Y}_t = \{1, 2, 3, 4, 5, 6\}$
2. `partial_75`: 75% overlap $\mathcal{Y}_t = \{1, 2, 3, 4\}$ (Classes 5:Wave and 6:Pick-up absent)
3. `partial_50`: 50% overlap $\mathcal{Y}_t = \{1, 2, 3\}$ (Classes 4:Fall, 5:Wave, and 6:Pick-up absent)

#### Execution Commands:

1. **Standard Partial Shift Benchmark (75% Overlap — Primary Protocol)**:
   ```bash
   python -m src.evaluation.benchmark --protocol partial_75
   ```

2. **Severe Partial Shift Benchmark (50% Overlap)**:
   ```bash
   python -m src.evaluation.benchmark --protocol partial_50
   ```

3. **Closed-Set Benchmark (100% Overlap)**:
   ```bash
   python -m src.evaluation.benchmark --protocol closed_set
   ```

4. **Fast Smoke Test (~5 seconds)**:
   ```bash
   python -m src.evaluation.benchmark --protocol partial_75 --samples 16
   ```

#### CLI Options Reference:
* `--protocol <str>`: Domain shift protocol (`closed_set`, `partial_75`, or `partial_50`, default: `partial_75`).
* `--checkpoint <str>`: Path to model checkpoint to evaluate (default: `checkpoints/adapted_model.pt`).
* `--config <str>`: Path to YAML config file (default: `configs/default_config.yaml`).
* `--samples <int>`: Number of evaluation samples (default: `64`).

#### Output & Display:
* **Terminal Display**: Prints formatted ASCII comparison table directly in terminal output (comparing SHOT, WiFi-SFGADR, SAN, Partial-SFUDA, MU-SHOT-Fi, and CAPA-Fi).
* `results/benchmark_{protocol}.png`: Consolidated 2x2 subplot figure (Confusion Matrix, Paradigm Comparison, Occupancy MAE, Convergence/Support).
* `results/benchmark_{protocol}.json`: Machine-readable telemetry metrics (Slot Acc, Active Acc, Macro F1, Occ MAE, Exact Match).
* `--export-latex` (optional flag): If passed, exports `results/benchmark_table_{protocol}.tex`.

#### Programmatic Usage in Python:
```python
from src.evaluation import evaluate_predictions, run_benchmark

# 1. Direct evaluation on model output logits:
metrics = evaluate_predictions(logits=logits, target_slots=target_slots)
print(f"Slot Accuracy: {metrics['slot_wise_accuracy']*100:.1f}%")
print(f"Occupancy MAE: {metrics['occupancy_mae']:.3f}")

# 2. Run benchmark on checkpoint:
results = run_benchmark(protocol="partial_75", model_checkpoint="checkpoints/adapted_model.pt")
```

---

## 5. How to Run the Tests

```bash
# All tests
pytest tests/test_capa_fi.py -v

# Specific module only
pytest tests/test_capa_fi.py -v -k "Backbone"
pytest tests/test_capa_fi.py -v -k "Adaptation"
pytest tests/test_capa_fi.py -v -k "Permutation"

# Quick smoke (no tiny overfit — slow)
pytest tests/test_capa_fi.py -v -k "not tiny_overfit"
```

Tests that require real CSV/NPY files are automatically skipped if the data is not present.

---

## 6. Mental Model: One Forward Pass

```
x = torch.randn(4, 3, 3000, 30)   # 4 CSI samples

# Source training forward:
x_aug = CSIAugmentor()(x)                       # augment
x_rot = torch.flip(x_aug, dims=[-2, -1])        # 180° flip
x_combined = torch.cat([x_aug, x_rot], dim=0)  # [8, 3, 3000, 30]

logits_all, z_all = model(x_combined)           # backbone + classifier
logits = logits_all[:4]                         # [4, 6, 7] -- original only
rot_logits = model.forward_rotation(z_all)      # [8, 2] -- all samples

loss = SourceTrainingLoss(0.5)(logits, target_slots, rot_logits, rot_labels)
loss["total"].backward()
optimizer.step()

# Adaptation forward (no labels):
logits, z = model(x_target)                      # [B, 6, 7], [B, 256]
x_freq_inv = torch.flip(x_target, dims=[-1])    # Subcarrier-only flip (NOT dims=[-2,-1])
_, z_freq_inv = model(x_freq_inv)
rot_0   = model.forward_rotation(z)             # [B, 2]
rot_freq = model.forward_rotation(z_freq_inv)   # [B, 2]

w_i, gamma_k, p_occ = compute_weights_and_masks(logits, mask_engine)
loss_dict = CAPAFiLossEngine()(logits, rot_0, rot_freq, w_i, gamma_k, p_occ)
loss_dict["loss_total"].backward()
optimizer.step()
```

---

## 7. Common Questions

**Q: Why is the classifier frozen during adaptation?**

The classifier (`g_θ`) was trained on source labels and contains the accumulated knowledge of what activities look like. If we update it with no-label entropy signals from the target domain, it would drift and forget source knowledge. By freezing it and only updating the backbone (`f_θ`), we preserve source discrimination while adapting the feature extractor to the new environment.

**Q: Why do we need the Hungarian matcher?**

If we have two users [Walk, Sit], the DataLoader might put Walk in slot 0 and Sit in slot 1. But the model might naturally predict Sit in slot 0 and Walk in slot 1. Standard cross-entropy would penalise this even though both predictions are correct. The Hungarian matcher finds the permutation that minimizes total cost, making the loss truly permutation-invariant.

**Q: What is the rotation SSL trick?**

Frequency inversion (subcarrier-axis flip: `dims=[-1]`) is a synthetic label we can create from any unlabeled sample. Original → label 0, freq-inverted → label 1. The model must learn to distinguish them. This forces the backbone to maintain meaningful spectral representations even when there are no activity labels, preventing the backbone from collapsing during adaptation.

**Q: Why do we flip only the subcarrier axis (`dims=[-1]`) and NOT the time axis (`dims=[-2]`)?**

This is a BiLSTM architecture constraint. A Bidirectional LSTM computes both a forward sequence (`h_forward_t`) and a backward sequence (`h_backward_t`) at every timestep. When you flip the time axis of the input, the BiLSTM’s forward pass on the reversed sequence traces *exactly* the backward pass on the original sequence:

```
h_forward_t(x_reversed) ≈ h_backward_t(x_original)
```

After global temporal mean-pooling over T=375 steps, the resulting embedding `z` for the flipped input is nearly *identical in magnitude* to the original — just with some feature dimensions permuted. The rotation head receives inputs that are effectively indistinguishable, so it outputs uniform 50/50 logits: **loss stays stuck at `ln(2) ≈ 0.693` forever** (random guessing).

Flipping only the OFDM subcarrier ordering (`dims=[-1]`) reverses the frequency-bin layout (high subcarriers become low and vice versa) without touching the time axis. The CNN spatial blocks process this as a genuinely different spectral fingerprint, giving the rotation head a real discriminative signal to learn from.

**Q: What does the dual learning rate optimizer do?**

During source training, the backbone and classifier use LR=1e-3 while the rotation head uses LR=1e-2 (10x higher). This is necessary because the rotation head is initialized from scratch and its gradients are much weaker than the matched cross-entropy signal driving the backbone. A shared single LR starves the rotation head and keeps its loss at 0.693. The dual-group `AdamW` setup ensures the rotation head adapts at the right speed.

**Q: What happens when `γ_k → 0` for a class?**

When `γ_k` is near 0 for class `k`, the diversity loss term for class `k` is essentially zeroed out. This means the model won’t be penalised for ignoring class `k` (which doesn’t appear in the target domain anyway). Without this mask, the model would be pushed to predict class `k` uniformly which would cause negative transfer.

**Q: Why is `p_occ` multiplied into the diversity loss?**

If the target environment has empty rooms, the average probability of NO_PERSON is high and `p_occ` is low. Multiplying the diversity loss by `p_occ` reduces its magnitude for empty-room batches — we don’t want to force activity diversity when nobody is there.

**Q: What is `MockHARModel` used for?**

It’s a lightweight stand-in for the full `HARModel` that runs faster in tests and doesn’t require the full CSI input shape. It has the same external API (`forward`, `forward_rotation`, `freeze_classifier`, `slot_head`, `rot_head`) so the adaptation engine can be tested independently of the full model.

---

## 8. Performance Fixes Applied

### 8.1 Rotation SSL Stagnation Fix (Rot Loss stuck at 0.693)

| | Before fix | After fix |
|---|---|---|
| **Flip axis** | `dims=[-2, -1]` (time + subcarrier) | `dims=[-1]` (subcarrier only) |
| **Expected rot loss** | ~0.693 (random guessing) | ~0.4–0.5 (converging) |
| **Root cause** | BiLSTM temporal symmetry under global mean-pool | Fixed by using spectral-only inversion |

**Files changed:**
- [`src/models/train_source.py`](../src/models/train_source.py) — `create_rotation_pair()`: `dims=[-2,-1]` → `dims=[-1]`
- [`src/adaptation/trainer.py`](../src/adaptation/trainer.py) — `x_target_180 = flip(...)`: `dims=[-2,-1]` → `dims=[-1]`

### 8.2 Dual Learning Rate for Rotation Head

The rotation head now has an independent AdamW parameter group at `lr = 1e-2` (10x the backbone’s 1e-3):

```python
# In train_source.py
optimizer = AdamW([
    {"params": base_params,    "lr": 1e-3},   # backbone + classifier
    {"params": rot_head_params, "lr": 1e-2},  # rotation head (10x)
])
```

Configurable in `configs/default_config.yaml`:
```yaml
source_training:
  learning_rate: 1.0e-3           # backbone + classifier
  learning_rate_rot_head: 1.0e-2  # rotation head (10x backbone)
```

### 8.3 Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` added before every `optimizer.step()` in:
- Source training loop (`train_one_epoch` in `train_source.py`)
- Target adaptation loop (`adapt_epoch` in `trainer.py` — clipped only on trainable parameters)

This prevents occasional large gradient spikes from the BiLSTM and the Hungarian matching loss from destabilizing training.

### 8.4 Convergence Monitoring Checklist

After re-running training you should observe:

| Metric | Epoch 1 | Epoch 10 | Healthy end |
|--------|---------|----------|-------------|
| `rot_loss` | ~0.50–0.60 | ~0.25–0.35 | <0.20 |
| `ce_loss` | ~1.80–2.00 | ~0.60–0.90 | <0.50 |
| `total` | ~2.2–2.5 | ~0.85–1.2 | <0.70 |

If `rot_loss` stays above 0.60 after 5 epochs, check:
1. That `create_rotation_pair` uses `dims=[-1]` (not `[-2, -1]`)
2. That `learning_rate_rot_head` is set and at least 5x the backbone LR
3. Run the diagnostic: `python -c "from src.models.train_source import create_rotation_pair; import torch; x=torch.randn(2,3,3000,30); xc,_,_=create_rotation_pair(x); print((x[0]-xc[2]).abs().mean())"` — should print a value > 0.5
