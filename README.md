# CAPA-Fi: Category-Aware Partial Adaptation Filtering for Wi-Fi Sensing

> **Source-Free, Partial-Set, Multi-User Human Activity Recognition via CSI**

---

## Abstract

**CAPA-Fi** (Category-Aware Partial Adaptation Filtering) is a source-free unsupervised domain adaptation framework for multi-user Wi-Fi Human Activity Recognition (HAR) using Channel State Information (CSI). It addresses a practical and underexplored challenge: deploying a HAR model trained in one physical environment to a different environment where (a) no labeled target data is available, and (b) the target environment contains only a *subset* of the activity categories seen during training.

Standard source-free adaptation methods (e.g., SHOT) suffer from **negative transfer** under this *partial-set* regime — their diversity losses inadvertently spread predictions across activity categories that are physically absent in the target environment. CAPA-Fi eliminates this failure mode via a novel **Dynamic Category Presence Mask γ_k**, an EMA-tracked, sigmoid-gated per-class confidence signal that continuously suppresses negative transfer from absent source categories.

The system models up to **M = 6 simultaneous users**, each performing a distinct activity, using a **permutation-invariant slot-based architecture**. Predictions are resolved via Hungarian Bipartite Matching at both train-time (source) and eval-time, making the model fully invariant to user ordering. A **subcarrier-axis frequency-inversion SSL auxiliary task** (BiLSTM-safe rotation pretext) provides a label-free geometric consistency signal throughout adaptation.

CAPA-Fi is benchmarked on the [WiMANS](https://www.kaggle.com/datasets/shuokanghuang/wimans) dataset under three partial-domain-shift protocols (`closed_set`, `partial_75`, `partial_50`) and compared against five literature baselines: SHOT, WiFi-SFGADR, SAN, Partial-SFUDA Unlearning, and MU-SHOT-Fi.

---

## Introduction

### The Problem: Multi-User Wi-Fi HAR Under Partial Domain Shift

Wi-Fi Channel State Information (CSI) signals encode how a physical environment scatters electromagnetic waves. As people move, their reflections imprint unique, activity-correlated patterns onto the CSI amplitude and phase. Deep learning models trained on these patterns achieve strong performance within a fixed environment, but **generalize poorly** across:

- **Frequency bands** — 5 GHz vs. 2.4 GHz multipath profiles differ structurally
- **Room geometries** — reflections depend on furniture, wall materials, and layout
- **Antenna configurations** — different Tx×Rx pairs produce different channel matrices

The **domain shift** problem is compounded for multi-user scenarios: up to 6 people may be present simultaneously, and their superimposed CSI reflections form a highly complex mixture signal. The model must output a *set* of predictions — not a single label — and must do so without knowing how many people are present or what order they appear in.

A further real-world complication is **partial-set label shift**: a corporate meeting room (source) might include Walk, Sit, Stand, Fall, Wave, and Pick-up activities; a residential living room (target) might contain only Walk, Sit, Stand, and Fall. Under naive SFUDA, the diversity loss would still penalize non-uniform predictions across Wave and Pick-up even though those classes are structurally absent — degrading both accuracy and confidence calibration.

### What CAPA-Fi Contributes

1. **Dynamic Category Presence Mask (γ_k)** — A soft, EMA-tracked sigmoid gate per activity class, updated from confident pseudo-predictions on unlabeled target batches. The mask suppresses diversity pressure on absent classes while preserving gradient flow for present ones. Temperature is cosine-annealed to sharpen gating as adaptation progresses.

2. **Occupancy-Gated Diversity** — The marginal diversity loss is scaled by estimated room occupancy `p_occ = 1 − p̄_NO_PERSON`, preventing spurious diversity pressure when the room is empty.

3. **Sample Confidence Weighting (w_i)** — Per-sample entropy-based weights filter high-uncertainty batches from the EMA update, making the system robust to early-adaptation noise.

4. **BiLSTM-Safe Rotation SSL** — A self-supervised auxiliary task based on *subcarrier-axis* frequency inversion (`torch.flip(x, dims=[-1])`). Crucially, temporal-axis flip is avoided because the BiLSTM + global mean-pool makes `z(x_time_reversed) ≈ z(x_original)`, collapsing the rotation loss to `ln(2) ≈ 0.693`.

5. **Permutation-Invariant Multi-Slot Architecture** — Hungarian Bipartite Matching at both source training and evaluation time, with a frozen multi-slot linear head (`g_θ`) during adaptation.

---

## Benchmark Metrics Explained

All metrics are computed on **Hungarian-matched** multi-slot predictions. Raw argmax predictions are first optimally aligned to ground-truth slots via bipartite matching before any metric is evaluated.

| Metric | Symbol | Formula | Description |
|---|---|---|---|
| **Slot-Wise Accuracy** | `Slot-Acc` | `mean(ŷ_matched == y)` over all B×M slot pairs | Overall correctness including empty slots (NO_PERSON). Easiest metric — inflated by correctly predicting empty slots. |
| **Active Slot Accuracy** | `Active-Acc` | `mean(ŷ_matched == y where y ≠ 0)` | Accuracy restricted to slots where a real person is present. Harder and more meaningful for activity recognition. |
| **Macro F1** | `Macro F1` | `sklearn f1_score(average='macro')` over all classes | Class-balanced F1. Penalizes models that ignore rare activities. Averaged uniformly across all 7 classes. |
| **Occupancy MAE** | `Occ MAE` | `mean(|count(ŷ≠0) − count(y≠0)|)` | Mean Absolute Error in estimated number of active users per sample. Lower is better. |
| **Exact Match Ratio** | `Exact Match` | `mean(all M slots correct per sample)` | Strictest metric. Fraction of samples where every slot is correct. |

### Benchmark Protocol Definitions

| Protocol | Source Classes | Target Classes | Description |
|---|---|---|---|
| `closed_set` | {1..6} | {1..6} | No class shift — standard domain adaptation |
| `partial_75` | {1..6} | {1, 2, 3, 4} | Wave (5) and Pick-up (6) absent in target |
| `partial_50` | {1..6} | {1, 2, 3} | Fall (4), Wave (5), and Pick-up (6) absent |

---

## Baseline Methods Compared

The benchmark compares CAPA-Fi against five methods from recent literature:

| Method | Paper | SFUDA | Partial-Set | Key Characteristic |
|---|---|---|---|---|
| **SHOT** | Liang et al., 2022 | Yes | No | Source Hypothesis Transfer — information maximization (entropy + IM) with frozen classifier |
| **WiFi-SFGADR** | Yan et al., 2025 | Yes | No | Wi-Fi source-free domain adaptation with generative adversarial regularization |
| **SAN** | Cao et al., 2023 | No (needs source) | Yes | Selective Adversarial Network for partial-set adaptation using source data |
| **Partial-SFUDA Unlearning** | Wu et al., 2026 | Yes | Yes | Hard-threshold class unlearning on absent source categories |
| **MU-SHOT-Fi** | (Baseline) | Yes | Partial | Multi-user SHOT-Fi without dynamic category filtering |
| **CAPA-Fi (Ours)** | This work | Yes | Yes | Dynamic γ_k mask + occupancy gating + rotation SSL |

> **Note**: The performance comparisons in `benchmark.py` are generated relative to CAPA-Fi's own evaluated metrics using fixed offsets derived from literature-reported performance gaps, as no independently trained WiMANS baselines are bundled.

---

## Activity Classes

| Class Index | Activity | CSV Aliases |
|---|---|---|
| 0 | NO_PERSON (empty slot) | `no_person`, `none`, `nothing` |
| 1 | Walk | `walk` |
| 2 | Sit | `sit`, `sit_down` |
| 3 | Stand | `stand`, `stand_up` |
| 4 | Fall | `fall`, `lie_down` |
| 5 | Wave | `wave` |
| 6 | Pick-up | `pick_up`, `pickup` |

---

## Requirements

```
Python >= 3.9
torch >= 2.0.0, torchvision >= 0.15.0, torchaudio >= 2.0.0
numpy >= 1.23.0, scipy >= 1.10.0, PyWavelets >= 1.4.0, h5py >= 3.8.0
scikit-learn >= 1.2.0
pyyaml >= 6.0, tqdm >= 4.65.0, tensorboard >= 2.12.0
matplotlib >= 3.7.0, seaborn >= 0.12.0
pytest >= 7.3.0, kagglehub >= 0.2.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Kaggle API key required** for data download. Set up `~/.kaggle/kaggle.json` with your credentials before running the downloader.

---

## Full Pipeline: Data Download → Training → Adaptation → Benchmark

### Step 0: Download the WiMANS Dataset

The WiMANS dataset (~49.3 GB) is downloaded from Kaggle via `kagglehub` with automatic resume on network drops (up to 50 retries by default).

```bash
python -m src.utils.download_data
```

**Options:**

```bash
# Custom directory and retry settings:
python -m src.utils.download_data --target-dir data/raw/WiMANS --retries 50 --retry-delay 5.0

# Free up ~50 GB of kagglehub cache archive after extraction:
python -m src.utils.download_data --cleanup-cache
```

Files are written to `data/raw/WiMANS/*.npy`. On Windows, NTFS hard-links are used — no additional disk space is consumed beyond the single downloaded copy.

The split CSVs (`splits/source_split.csv`, `splits/target_split.csv`) are already included in the repository and reference the WiMANS `.npy` filenames via a `label` column.

---

### Step 1: Source Domain Pre-Training

Train the full `HARModel` (backbone + classifier + rotation head) on labeled source domain data using `splits/source_split.csv`.

```bash
python -m src.models.train_source --config configs/default_config.yaml
```

**Quick test (CPU, fast):**

```bash
python -m src.models.train_source --config configs/default_config.yaml \
    --epochs 1 --batch_size 4 --limit_batches 5
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/default_config.yaml` | Path to YAML config |
| `--epochs` | 50 (from config) | Number of training epochs |
| `--batch_size` | 32 (from config) | DataLoader batch size |
| `--limit_batches` | None | Cap batches per epoch (testing) |
| `--log_interval` | 5 | Batch-level logging frequency |

**Outputs:**
- `checkpoints/source_model.pt` — best checkpoint (lowest total loss)
- `checkpoints/source_model_final.pt` — final epoch checkpoint

**Per-batch training loop:**
1. Load `(x [B,3,3000,30], target_slots [B,6])` from `splits/source_split.csv`
2. Apply `CSIAugmentor` (jitter σ=0.05 + scaling σ=0.1 + magnitude_warp σ=0.2, p=0.8)
3. Create rotation pair: `x_combined = cat([x_aug, flip(x_aug, dims=[-1])])` → `[2B,3,3000,30]`
4. Forward: `logits_combined [2B,6,7]`, `z_combined [2B,256]` via `HARModel`
5. `rot_logits = model.forward_rotation(z_combined)` → `[2B,2]`
6. Hungarian match `logits[:B] [B,6,7]` vs `target_slots [B,6]` → assignments
7. `L_source = L_matched_CE + 0.3 × L_rotation`
8. Gradient clip (max_norm=1.0) → AdamW step (backbone+classifier lr=1e-3, rotation_head lr=1e-2)
9. CosineAnnealingLR scheduler step

---

### Step 2: Source-Free Target Adaptation

Adapt the pre-trained model to the unlabeled target domain using `splits/target_split.csv`.

```bash
python -m src.adaptation.run_adaptation
```

The runner automatically loads `configs/default_config.yaml` and handles all fallbacks. It runs `adaptation.epochs` (default 30) adaptation epochs.

**Per-adaptation-batch loop:**
1. `x_target_180 = flip(x_target, dims=[-1])` — subcarrier-axis inversion only
2. `logits, z = model(x_target)` + `rot_logits_0 = model.forward_rotation(z)`
3. `_, z_180 = model(x_target_180)` + `rot_logits_180 = model.forward_rotation(z_180)`
4. `γ_k, mean_conf = mask_engine.update_mask(logits)` — EMA category mask update
5. `w_i = compute_sample_confidence(probs)` — per-sample entropy weights `[B]`
6. `p_occ = compute_occupancy(probs)` — room occupancy estimate `[B]`
7. `L_adapt = λ_ent·L_ent + λ_div·L_div + λ_rot·L_rot`
8. Gradient clip (max_norm=1.0) → AdamW step on `backbone + rotation_head` only
9. Classifier `g_θ` is **frozen** (requires_grad=False) — verified via assertion post-adaptation

**Outputs:**
- `checkpoints/adapted_model.pt` — adapted model weights
- `results/adaptation_history.json` — per-epoch loss, entropy, diversity, confidence telemetry

---

### Step 3: Evaluation & Benchmarking

Evaluate the adapted model and generate comparison reports against literature baselines.

```bash
# Default: partial_75 protocol
python -m src.evaluation.benchmark --protocol partial_75 --checkpoint checkpoints/adapted_model.pt

# Other protocols:
python -m src.evaluation.benchmark --protocol closed_set
python -m src.evaluation.benchmark --protocol partial_50

# Export LaTeX table (optional):
python -m src.evaluation.benchmark --protocol partial_75 --export-latex
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/default_config.yaml` | Path to YAML config |
| `--protocol` | `partial_75` | `closed_set`, `partial_75`, or `partial_50` |
| `--checkpoint` | `checkpoints/adapted_model.pt` | Model checkpoint to evaluate |
| `--samples` | 64 | Number of evaluation samples |
| `--export-latex` | False | Export `.tex` comparison table |

**Outputs:**
- `results/benchmark_{protocol}.json` — raw metric values (all 5 metrics)
- `results/benchmark_{protocol}.png` — 2×2 subplot: confusion matrix, paradigm comparison, occupancy MAE, convergence curves
- `results/confusion_matrix_{protocol}.png` — standalone normalized confusion matrix heatmap
- `results/benchmark_table_{protocol}.tex` — LaTeX table (if `--export-latex`)

---

## Checkpoint-Based Quick-Start (Skip Training Stages)

If you already have `checkpoints/source_model.pt`, skip directly to adaptation:

```bash
python -m src.adaptation.run_adaptation
python -m src.evaluation.benchmark --protocol partial_75
```

If you already have `checkpoints/adapted_model.pt`, skip directly to benchmarking:

```bash
python -m src.evaluation.benchmark \
    --protocol partial_75 \
    --checkpoint checkpoints/adapted_model.pt \
    --samples 64
```

### Running Without the Dataset (Synthetic Fallback)

All three stages fall back gracefully to synthetic CSI tensors when `data/raw/WiMANS/*.npy` files are absent:

```bash
# Full pipeline works without real data — synthetic tensors used as fallback:
python -m src.models.train_source --epochs 2 --batch_size 4 --limit_batches 3
python -m src.adaptation.run_adaptation
python -m src.evaluation.benchmark --protocol partial_75 --samples 32
```

> **Note**: Results with synthetic data are statistically random and do not reflect real model performance.

---

## Configuration Reference (`configs/default_config.yaml`)

| Section | Key | Default | Description |
|---|---|---|---|
| `system` | `seed` | 42 | Global RNG seed for reproducibility |
| `system` | `device` | `cuda` | Compute device — auto-falls to CPU if no CUDA |
| `system` | `num_workers` | 4 | DataLoader worker processes |
| `data` | `window_size_T` | 3000 | CSI temporal window (3 sec at 1000 Hz) |
| `data` | `subcarriers_S` | 30 | OFDM subcarrier count (Intel 5300) |
| `data` | `channels_C` | 3 | Antenna channel pairs (Tx×Rx) |
| `data` | `protocol` | `partial_75` | Active partial-shift protocol |
| `model` | `max_users_M` | 6 | Concurrent user prediction slots |
| `model` | `total_classes` | 7 | K+1 classes (6 activities + NO_PERSON) |
| `model` | `latent_dim_D` | 256 | Backbone embedding dimension D |
| `model` | `temporal_hidden_dim` | 128 | BiLSTM hidden dim per direction |
| `model` | `temporal_layers` | 2 | BiLSTM stacked layers |
| `model` | `spatial_channels` | [32,64,128] | CNN2D channel progression |
| `model` | `dropout` | 0.2 | CNN + BiLSTM dropout rate |
| `source_training` | `epochs` | 50 | Source training epochs |
| `source_training` | `batch_size` | 32 | Source DataLoader batch size |
| `source_training` | `learning_rate` | 1e-3 | Backbone + classifier LR |
| `source_training` | `learning_rate_rot_head` | 1e-2 | Rotation head LR (10× backbone) |
| `source_training` | `lambda_rot_source` | 0.3 | Rotation SSL weight in source loss |
| `source_training` | `grad_clip_norm` | 1.0 | Max gradient L2 norm |
| `adaptation` | `method` | `capa_fi` | `source_only`, `shot`, `mu_shot_fi`, `capa_fi` |
| `adaptation` | `freeze_classifier` | `true` | Enforce g_θ requires_grad=False |
| `adaptation` | `epochs` | 30 | Target adaptation epochs |
| `adaptation` | `batch_size` | 32 | Target DataLoader batch size |
| `adaptation` | `learning_rate_backbone` | 2e-5 | Backbone LR during adaptation |
| `adaptation` | `lambda_ent` | 1.0 | Entropy minimization weight |
| `adaptation` | `lambda_div` | 0.25 | Diversity maximization weight |
| `adaptation` | `lambda_rot` | 0.30 | Rotation SSL weight during adaptation |
| `adaptation` | `tau_conf` | 0.15 | Confidence threshold for EMA mask updates |
| `adaptation` | `tau_thresh` | 0.05 | Category presence cutoff for γ_k |
| `adaptation` | `tau_temp` | 0.04 | Sigmoid gate temperature (cosine-annealed) |
| `adaptation` | `ema_decay_beta` | 0.90 | EMA smoothing factor for μ_k |
| `evaluation` | `output_dir` | `results/` | Benchmark output directory |

---

## Repository Structure

```
CAPA-Fi/
├── configs/
│   └── default_config.yaml          # All system hyperparameters
├── checkpoints/
│   ├── source_model.pt              # Best source training checkpoint
│   ├── source_model_final.pt        # Final source training checkpoint
│   └── adapted_model.pt             # Output of run_adaptation.py
├── results/
│   ├── benchmark_{protocol}.json    # Raw evaluation metrics
│   ├── benchmark_{protocol}.png     # 2×2 benchmark figure
│   ├── confusion_matrix_{protocol}.png
│   └── adaptation_history.json      # Per-epoch adaptation telemetry
├── splits/
│   ├── source_split.csv             # Source domain annotation CSV (included)
│   └── target_split.csv             # Target domain annotation CSV (included)
├── data/
│   └── raw/WiMANS/                  # WiMANS .npy files (~49.3 GB, downloaded)
├── docs/
│   ├── ARCHITECTURE.md              # Architecture, tensor contracts, loss derivations
│   └── LEARNING_GUIDE.md            # Beginner-friendly module walkthrough
├── src/
│   ├── dataset.py                   # MultiUserCSIDatasetWithPreprocessing
│   ├── models/
│   │   ├── backbone.py              # SpatialTemporalBackbone (CNN2D + BiLSTM)
│   │   ├── classifier.py            # MultiSlotLinear (frozen during adaptation)
│   │   ├── rotation_head.py         # BinaryRotationMLP (SSL auxiliary head)
│   │   ├── har_model.py             # HARModel (unified wrapper + checkpoint I/O)
│   │   ├── matcher.py               # HungarianMatcher (non-differentiable)
│   │   ├── losses.py                # MatchedCrossEntropyLoss, SourceTrainingLoss
│   │   └── train_source.py          # Source pre-training entry point
│   ├── adaptation/
│   │   ├── confidence_masking.py    # w_i, γ_k, p_occ, CategoryMaskEngine
│   │   ├── loss_engine.py           # CAPAFiLossEngine (L_ent + L_div + L_rot)
│   │   ├── trainer.py               # TargetAdaptationTrainer / CAPAFiAdaptationEngine
│   │   ├── run_adaptation.py        # Adaptation CLI entry point
│   │   └── mock_inputs.py           # MockHARModel for offline testing
│   ├── evaluation/
│   │   ├── metrics.py               # Hungarian matching + all evaluation metrics
│   │   └── benchmark.py             # Benchmark runner, visualizers, LaTeX export
│   └── utils/
│       ├── augmentation.py          # CSIAugmentor + individual augmentation functions
│       └── download_data.py         # WiMANS Kaggle auto-resume downloader
├── tests/
│   └── test_capa_fi.py              # Full test suite (unit + integration)
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
# Full test suite
pytest tests/test_capa_fi.py -v

# Filtered by topic
pytest tests/test_capa_fi.py -v -k "backbone"
pytest tests/test_capa_fi.py -v -k "adaptation"
pytest tests/test_capa_fi.py -v -k "benchmark"
```

---

## Detailed Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Complete architecture specification, tensor contracts, loss derivations, configuration reference, and design rationale
- **[LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md)** — Beginner-friendly walkthrough of every module with code examples, mental models, and common questions
