# CAPA-Fi: Category-Aware Partial Adaptation Filtering for Wi-Fi Sensing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Milestone%201%20Ready-success.svg)]()

> **Publication-Grade PyTorch Framework for Source-Free Multi-User Human Activity Recognition (HAR) under Physical Multipath Shifts and Partial Category Mismatch ($\mathcal{Y}_t \subset \mathcal{Y}_s$).**

---

## 1. Overview & Problem Statement

Wi-Fi Channel State Information (CSI) provides rich physical-layer spatial signatures for non-invasive Human Activity Recognition (HAR). However, deploying pre-trained models into unseen target physical environments introduces three structural bottlenecks:

1. **Source-Free & Privacy Guarantee (SFUDA):** Edge sensing devices cannot access or store raw source training data $\mathcal{D}_s$ due to bandwidth limits and strict privacy regulations (GDPR/HIPAA). The model must adapt exclusively from the unlabeled target stream $\mathcal{D}_t$ and frozen source model parameters.
2. **Category Mismatch & Partial Domain Shift ($\mathcal{Y}_t \subset \mathcal{Y}_s$):** Target environments frequently contain an incomplete subset or mismatch of source activity categories. Standard adaptation forces missing source classes onto target predictions, causing catastrophic **negative transfer** and accuracy collapse.
3. **Multi-User Spatial Interference & Edge RAM Constraints:** Concurrent reflections from $M$ co-present users mix non-linearly. Without occupancy conditioning, unsupervised adaptation collapses into dominant "No-Person" background states. Furthermore, commodity hardware imposes strict **4–8 GB host RAM constraints** when streaming multi-gigabyte public benchmarks such as **WiMANS**.

**CAPA-Fi** operates as a **Negative-Transfer Mitigation Engine** under partial category shift. It combines frozen source hypothesis transfer, occupancy-weighted information maximization, 180° rotation self-supervised learning, and a novel **Category-Aware Dynamic Filtering Mask ($\gamma_k$)** with sample-level confidence gating ($w_i$) that suppresses diversity loss on absent source categories without requiring target labels.

---

## 2. System Pipeline & Architecture

```text
                                   TARGET UNLABELED CSI STREAM
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ WiMANS Lazy Streaming │ (RAM <= 8GB)
                                    └───────────┬───────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        Original CSI X (0°)                                   180° Inverted CSI X (180°)
                     │                                                     │
                     ▼                                                     ▼
        ┌─────────────────────────┐                           ┌─────────────────────────┐
        │  Spatial-Temporal f_θ   │                           │  Spatial-Temporal f_θ   │
        │  (Trainable Encoder)    │                           │  (Shared Weights)       │
        └────────────┬────────────┘                           └────────────┬────────────┘
                     │                                                     │
         Latents z ──┴──────────────────────────┐                          │
                     │                          ▼                          ▼
                     ▼               ┌────────────────────┐   ┌─────────────────────────┐
        ┌─────────────────────────┐  │  Rotation Head h_ψ │──►│ Rotation SSL Loss L_rot │
        │  Multi-Slot Head g_θ    │  └────────────────────┘   └─────────────────────────┘
        │  (FROZEN CLASSIFIER)    │
        └────────────┬────────────┘
                     │
                     ▼
        Slot Probabilities P_{m,k}
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐   ┌────────────────────────────────────────────────────────┐
│  Entropy Loss    │   │ Category-Aware Dynamic Filtering Mask γ_k              │
│  L_ent           │   │  - Sample-level confidence weighting w_i               │
│  (Minimization)  │   │  - Gated by slot occupancy p_occ                       │
│                  │   │  - Down-weights diversity on absent source classes     │
└────────┬─────────┘   └──────────────────────────┬─────────────────────────────┘
         │                                        ▼
         │                             ┌────────────────────────────────────────┐
         │                             │ Dynamic Masked Diversity L_CAPA-div    │
         │                             └──────────────────┬─────────────────────┘
         │                                                │
         └───────────────────────┬────────────────────────┘
                                 ▼
                     ┌─────────────────────────┐
                     │ TOTAL CAPA-Fi OBJECTIVE │ ──► Gradient Update on f_θ
                     └─────────────────────────┘
```

---

## 3. Team Member Module Ownership

The repository is modularly structured to reflect the 4 core team responsibilities:

| Module & Directory | Lead Owner | Focus Area | Core Responsibilities |
| :--- | :--- | :--- | :--- |
| **Module 1 (`src/data/`)** | **Abey John Pramod** | CSI Data Engine & Preprocessing | WiMANS lazy streaming under 4–8 GB RAM, DWT de-noising, phase sanitization, multi-room/dual-band split generators, and partial shift split protocols. |
| **Module 2 (`src/models/`)** | **Ashlin Joe** | Permutation-Invariant Set Predictor | Spatial CNN + Temporal Transformer/BiLSTM, $M$-slot output head ($K+1$ classes), Hungarian Bipartite Matching for source pre-training, and rotation SSL auxiliary head $h_\psi$. |
| **Module 3 (`src/adaptation/`)** | **Athishta P. A. [LEAD ROLE]** | Source-Free Adaptation & SSL Engine | Target adaptation loop, frozen classifier logic, standard SHOT $\mathcal{L}_{\text{IM}}$, Occupancy IM $\mathcal{L}_{\text{IM-multi}}$, Rotation SSL $\mathcal{L}_{\text{rot}}$, dynamic confidence weighting $w_i$, and Category Filtering Mask $\gamma_k$. |
| **Module 4 (`src/evaluation/`)** | **Fathima Bushara M. C.** | Partial Shift & Evaluation Analyst | Partial domain shift scenarios ($\mathcal{Y}_t \subset \mathcal{Y}_s$), evaluation metrics suite (Slot-Acc, Occupancy MAE, Macro-F1, Exact Match), baseline comparison runners, and publication visualizers. |

---

## 4. Repository Structure & Documentation Sitemap

```text
CAPA-Fi/
├── docs/                                  # Central consolidated documentation
│   ├── ARCHITECTURE.md                    # Total system architecture, workflow pipeline & methodology
│   └── LEARNING_GUIDE.md                  # Comprehensive educational codebase walkthrough & guide
├── src/                                   # Source code directory
│   ├── dataset.py                         # Module 1: CSI data loading, WiMANS streaming & preprocessing
│   ├── models/                            # Module 2: SpatialTemporalBackbone, multi-slot head, Hungarian matcher, source training
│   ├── adaptation/                        # Module 3: Confidence masking, loss engine, TargetAdaptationTrainer
│   ├── evaluation/                        # Module 4: Hungarian metrics suite, benchmark protocols & visualizers
│   └── utils/                             # Shared CSI augmentation & utilities
├── configs/
│   └── default_config.yaml                # System hyperparameters & execution flags
├── tests/
│   ├── __init__.py
│   └── test_capa_fi.py                    # Consolidated test suite (77 unit & integration tests)
├── checkpoints/                           # Checkpoint storage directory
├── results/                               # Generated figures, tables, and adaptation logs
├── splits/                                # Split CSV definitions (source_split.csv, target_split.csv)
├── requirements.txt                       # Python environment dependencies
└── README.md                              # Root repository landing page
```

### Detailed Documentation Links
* [System Architecture, Workflow Pipeline & Methodology](docs/ARCHITECTURE.md)
* [Comprehensive Codebase Learning Guide](docs/LEARNING_GUIDE.md)

---

## 5. Quickstart Guide

### 5.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/AsherWood39/CAPA-Fi.git
cd CAPA-Fi

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Run Consolidated Test Suite
```bash
# Execute all 77 unit & integration tests
pytest tests/test_capa_fi.py -v
```

### 5.3 Source Model Pre-Training (Supervised Hungarian Matching)
```bash
python -m src.models.train_source --config configs/default_config.yaml
```

### 5.4 Source-Free Target Adaptation (CAPA-Fi)
```bash
python -m src.adaptation.run_adaptation --config configs/default_config.yaml
```

### 5.5 Benchmark Evaluation & Publication Visualizer
```bash
# Evaluate on 75% partial category shift (Primary protocol)
python -m src.evaluation.benchmark --protocol partial_75

# Evaluate on severe partial shift (50% category overlap)
python -m src.evaluation.benchmark --protocol partial_50

# Evaluate on closed-set domain shift (100% overlap)
python -m src.evaluation.benchmark --protocol closed_set
```
* Generates `results/confusion_matrix_{protocol}.png`, `results/tsne_{protocol}.png`, `results/adaptation_loss_curves.png`, `results/benchmark_table_{protocol}.tex`, and `results/benchmark_{protocol}.json`.

---

## 6. Milestone Roadmap

- [x] **Phase 0: Architecture Blueprint & Formal Specifications (Completed)**
  - All 4 documentation deliverables finalized with formal mathematical proofs (Theorems 1, 2, 3) and theoretical grounding (SAN, CPC, Machine Unlearning).
  - Directory outlines initialized without `.py` files to preserve clean team module ownership.
- [x] **Milestone 1 (50% Target):**
  - Member 1: Dataset preprocessing & cleaning pipeline (`src/dataset.py`). `[COMPLETED ✅]`
  - Member 2: Spatial-temporal backbone + $M$-slot head trained via Hungarian bipartite matching. `[COMPLETED ✅]`
  - Member 3: Target adaptation manager with frozen classifier $g_\theta$ and standard SHOT $\mathcal{L}_{\text{IM}}$. `[COMPLETED ✅]`
  - Member 4: Zero-shot unadapted baseline evaluation suite (`src/evaluation/metrics.py`). `[COMPLETED ✅]`
- [x] **Milestone 2 (70% Target):**
  - Member 1: DWT filtering and spatial-temporal tensor reshaping (`src/dataset.py`). `[COMPLETED ✅]`
  - Member 2: Auxiliary rotation head $h_\psi$ and permutation-invariance verification unit tests. `[COMPLETED ✅]`
  - Member 3: Full MU-SHOT-Fi engine ($\mathcal{L}_{\text{IM-multi}} + \mathcal{L}_{\text{rot}}$). `[COMPLETED ✅]`
  - Member 4: Benchmark replication and empirical demonstration of negative transfer under partial category shift. `[COMPLETED ✅]`
- [x] **Milestone 3 (100% Target):**
  - Member 1: Partial domain shift configurations (Closed-set, Partial 75%, Partial 50%). `[COMPLETED ✅]`
  - Member 2: Model export profiling (ONNX / Jetson latency) and latent feature visualizer hooks. `[COMPLETED ✅]`
  - Member 3: Category-Aware Dynamic Filtering Mask ($\gamma_k$) and confidence weighting ($w_i$) eliminating negative transfer. `[COMPLETED ✅]`
  - Member 4: Publication visualizer generating confusion matrices, t-SNE embeddings, and automated LaTeX tables. `[COMPLETED ✅]`

---

## 7. Selected References & Seminar Literature Stack

1. **Ma, Y., et al.** (IEEE COMST 2019). *WiFi Sensing with Channel State Information: A Survey.*
2. **Chen, Z., et al.** (IEEE TMC 2023). *Cross-Domain Wi-Fi Sensing via Spatial-Temporal Representation Learning.*
3. **Liang, J., et al.** (ICML 2020). *Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation (SHOT).*
4. **Cao, Z., et al.** (CVPR 2018 / IEEE TPAMI 2019). *Partial Transfer Learning with Selective Adversarial Networks (SAN).*
5. **van den Oord, A., et al.** (2018). *Representation Learning with Contrastive Predictive Coding (CPC).*
6. **Li, Z., et al.** (IEEE TPAMI 2024). *A Comprehensive Survey on Source-Free Universal Domain Adaptation.*
7. **Yan, H., et al.** (IEEE INFOCOM 2025). *Wi-SFDAGR: Wi-Fi Source-Free Domain Adaptation with Graph Regularization.*
8. **Radwan, A., & Tabassum, H.** (IEEE TWC 2026). *MU-SHOT-Fi: Multi-User Source-Free Hypothesis Transfer for Wi-Fi Sensing.*
