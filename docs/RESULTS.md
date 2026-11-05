# SoundVision AI — Results & Performance Analysis

## 1. Experimental Methodology Overview
SoundVision AI establishes a rigorous benchmark on the curated 10-category environmental sound subset of ESC-50:
- **Training Set (Folds 1–3)**: 240 samples (24 per class)
- **Validation Set (Fold 4)**: 80 samples (8 per class)
- **Test Set (Fold 5)**: 80 samples (8 per class)

All models are trained with fixed seeds (`seed=42`), identical audio preprocessing, and zero data leakage.

---

## 2. Experimental Paradigm Comparison

The table below contrasts historical Google Colab experiments with our locally reproduced and evaluated runs on official ESC-50 splits:

### A. Locally Executed & Evaluated Benchmarks (Zero Data Leakage)

| Model Experiment | Paradigm | Pretrained Weights | Best Val Accuracy | Best Val Macro F1 | Test Accuracy (Fold 5) | Test Macro F1 | Status |
|:---|:---|:---:|---:|---:|---:|---:|:---:|
| **Experiment A** | ResNet-18 From Scratch | None (Random) | 70.00% | 67.05% | 55.00% | 51.14% | Complete (25 ep) |
| **Experiment B** | ResNet-18 Feature Extraction | ImageNet-1K | 10.00% | 6.61% | N/A | N/A | Early Stopped (ep 17) |
| **Experiment C** | ResNet-18 Fine-Tuning (Production) | ImageNet-1K | **75.00%** | **73.68%** | **78.75%** | **75.82%** | Complete (Best ep 13) |

### B. Historical Google Colab Reference Experiments

| Model Experiment | Best Validation Accuracy | Test Accuracy | Note |
|:---|---:|---:|:---|
| ResNet-18 from scratch | 61.25% | 46.25% | Baseline random initialization |
| Feature extraction (Baseline) | 12.50% | 10.00% | Suffered initial exploding loss due to unnormalized dB inputs |
| Fine-tuning (Baseline) | 86.25% | 70.00% | Pretrained ImageNet with uncalibrated head |
| Feature extraction V2 | 78.75% | Not evaluated on test | Added training-set normalization |
| Fine-tuning V2 | 90.00% | Not evaluated on test | Colab run with aggressive tuning |

*(All verified local runs are preserved in `results/metrics/experiment_results.csv` and `results/metrics/test_metrics.json`).*

---

## 3. Analysis of Transfer Learning Effectiveness
1. **From-Scratch Overfitting**: With only 240 training recordings, the from-scratch ResNet-18 architecture memorizes training sample nuances without learning robust invariant acoustic representations, resulting in a large generalization gap between validation (61.25%) and test (46.25%).
2. **Pretrained Features Utility**: Initializing from ImageNet provides fundamental 2D spatial feature detectors (edges, textures, frequency-time envelopes) that drastically improve sample efficiency.
3. **End-to-End Fine-Tuning**: Fine-tuning with discriminative learning rates allows convolutional kernels in deeper residual blocks to specialize in acoustic harmonic patterns and formant structures, achieving superior performance on held-out evaluation folds.
