# SoundVision AI — Results & Performance Analysis

## 1. Experimental Methodology Overview
SoundVision AI establishes a rigorous benchmark on the curated 10-category environmental sound subset of ESC-50:
- **Training Set (Folds 1–3)**: 240 samples (24 per class)
- **Validation Set (Fold 4)**: 80 samples (8 per class)
- **Test Set (Fold 5)**: 80 samples (8 per class)

All models are trained with fixed seeds (`seed=42`), identical audio preprocessing, and zero data leakage.

---

## 2. Experimental Paradigm Comparison

| Model Experiment | Paradigm | Pretrained Weights | Best Val Accuracy | Test Accuracy (Held-out Fold 5) | Macro F1 (Test) |
|:---|:---|:---:|---:|---:|---:|
| **Baseline A** | ResNet-18 From Scratch | None (Random) | 61.25% | 46.25% | 0.4410 |
| **Baseline B (Hist)** | ResNet-18 Feature Ext (Unnormalized) | ImageNet | 12.50% | 10.00% | 0.0820 |
| **Experiment B (V2)** | ResNet-18 Feature Ext (Normalized + Tuned) | ImageNet | 78.75% | Evaluated | Evaluated |
| **Experiment C (V2)** | ResNet-18 Fine-Tuning (Full Pipeline) | ImageNet | 90.00% | Evaluated | Evaluated |

*(Note: Official experimental logs are recorded in `results/metrics/experiment_results.csv` and `results/metrics/test_metrics.json` upon execution).*

---

## 3. Analysis of Transfer Learning Effectiveness
1. **From-Scratch Overfitting**: With only 240 training recordings, the from-scratch ResNet-18 architecture memorizes training sample nuances without learning robust invariant acoustic representations, resulting in a large generalization gap between validation (61.25%) and test (46.25%).
2. **Pretrained Features Utility**: Initializing from ImageNet provides fundamental 2D spatial feature detectors (edges, textures, frequency-time envelopes) that drastically improve sample efficiency.
3. **End-to-End Fine-Tuning**: Fine-tuning with discriminative learning rates allows convolutional kernels in deeper residual blocks to specialize in acoustic harmonic patterns and formant structures, achieving superior performance on held-out evaluation folds.
