# SoundVision AI — Controlled Experiments & Optimization Log

## 1. Experimental Protocol
- **Dataset**: ESC-50 10-Class Subset (400 samples total, 40 per class)
- **Official Folds**:
  - Training: Folds 1, 2, 3 (240 samples)
  - Validation: Fold 4 (80 samples)
  - Test: Fold 5 (80 samples)
- **Model Selection Rule**: All model selections, hyperparameter decisions, and early stopping triggers are determined **strictly on Validation Fold 4**. Fold 5 is reserved exclusively for the final held-out test evaluation.

---

## 2. Historical Colab Experiments (Baseline Context)

The following baseline metrics were recorded in early Google Colab prototype iterations:

| Model ID | Paradigm | Best Val Accuracy | Test Accuracy (Fold 5) | Notes |
|:---|:---|---:|---:|:---|
| `hist_scratch` | ResNet18 From Scratch | 61.25% | 46.25% | Severe overfitting; random weights struggled on 240 samples. |
| `hist_feat_ext` | ResNet18 Feature Extraction | 12.50% | 10.00% | Suffered from exploding initial loss due to unnormalized dB spectrograms. |
| `hist_finetune` | ResNet18 Fine-Tuning | 86.25% | 70.00% | Significant leap in validation accuracy, confirming transfer learning utility. |

### Intermediate Improved Iterations (V2, Unverified on Test)
- `hist_feat_ext_v2`: Val Acc **78.75%** (Normalized inputs, tuned head LR)
- `hist_finetune_v2`: Val Acc **90.00%** (SpecAugment, AdamW, discriminative LRs, weight decay)
*(Note: As mandated by engineering protocol, the 90.00% metric was recorded exclusively on validation Fold 4 and was never claimed as test performance).*

---

## 3. Systematic Controlled Experiments Tracking
All newly executed local experiments are appended automatically to:
`results/metrics/experiment_results.csv`

The experiment pipeline logs:
1. `experiment_id`
2. `paradigm` (`scratch`, `feature_extraction`, `fine_tuning`)
3. `architecture` (`resnet18`)
4. `pretrained` (`True`/`False`)
5. `learning_rate_head`
6. `learning_rate_backbone`
7. `batch_size`
8. `epochs_trained`
9. `best_val_accuracy`
10. `best_val_loss`
11. `best_val_f1`
12. `duration_seconds`
13. `checkpoint_path`

---

## 4. Hyperparameter Search Space & Sensibility

| Hyperparameter | Evaluated Values | Selected Default | Technical Justification |
|---|---|---|---|
| **Batch Size** | 8, 16, 32 | `16` | 16 provides smooth stochastic gradients over 240 training samples (15 batches per epoch) without high memory overhead. |
| **Backbone LR** | $10^{-5}$, $10^{-4}$, $3 \times 10^{-4}$ | $10^{-4}$ | Prevents catastrophic forgetting of ImageNet representations while allowing kernel adaptation. |
| **Head LR** | $5 \times 10^{-4}$, $10^{-3}$, $3 \times 10^{-3}$ | $10^{-3}$ | Standard learning rate for training a newly initialized linear layer. |
| **Weight Decay** | $10^{-5}$, $10^{-4}$, $10^{-3}$ | $10^{-4}$ | Regularizes $L_2$ norm without excessive parameter shrinkage. |
| **Label Smoothing**| $0.0$, $0.05$, $0.1$ | $0.05$ | Prevents overconfident softmax predictions, encouraging better calibration. |
| **SpecAugment** | Low, Moderate, High | Moderate ($F=16, T=24$) | Effectively prevents acoustic overfitting without distorting dominant sound signatures. |
