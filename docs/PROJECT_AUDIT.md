# SoundVision AI — Project Audit & Implementation Plan

## 1. Executive Summary & Existing Workspace Audit

**Audit Date**: 2026-11-05  
**Auditor**: Senior Machine Learning & MLOps Engineer  
**Workspace**: `C:\Users\davro\Desktop\SoundVision-AI`

### Initial State Analysis:
- **Files & Assets Present**: The repository workspace was initially initialized empty. No legacy scratch scripts, intermediate notebooks, or local cached datasets were present.
- **Environment**: Windows 11 64-bit with Python 3.12 (`py -3.12`) runtime. Virtual environment initialized at `.venv`.
- **Pretrained Weights & Compute**: Hardware execution is configured with robust CPU device fallback (and automatic CUDA GPU acceleration when accessible).
- **Historical Experiments Context**:
  - ResNet18 from scratch: Val Acc 61.25%, Test Acc 46.25%
  - ResNet18 Feature extraction baseline: Val Acc 12.50%, Test Acc 10.00% (suffered from exploding loss caused by unnormalized Mel-spectrogram inputs, inappropriate head initialization, and frozen BatchNorm running statistics mismatch)
  - ResNet18 Fine-tuning baseline: Val Acc 86.25%, Test Acc 70.00%
  - Improved V2 iterations (with SpecAugment, training-set normalization, AdamW, discriminative learning rates): Feature extraction V2 achieved Val Acc 78.75%; Fine-tuning V2 achieved Val Acc 90.00% (untested on held-out fold 5).

---

## 2. Identified Technical Challenges & Root Causes

1. **Feature Extraction Initial Exploding Loss**:
   - *Root cause*: Mel-spectrograms converted to dB scale typically have values in ranges like `[-80, 0]` or `[-100, 20]`. Passing unnormalized inputs directly into an ImageNet-pretrained ResNet backbone (which expects z-score normalized inputs `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) causes extreme activations at early convolutional layers. Combined with a freshly initialized dense linear layer with default weights and high learning rate, gradients explode or saturate softmax.
   - *Solution*: Calculate exact channel-wise mean and standard deviation strictly over the training folds (folds 1–3). Apply standard z-score normalization prior to the backbone. Furthermore, calibrate learning rate (e.g. 1e-3 for head, 1e-4 for backbone in fine-tuning), apply gradient clipping (`max_norm=1.0`), and ensure proper BatchNorm operational modes (`eval` mode for frozen backbones).

2. **Data Leakage Prevention**:
   - In environmental sound datasets, fold splits are clustered by original sound recordings (ESC-50 author Karol J. Piczak explicitly structured the 5 folds so that fragments from the same original source recording do not cross folds).
   - *Rule*: Normalization mean and standard deviation must be computed strictly on training folds (folds 1–3) and never on validation (fold 4) or test (fold 5) sets.
   - Data augmentations (SpecAugment frequency & time masking, random gain, time shifting) must only be applied during training.

3. **Reproducibility**:
   - Fix seeds for Python `random`, `numpy`, and `torch` (`manual_seed`, `cuda.manual_seed_all`).
   - Use deterministic algorithms when possible.
   - Save full checkpoint state dictionaries including model weights, optimizer state, epoch, preprocessing statistics, and class mapping.

---

## 3. Systematic Implementation Plan

| Phase | Milestone | Objectives |
|:---:|:---|:---|
| **Phase 1** | Project Audit | Complete workspace audit and architectural specification (this document). |
| **Phase 2** | Project Architecture | Scaffold full production directory tree, configuration files, and package metadata. |
| **Phase 3** | Data Engineering | Implement dataset downloader, verification, audio loader (22050 Hz mono), Mel-spectrogram extractor (1024 FFT, 512 hop, 128 mels), training-set normalizer, and SpecAugment. |
| **Phase 4** | Model Architectures | Implement ResNet18 model factory supporting from-scratch, feature extraction (frozen backbone, trainable head), and fine-tuning with discriminative learning rates. |
| **Phase 5** | Training Engine | Implement modular PyTorch trainer with early stopping, learning rate scheduler, gradient clipping, metrics tracking (Accuracy, Macro F1, Loss), and checkpointing. |
| **Phase 6** | Controlled Experiments | Execute baseline models and improved models. Log all metrics to `results/metrics/experiment_results.csv`. |
| **Phase 7** | Evaluation & Error Analysis | Generate confusion matrices, classification reports, per-class F1 charts, training curves, and deep-dive error analysis on confused classes. |
| **Phase 8** | Inference API | Build robust `AudioPredictor` supporting variable-length WAV audio, padding/segmentation, top-k predictions, and calibrated outputs. |
| **Phase 9** | Streamlit Application | Build multi-tab interactive web app with audio playback, waveform/spectrogram visualization, real-time prediction, and evaluation dashboard. |
| **Phase 10** | Automated Testing | Comprehensive unit and integration test suite covering dataset, preprocessing, model outputs, inference, and splits. |
| **Phase 11** | Professional Documentation | Production GitHub README, METHODOLOGY.md, RESULTS.md, ERROR_ANALYSIS.md, MODEL_CARD.md, and LIMITATIONS.md. |
