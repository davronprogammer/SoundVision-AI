# SoundVision AI — Model Card

## Model Details
- **Developer**: SoundVision AI Engineering Team
- **Model Architecture**: ResNet-18 (Deep Residual Learning for Image Recognition, adapted for 2D audio Mel-spectrograms)
- **Model Version**: 1.0.0
- **Supported Classes (10 categories)**:
  `dog`, `thunderstorm`, `fireworks`, `airplane`, `train`, `sheep`, `wind`, `frog`, `helicopter`, `laughing`
- **Release Date**: 2026-11-05
- **License**: MIT License

---

## Intended Use
- **Primary Use Case**: Automatic acoustic event tagging and classification of environmental audio clips.
- **Input Modality**: Single or multi-channel WAV audio files.
- **Output**: Multi-class posterior probability distribution over the 10 supported environmental sound classes, accompanied by top-$k$ predictive rankings.
- **Intended Users**: Audio engineers, environmental researchers, acoustic IoT developers, and ML portfolio evaluators.

---

## Acoustic Preprocessing & Input Specifications
- **Target Sampling Rate**: 22,050 Hz (automatic resampling applied to arbitrary input rates)
- **Channel Format**: Mono (multi-channel inputs are averaged)
- **Window Length (N_FFT)**: 1024 samples ($\approx 46.4\text{ ms}$)
- **Hop Size (HOP_LENGTH)**: 512 samples ($\approx 23.2\text{ ms}$)
- **Frequency Scale**: Mel-scale filterbank with 128 triangular filters spanning $0\text{ Hz}$ to $11,025\text{ Hz}$
- **Dynamic Range**: Logarithmic decibel scale ($80\text{ dB}$ top limit)
- **Input Tensor Dimensions**: `(Batch, 3, 128, 216)` (3 channels repeated, 128 mel frequency bins, 216 time frames for 5.0-second audio)

---

## Performance Summary
- **Evaluation Split**: Held-out Test Fold 5 of ESC-50 (80 recordings, strictly unseen during training and validation)
- **Validation Split**: Validation Fold 4 of ESC-50 (80 recordings)
- **Primary Metrics**: Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall, and Normalized Confusion Matrix.

---

## Ethical Considerations & Responsible AI
- **Closed-World Constraint**: The model produces softmax probability distributions strictly conditioned on the 10 target categories. It does not perform open-set anomaly detection.
- **Human Voice Considerations**: The `laughing` category models non-verbal paralinguistic laughter sounds. The model does not perform speech transcription, speaker identification, or sentiment inference.
