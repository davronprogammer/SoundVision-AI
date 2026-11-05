# ESC-50 Dataset — SoundVision AI Subset

This project utilizes a curated 10-category environmental sound subset from the official **ESC-50 (Dataset for Environmental Sound Classification)** repository:
`https://github.com/karolpiczak/ESC-50`

## Selected Categories
1. `dog`
2. `thunderstorm`
3. `fireworks`
4. `airplane`
5. `train`
6. `sheep`
7. `wind`
8. `frog`
9. `helicopter`
10. `laughing`

## Specifications
- **Total recordings**: 400 audio files (40 recordings per class)
- **Duration**: 5.0 seconds per recording (standard ESC-50 clip length)
- **Audio format**: WAV, mono 22,050 Hz
- **Official Folds Distribution**:
  - **Folds 1, 2, 3**: Training set (240 clips)
  - **Fold 4**: Validation set (80 clips)
  - **Fold 5**: Held-out Test set (80 clips)

## Integrity & No Leakage Guarantee
ESC-50 is partitioned into 5 cross-validation folds ensuring that clips originating from the same source recording belong to the exact same fold.
All feature normalization statistics (mean and standard deviation) are computed strictly across training folds (1–3), preventing test-set data leakage.
