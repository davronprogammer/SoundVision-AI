# SoundVision AI — Error Analysis Report

**Model Evaluated**: SoundVision ResNet18 (test)  
**Test Set Samples**: 80  
**Misclassified Samples**: 36  
**Empirical Accuracy**: 55.00%  

## 1. Top Confused Acoustic Pairs

| True Category | Predicted Category | Error Count | Acoustic Hypothesis |
|---|---|---:|---|
| `helicopter` | `wind` | 5 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `airplane` | `thunderstorm` | 4 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `wind` | `thunderstorm` | 3 | Continuous low-pass filtered brown noise lacking sharp lightning transient attacks. |
| `frog` | `thunderstorm` | 3 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `airplane` | `train` | 3 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `sheep` | `thunderstorm` | 2 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `frog` | `dog` | 2 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `wind` | `airplane` | 2 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `sheep` | `fireworks` | 2 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `wind` | `laughing` | 1 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |

## 2. Per-Class Error Breakdown

| Category | Total Errors |
|---|---:|
| `airplane` | 8 |
| `helicopter` | 7 |
| `wind` | 7 |
| `frog` | 5 |
| `sheep` | 4 |
| `laughing` | 2 |
| `thunderstorm` | 2 |
| `fireworks` | 1 |

## 3. Notable Misclassified Examples

| Filename | True Label | Predicted Label | Confidence | Top-3 Alternatives |
|---|---|---|---:|---|
| `5-156026-A-4.wav` | `frog` | `dog` | 0.553 | dog (0.55), thunderstorm (0.29), fireworks (0.16) |
| `5-156026-B-4.wav` | `frog` | `thunderstorm` | 0.977 | thunderstorm (0.98), fireworks (0.02), dog (0.00) |
| `5-156026-C-4.wav` | `frog` | `thunderstorm` | 0.999 | thunderstorm (1.00), fireworks (0.00), airplane (0.00) |
| `5-156026-D-4.wav` | `frog` | `dog` | 1.000 | dog (1.00), fireworks (0.00), laughing (0.00) |
| `5-156999-C-19.wav` | `thunderstorm` | `airplane` | 0.476 | airplane (0.48), thunderstorm (0.41), wind (0.11) |
| `5-156999-E-19.wav` | `thunderstorm` | `train` | 0.545 | train (0.54), helicopter (0.35), airplane (0.09) |
| `5-157204-A-16.wav` | `wind` | `airplane` | 0.990 | airplane (0.99), sheep (0.01), train (0.00) |
| `5-157204-B-16.wav` | `wind` | `airplane` | 0.966 | airplane (0.97), sheep (0.02), wind (0.01) |
| `5-160614-B-48.wav` | `fireworks` | `frog` | 0.874 | frog (0.87), laughing (0.12), sheep (0.00) |
| `5-171118-A-26.wav` | `laughing` | `fireworks` | 0.972 | fireworks (0.97), dog (0.02), sheep (0.01) |
| `5-177957-A-40.wav` | `helicopter` | `wind` | 0.962 | wind (0.96), train (0.04), airplane (0.00) |
| `5-177957-B-40.wav` | `helicopter` | `wind` | 0.995 | wind (1.00), train (0.00), helicopter (0.00) |
| `5-177957-C-40.wav` | `helicopter` | `wind` | 0.985 | wind (0.99), helicopter (0.01), train (0.00) |
| `5-177957-D-40.wav` | `helicopter` | `wind` | 0.997 | wind (1.00), airplane (0.00), helicopter (0.00) |
| `5-177957-E-40.wav` | `helicopter` | `wind` | 0.998 | wind (1.00), train (0.00), helicopter (0.00) |