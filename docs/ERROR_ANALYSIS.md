# SoundVision AI — Error Analysis Report

**Model Evaluated**: SoundVision ResNet18 (test)  
**Test Set Samples**: 80  
**Misclassified Samples**: 17  
**Empirical Accuracy**: 78.75%  

## 1. Top Confused Acoustic Pairs

| True Category | Predicted Category | Error Count | Acoustic Hypothesis |
|---|---|---:|---|
| `fireworks` | `frog` | 4 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `fireworks` | `laughing` | 3 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `airplane` | `wind` | 2 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `dog` | `laughing` | 2 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `helicopter` | `airplane` | 2 | Periodic blade slaps smoothed by distance attenuation into continuous low-frequency roar. |
| `airplane` | `sheep` | 1 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `airplane` | `train` | 1 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `helicopter` | `sheep` | 1 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |
| `thunderstorm` | `train` | 1 | Shared acoustic envelope, similar frequency distribution or environmental background noise. |

## 2. Per-Class Error Breakdown

| Category | Total Errors |
|---|---:|
| `fireworks` | 7 |
| `airplane` | 4 |
| `helicopter` | 3 |
| `dog` | 2 |
| `thunderstorm` | 1 |

## 3. Notable Misclassified Examples

| Filename | True Label | Predicted Label | Confidence | Top-3 Alternatives |
|---|---|---|---:|---|
| `5-156999-A-19.wav` | `thunderstorm` | `train` | 0.653 | train (0.65), thunderstorm (0.34), wind (0.00) |
| `5-160614-A-48.wav` | `fireworks` | `frog` | 0.998 | frog (1.00), laughing (0.00), wind (0.00) |
| `5-160614-B-48.wav` | `fireworks` | `frog` | 1.000 | frog (1.00), train (0.00), helicopter (0.00) |
| `5-160614-C-48.wav` | `fireworks` | `frog` | 1.000 | frog (1.00), laughing (0.00), fireworks (0.00) |
| `5-160614-D-48.wav` | `fireworks` | `laughing` | 1.000 | laughing (1.00), frog (0.00), fireworks (0.00) |
| `5-160614-E-48.wav` | `fireworks` | `laughing` | 1.000 | laughing (1.00), frog (0.00), fireworks (0.00) |
| `5-160614-F-48.wav` | `fireworks` | `laughing` | 1.000 | laughing (1.00), frog (0.00), fireworks (0.00) |
| `5-160614-H-48.wav` | `fireworks` | `frog` | 0.518 | frog (0.52), airplane (0.48), wind (0.01) |
| `5-191131-A-40.wav` | `helicopter` | `airplane` | 0.873 | airplane (0.87), wind (0.13), train (0.00) |
| `5-203128-B-0.wav` | `dog` | `laughing` | 1.000 | laughing (1.00), dog (0.00), frog (0.00) |
| `5-205898-A-40.wav` | `helicopter` | `sheep` | 0.869 | sheep (0.87), airplane (0.09), wind (0.03) |
| `5-212454-A-0.wav` | `dog` | `laughing` | 0.603 | laughing (0.60), dog (0.40), helicopter (0.00) |
| `5-215445-A-47.wav` | `airplane` | `wind` | 1.000 | wind (1.00), frog (0.00), airplane (0.00) |
| `5-215449-A-47.wav` | `airplane` | `wind` | 0.400 | wind (0.40), train (0.34), airplane (0.25) |
| `5-220955-A-40.wav` | `helicopter` | `airplane` | 1.000 | airplane (1.00), frog (0.00), fireworks (0.00) |