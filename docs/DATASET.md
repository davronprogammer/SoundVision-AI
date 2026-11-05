# SoundVision AI — Dataset Specification

## 1. Source & Attribution
- **Dataset**: ESC-50 (Dataset for Environmental Sound Classification)
- **Author**: Karol J. Piczak
- **Original Publication**: *"ESC: Dataset for Environmental Sound Classification"*, ACM Multimedia 2015.
- **Repository**: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
- **License**: Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0)

---

## 2. Selected Categories
We curated a balanced 10-category environmental sound subset from the original 50 ESC-50 classes:

| Index | Category | Type / Sub-Domain | Typical Acoustic Properties |
|:---:|:---|:---|:---|
| 0 | `dog` | Animal / Canine | Impulsive, harmonic vocalizations, high peak energy bursts |
| 1 | `thunderstorm` | Natural soundscape | Low-frequency rumble, long reverberant decays, sudden explosive transients |
| 2 | `fireworks` | Urban / Explosive | Sharp impulsive attacks, broadband explosive crackles, high dynamic range |
| 3 | `airplane` | Transportation / Aircraft | Broadband jet thrust noise, steady stationary drone, mid-frequency emphasis |
| 4 | `train` | Transportation / Rail | Periodic mechanical clatter, screeching wheel-rail friction, horn blasts |
| 5 | `sheep` | Animal / Livestock | Sustained harmonic bleats, distinctive formant frequencies |
| 6 | `wind` | Natural soundscape | Continuous turbulence, brown-noise-like spectrum, gentle amplitude modulations |
| 7 | `frog` | Animal / Amphibian | Rhythmic croaking, periodic low-to-mid frequency pulses |
| 8 | `helicopter` | Transportation / Aircraft | Characteristic low-frequency blade-slap modulation, strong rhythmic harmonics |
| 9 | `laughing` | Human non-speech | Rapid pseudo-periodic acoustic pulses, human formant vocal tract resonance |

---

## 3. Dataset Statistics & Structure
- **Total Recordings**: 400 audio files
- **Number of Classes**: 10
- **Recordings per Class**: 40
- **Duration per Clip**: 5.0 seconds
- **Original Format**: WAV, mono & stereo 44,100 Hz 16-bit
- **Standardized Format**: WAV, mono 22,050 Hz float32

---

## 4. Official Fold-Based Partitioning & Leakage Prevention
ESC-50 is officially partitioned into 5 cross-validation folds. A critical property of the ESC-50 benchmark is that audio fragments extracted from the exact same source recording belong to the exact same fold. 

To strictly prevent data leakage:
- **Folds 1, 2, 3 (Training)**: 240 samples (24 recordings per class)
- **Fold 4 (Validation)**: 80 samples (8 recordings per class)
- **Fold 5 (Held-out Test)**: 80 samples (8 recordings per class)

**Important Rule**: Normalization statistics (mean $\mu_{train}$ and standard deviation $\sigma_{train}$) must be calculated **strictly from Folds 1–3**. No validation or test samples may ever influence normalization statistics or training parameters.
