# SoundVision AI — Known Limitations & Failure Modes

## 1. Closed-Set Classification Boundary
- **Limitation**: SoundVision AI operates under a closed-world assumption. It assumes every input belongs to one of the 10 known environmental sound categories.
- **Consequence**: When presented with out-of-distribution sounds (e.g., human speech, music, baby crying, car horns, water drops), the model forces the input into the nearest known acoustic category, potentially with high softmax confidence.
- **Remediation**: In production systems, an energy threshold detector or open-set out-of-distribution (OOD) rejection head (e.g. Mahalanobis distance in feature space or temperature-scaled entropy filtering) should precede classification.

---

## 2. Acoustic Confusion Between Similar Spectral Profiles
Based on empirical error analysis, specific sound pairs share significant acoustic similarities:

1. **`airplane` vs `helicopter`**:
   - Both categories feature dominant low-to-mid frequency broadband engine noise. While helicopters produce distinct periodic rotor blade slap modulations, distance attenuation, Doppler shifting, and microphone compression can smooth these periodic peaks into a continuous drone similar to jet engines.

2. **`wind` vs `thunderstorm`**:
   - High-velocity turbulent wind displays a low-pass brown noise spectrum. In recordings of thunderstorms without sharp, close lightning strikes, distant thunder rumbles appear spectrally almost indistinguishable from deep aerodynamic wind turbulence.

3. **`fireworks` vs `thunderstorm`**:
   - Both sound types contain explosive impulse attacks followed by low-frequency reverberant decays.

---

## 3. Recording Duration & Temporal Dynamics
- **Limitation**: The model was trained on 5.0-second stationary clips from the ESC-50 dataset.
- **Short Audio (< 1.0 second)**: Very brief sound bursts padded with long silence can exhibit diluted acoustic feature maps.
- **Long Audio (> 10.0 seconds)**: While `AudioPredictor` implements sliding-window segmentation with 50% overlap, time-averaged predictions can obscure transient events if the foreground sound occurs in only a single 1-second segment amidst long silence.

---

## 4. Hardware and Computational Constraints
- **Limitation**: The current benchmark was developed with CPU/GPU dual compatibility. On resource-constrained edge microcontrollers, real-time spectrogram calculation via STFT and ResNet-18 forward passes require substantial RAM and floating-point throughput.
- **Future Improvement**: Quantization (INT8 post-training quantization) or migration to lightweight architectures such as MobileNetV4, EfficientNet-B0, or EdgeNeXt for embedded deployments.
