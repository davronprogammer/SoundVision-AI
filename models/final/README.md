# SoundVision AI — Model Checkpoints

This directory houses the trained PyTorch model checkpoints for SoundVision AI.

## Checkpoint Specifications
- **Architecture**: `SoundVisionResNet` (ResNet-18 backbone with custom 10-class environmental classifier head)
- **Input Channels**: 3 (repeated log-Mel spectrogram channels)
- **Mel Filterbanks**: 128
- **Audio Sample Rate**: 22,050 Hz
- **Classes**: `dog`, `thunderstorm`, `fireworks`, `airplane`, `train`, `sheep`, `wind`, `frog`, `helicopter`, `laughing`

## GitHub Storage & LFS Note
In accordance with GitHub's 100 MB single-file limitation, raw `.pth` binary checkpoints (which include model weights, optimizer state, scheduler state, and normalization parameters $\approx 134 \text{ MB}$) are excluded from git history via `.gitignore`.

To produce the model checkpoint locally, simply run:
```powershell
python scripts/train.py --experiment-type fine_tuning --epochs 25
```
This automatically saves the trained model to `models/final/soundvision_resnet18.pth`.
