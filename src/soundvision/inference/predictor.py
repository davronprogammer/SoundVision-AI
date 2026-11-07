"""Production inference API for SoundVision AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from soundvision.data.dataset import SELECTED_CLASSES
from soundvision.data.preprocessing import AudioPreprocessor
from soundvision.models.resnet import SoundVisionResNet


class AudioPredictor:
    """Robust inference engine for environmental sound classification."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads model weights, preprocessing statistics, and class definitions."""
        try:
            # First attempt secure weights_only=True loading (recommended for PyTorch >= 2.6)
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback to weights_only=False for backward compatibility with full training checkpoints
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint at '{self.checkpoint_path}': {e}") from e

        self.class_names: List[str] = checkpoint.get("class_names", SELECTED_CLASSES)
        self.num_classes = len(self.class_names)
        self.sample_rate: int = checkpoint.get("sample_rate", 22050)
        self.n_fft: int = checkpoint.get("n_fft", 1024)
        self.hop_length: int = checkpoint.get("hop_length", 512)
        self.n_mels: int = checkpoint.get("n_mels", 128)
        self.norm_mean: Optional[float] = checkpoint.get("norm_mean", None)
        self.norm_std: Optional[float] = checkpoint.get("norm_std", None)

        # Build model
        self.model = SoundVisionResNet(
            num_classes=self.num_classes,
            pretrained=False,
            freeze_backbone=False,
            dropout=0.0,
            in_channels=3,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Build preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            target_duration=5.0,
            in_channels=3,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )

    def predict(
        self,
        audio_input: Union[str, Path, np.ndarray, torch.Tensor],
        top_k: int = 3,
        segment_overlap: float = 0.5,
    ) -> Dict[str, Any]:
        """Classifies an audio file or raw waveform tensor.

        Supports arbitrary duration audio via sliding window segmentation.

        Args:
            audio_input: Path to audio file or raw 1D/2D audio array/tensor.
            top_k: Number of highest-ranked classes to return.
            segment_overlap: Overlap fraction [0.0, 0.9] between sliding windows for audio > 5.0s.

        Returns:
            Dictionary containing predicted_class, confidence, and top_predictions list.
        """
        if top_k < 1 or top_k > self.num_classes:
            raise ValueError(f"top_k must be between 1 and {self.num_classes}, got {top_k}")

        # 1. Load or standardize input waveform
        if isinstance(audio_input, (str, Path)):
            waveform, _ = self.preprocessor.load_audio(audio_input)
        elif isinstance(audio_input, np.ndarray):
            if audio_input.ndim == 1:
                waveform = torch.from_numpy(audio_input).float().unsqueeze(0)
            elif audio_input.ndim == 2:
                # Average to mono
                mono = np.mean(audio_input, axis=0 if audio_input.shape[0] < audio_input.shape[1] else 1)
                waveform = torch.from_numpy(mono).float().unsqueeze(0)
            else:
                raise ValueError(f"Expected 1D or 2D audio array, got ndim={audio_input.ndim}")
        elif isinstance(audio_input, torch.Tensor):
            if audio_input.ndim == 1:
                waveform = audio_input.unsqueeze(0).float()
            elif audio_input.ndim == 2:
                if audio_input.shape[0] > 1:
                    waveform = audio_input.mean(dim=0, keepdim=True).float()
                else:
                    waveform = audio_input.float()
            else:
                raise ValueError(f"Expected 1D or 2D tensor, got shape={audio_input.shape}")
        else:
            raise TypeError(f"Unsupported audio_input type: {type(audio_input)}")

        total_samples = waveform.shape[-1]
        target_samples = self.preprocessor.target_samples

        # 2. Windowing / Segmentation if duration > 5.0 seconds
        if total_samples <= target_samples:
            segments = [self.preprocessor.adjust_duration(waveform)]
        else:
            step = max(int(target_samples * (1.0 - segment_overlap)), 1)
            segments = []
            for start in range(0, total_samples - target_samples + 1, step):
                segments.append(waveform[..., start : start + target_samples])
            # If the remainder at the end was missed, include the final tail window
            if (total_samples - target_samples) % step != 0:
                segments.append(waveform[..., total_samples - target_samples : total_samples])

        # 3. Process each segment through model and aggregate probabilities
        segment_tensors = []
        for seg in segments:
            mel_db = self.preprocessor.extract_mel_spectrogram(seg)
            mel_norm = self.preprocessor.normalize(mel_db)
            feat = self.preprocessor.to_multi_channel(mel_norm)
            segment_tensors.append(feat)

        batch_tensors = torch.stack(segment_tensors, dim=0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            mean_probs = np.mean(probs, axis=0)

        # 4. Format top-k results
        top_indices = np.argsort(mean_probs)[::-1][:top_k]
        top_predictions = [
            {
                "class": self.class_names[idx],
                "probability": float(round(float(mean_probs[idx]), 4)),
            }
            for idx in top_indices
        ]

        best_idx = top_indices[0]
        return {
            "predicted_class": self.class_names[best_idx],
            "confidence": float(round(float(mean_probs[best_idx]), 4)),
            "top_predictions": top_predictions,
            "disclaimer": (
                "Confidence is derived from softmax activations over 10 closed-set environmental classes "
                "and should not be interpreted as calibrated open-world sound detection probability."
            ),
        }
