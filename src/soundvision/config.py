"""Configuration loader and schema definition for SoundVision AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class PreprocessingConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 11025.0
    top_db: float = 80.0
    in_channels: int = 3
    norm_mean: Optional[float] = None
    norm_std: Optional[float] = None


@dataclass
class AugmentationConfig:
    enabled: bool = True
    time_mask_max_len: int = 24
    freq_mask_max_len: int = 16
    num_time_masks: int = 2
    num_freq_masks: int = 2
    gain_range: List[float] = field(default_factory=lambda: [-3.0, 3.0])
    time_shift_max_ratio: float = 0.1


@dataclass
class DatasetConfig:
    name: str = "ESC-50"
    repo_url: str = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    meta_csv: str = "data/raw/ESC-50-master/meta/esc50.csv"
    audio_dir: str = "data/raw/ESC-50-master/audio"
    classes: List[str] = field(
        default_factory=lambda: [
            "dog",
            "thunderstorm",
            "fireworks",
            "airplane",
            "train",
            "sheep",
            "wind",
            "frog",
            "helicopter",
            "laughing",
        ]
    )
    sample_rate: int = 22050
    target_duration_sec: float = 5.0
    train_folds: List[int] = field(default_factory=lambda: [1, 2, 3])
    val_folds: List[int] = field(default_factory=lambda: [4])
    test_folds: List[int] = field(default_factory=lambda: [5])


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 10
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 30
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.05
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_name: str = "CosineAnnealingLR"


@dataclass
class PathsConfig:
    checkpoints_dir: str = "models/checkpoints"
    final_model_dir: str = "models/final"
    figures_dir: str = "results/figures"
    metrics_dir: str = "results/metrics"
    predictions_dir: str = "results/predictions"


@dataclass
class AppConfig:
    project_name: str = "SoundVision-AI"
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults if not found."""
    if config_path is None:
        config_path = "configs/config.yaml"

    path = Path(config_path)
    if not path.is_file():
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg = AppConfig()
    cfg.project_name = data.get("project", {}).get("name", cfg.project_name)
    cfg.seed = data.get("project", {}).get("seed", cfg.seed)

    # Dataset
    d_data = data.get("dataset", {})
    cfg.dataset.name = d_data.get("name", cfg.dataset.name)
    cfg.dataset.repo_url = d_data.get("repo_url", cfg.dataset.repo_url)
    cfg.dataset.raw_dir = d_data.get("raw_dir", cfg.dataset.raw_dir)
    cfg.dataset.processed_dir = d_data.get("processed_dir", cfg.dataset.processed_dir)
    cfg.dataset.meta_csv = d_data.get("meta_csv", cfg.dataset.meta_csv)
    cfg.dataset.audio_dir = d_data.get("audio_dir", cfg.dataset.audio_dir)
    cfg.dataset.classes = d_data.get("classes", cfg.dataset.classes)
    cfg.dataset.sample_rate = d_data.get("sample_rate", cfg.dataset.sample_rate)
    cfg.dataset.target_duration_sec = d_data.get("target_duration_sec", cfg.dataset.target_duration_sec)
    if "folds" in d_data:
        cfg.dataset.train_folds = d_data["folds"].get("train", cfg.dataset.train_folds)
        cfg.dataset.val_folds = d_data["folds"].get("val", cfg.dataset.val_folds)
        cfg.dataset.test_folds = d_data["folds"].get("test", cfg.dataset.test_folds)

    # Preprocessing
    p_data = data.get("preprocessing", {})
    cfg.preprocessing.sample_rate = p_data.get("sample_rate", cfg.preprocessing.sample_rate)
    cfg.preprocessing.n_fft = p_data.get("n_fft", cfg.preprocessing.n_fft)
    cfg.preprocessing.hop_length = p_data.get("hop_length", cfg.preprocessing.hop_length)
    cfg.preprocessing.n_mels = p_data.get("n_mels", cfg.preprocessing.n_mels)
    cfg.preprocessing.f_min = p_data.get("f_min", cfg.preprocessing.f_min)
    cfg.preprocessing.f_max = p_data.get("f_max", cfg.preprocessing.f_max)
    cfg.preprocessing.top_db = p_data.get("top_db", cfg.preprocessing.top_db)
    cfg.preprocessing.in_channels = p_data.get("in_channels", cfg.preprocessing.in_channels)
    cfg.preprocessing.norm_mean = p_data.get("norm_mean", cfg.preprocessing.norm_mean)
    cfg.preprocessing.norm_std = p_data.get("norm_std", cfg.preprocessing.norm_std)

    # Augmentation
    a_data = data.get("augmentation", {})
    cfg.augmentation.enabled = a_data.get("enabled", cfg.augmentation.enabled)
    cfg.augmentation.time_mask_max_len = a_data.get("time_mask_max_len", cfg.augmentation.time_mask_max_len)
    cfg.augmentation.freq_mask_max_len = a_data.get("freq_mask_max_len", cfg.augmentation.freq_mask_max_len)
    cfg.augmentation.num_time_masks = a_data.get("num_time_masks", cfg.augmentation.num_time_masks)
    cfg.augmentation.num_freq_masks = a_data.get("num_freq_masks", cfg.augmentation.num_freq_masks)
    cfg.augmentation.gain_range = a_data.get("gain_range", cfg.augmentation.gain_range)
    cfg.augmentation.time_shift_max_ratio = a_data.get("time_shift_max_ratio", cfg.augmentation.time_shift_max_ratio)

    # Model
    m_data = data.get("model", {})
    cfg.model.backbone = m_data.get("backbone", cfg.model.backbone)
    cfg.model.pretrained = m_data.get("pretrained", cfg.model.pretrained)
    cfg.model.num_classes = m_data.get("num_classes", cfg.model.num_classes)
    cfg.model.dropout = m_data.get("dropout", cfg.model.dropout)

    # Training
    t_data = data.get("training", {})
    cfg.training.batch_size = t_data.get("batch_size", cfg.training.batch_size)
    cfg.training.num_workers = t_data.get("num_workers", cfg.training.num_workers)
    cfg.training.epochs = t_data.get("epochs", cfg.training.epochs)
    cfg.training.early_stopping_patience = t_data.get("early_stopping_patience", cfg.training.early_stopping_patience)
    cfg.training.gradient_clip_norm = t_data.get("gradient_clip_norm", cfg.training.gradient_clip_norm)
    cfg.training.label_smoothing = t_data.get("label_smoothing", cfg.training.label_smoothing)
    lr_info = t_data.get("learning_rate", {})
    if isinstance(lr_info, dict):
        cfg.training.lr_backbone = lr_info.get("backbone", cfg.training.lr_backbone)
        cfg.training.lr_head = lr_info.get("head", cfg.training.lr_head)
    elif isinstance(lr_info, (int, float)):
        cfg.training.lr_head = float(lr_info)
        cfg.training.lr_backbone = float(lr_info) * 0.1
    cfg.training.weight_decay = t_data.get("weight_decay", cfg.training.weight_decay)

    # Paths
    pt_data = data.get("paths", {})
    cfg.paths.checkpoints_dir = pt_data.get("checkpoints_dir", cfg.paths.checkpoints_dir)
    cfg.paths.final_model_dir = pt_data.get("final_model_dir", cfg.paths.final_model_dir)
    cfg.paths.figures_dir = pt_data.get("figures_dir", cfg.paths.figures_dir)
    cfg.paths.metrics_dir = pt_data.get("metrics_dir", cfg.paths.metrics_dir)
    cfg.paths.predictions_dir = pt_data.get("predictions_dir", cfg.paths.predictions_dir)

    return cfg
