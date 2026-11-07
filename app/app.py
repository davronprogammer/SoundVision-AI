"""SoundVision AI — Environmental Sound Intelligence Web Application."""

import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root, src, and app dir are in sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for directory in (SRC_DIR, PROJECT_ROOT, APP_DIR):
    dir_str = str(directory)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchaudio.transforms as T

from components.ui import (
    CLASS_ICONS,
    apply_custom_css,
    plot_timeline,
    plot_top_predictions_bar,
    plot_waveform_and_spectrogram,
    read_audio,
)
from soundvision.config import load_config
from soundvision.data.dataset import SELECTED_CLASSES
from soundvision.data.preprocessing import AudioPreprocessor
from soundvision.inference.predictor import AudioPredictor

# Confidence threshold for out-of-set / low-confidence warnings
CONFIDENCE_THRESHOLD = 0.50

# Page Configuration
st.set_page_config(
    page_title="SoundVision AI — Environmental Sound Intelligence",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Cache model predictor safely (never cache None)
@st.cache_resource(show_spinner="Loading SoundVision Deep Learning Model...")
def load_predictor(checkpoint_path: str) -> AudioPredictor:
    """Instantiates and caches the production AudioPredictor."""
    return AudioPredictor(checkpoint_path=checkpoint_path, device="cpu")


# Paths
config_path = PROJECT_ROOT / "configs" / "config.yaml"
cfg = load_config(str(config_path))
default_checkpoint = PROJECT_ROOT / "models" / "final" / "soundvision_resnet18.pth"
samples_dir = PROJECT_ROOT / "data" / "samples"
raw_audio_dir = PROJECT_ROOT / "data" / "raw" / "ESC-50-master" / "audio"
raw_meta_path = PROJECT_ROOT / "data" / "raw" / "ESC-50-master" / "meta" / "esc50.csv"

# Model Checkpoint Availability
checkpoint_exists = default_checkpoint.is_file()
if checkpoint_exists:
    predictor = load_predictor(str(default_checkpoint))
    ckpt_size_mb = default_checkpoint.stat().st_size / (1024 * 1024)
else:
    predictor = None
    ckpt_size_mb = 0.0

# Sidebar Configuration
st.sidebar.title("🔊 SoundVision AI")
st.sidebar.caption("v1.0.0 | Environmental Audio Classifier")

# Language toggle (English / O'zbekcha)
lang = st.sidebar.selectbox("🌐 Interface Language / Til", ["English", "O'zbekcha"], index=0)

i18n = {
    "English": {
        "nav": "Navigation",
        "overview": "Overview",
        "classifier": "Audio Classifier",
        "eval": "Evaluation Dashboard",
        "method": "Model & Methodology",
        "target_cats": "Target Categories (10 Classes):",
        "model_status": "System & Model Status",
        "model_ready": "Ready (ResNet-18)",
        "model_missing": "Checkpoint Missing",
        "upload_title": "Audio Classifier & Live Inference",
        "upload_subtitle": "Upload an audio recording (WAV / MP3), try a bundled sample, or use your microphone.",
        "input_method": "Choose Input Method",
        "method_upload": "Upload Audio File (WAV / MP3)",
        "method_sample": "Try Bundled Demo Sample",
        "method_esc50": "Select Sample from ESC-50 Dataset",
        "method_mic": "Record with Microphone",
        "upload_prompt": "Choose a WAV or MP3 audio file",
        "pick_sample": "Pick an environmental sound sample:",
        "vis_header": "Acoustic Signal Visualization",
        "pred_header": "Model Prediction",
        "predicted_label": "Predicted Sound Category",
        "top3_label": "Top Classification Probabilities",
        "timeline_header": "Temporal Window Timeline",
        "silent_warn": "The provided audio is silent or contains near-zero acoustic energy. Please upload an audible recording.",
        "low_conf_warn": f"Low prediction confidence (< {int(CONFIDENCE_THRESHOLD * 100)}%). This recording may belong to an unmodeled category outside the 10 target classes or contain heavy ambient noise.",
        "model_missing_err": "Model checkpoint not found. Inference cannot proceed.",
    },
    "O'zbekcha": {
        "nav": "Bo'limlar",
        "overview": "Umumiy ma'lumot",
        "classifier": "Audio Klassifikator",
        "eval": "Baholash Paneli",
        "method": "Model va Metodologiya",
        "target_cats": "Maqsadli Ovoz Sinflari (10 ta):",
        "model_status": "Tizim va Model Holati",
        "model_ready": "Tayyor (ResNet-18)",
        "model_missing": "Model fayli topilmadi",
        "upload_title": "Audio Klassifikator va Jonli Bashorat",
        "upload_subtitle": "Ovoz yozuvini yuklang (WAV / MP3), tayyor namunani sinab ko'ring yoki mikrofon orqali yozib oling.",
        "input_method": "Kiritish usulini tanlang",
        "method_upload": "Audio fayl yuklash (WAV / MP3)",
        "method_sample": "Tayyor namunani sinash",
        "method_esc50": "ESC-50 to'plamidan tanlash",
        "method_mic": "Mikrofon orqali yozish",
        "upload_prompt": "WAV yoki MP3 formatidagi audio faylni tanlang",
        "pick_sample": "Atrof-muhit ovozi namunasini tanlang:",
        "vis_header": "Akustik Signal Vizualizatsiyasi",
        "pred_header": "Model Bashorati",
        "predicted_label": "Aniqlangan Ovoz Turi",
        "top3_label": "Yuqori Ehtimollikdagi Sinflar",
        "timeline_header": "Vaqt bo'yicha Segmentlar Tahlili",
        "silent_warn": "Taqdim etilgan audio jimjit yoki akustik signali yetarli emas. Iltimos, eshitilarli audio kiriting.",
        "low_conf_warn": f"Bashorat ishonchliligi past (< {int(CONFIDENCE_THRESHOLD * 100)}%). Ushbu audio 10 ta asosiy sinfga tegishli bo'lmasligi yoki shovqin ko'p bo'lishi mumkin.",
        "model_missing_err": "Model fayli topilmadi. Bashorat qilib bo'lmaydi.",
    },
}[lang]

st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    i18n["nav"],
    [i18n["overview"], i18n["classifier"], i18n["eval"], i18n["method"]],
)

# Sidebar model status badge
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{i18n['model_status']}**")
if checkpoint_exists and predictor is not None:
    st.sidebar.markdown(
        f'<span class="status-badge status-badge-ok">● {i18n["model_ready"]}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Path: `models/final/{default_checkpoint.name}`\n\nSize: **{ckpt_size_mb:.1f} MB** | Device: **CPU**")
else:
    st.sidebar.markdown(
        f'<span class="status-badge status-badge-warn">● {i18n["model_missing"]}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Path: `models/final/{default_checkpoint.name}`")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{i18n['target_cats']}**")
for cls in SELECTED_CLASSES:
    icon = CLASS_ICONS.get(cls, "🎵")
    st.sidebar.markdown(f"- {icon} `{cls}`")

st.sidebar.markdown("---")
st.sidebar.caption("SoundVision AI v1.0.0 | ESC-50 Transfer Learning")
st.sidebar.caption("[GitHub Repository](https://github.com/davronprogammer/SoundVision-AI) | MIT License")


# TAB 1: OVERVIEW
if app_mode == i18n["overview"]:
    st.markdown('<div class="main-header">SoundVision AI — Environmental Sound Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Production-grade deep learning system for environmental acoustic classification '
        'using transfer learning on log-Mel spectrogram representations.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">10</div><div class="metric-label">Sound Classes</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">400</div><div class="metric-label">Benchmark Audio Clips</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">ResNet-18</div><div class="metric-label">Backbone Architecture</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-val">22,050 Hz</div><div class="metric-label">Target Sample Rate</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Supported Environmental Sound Categories")
    cols = st.columns(5)
    for i, cls in enumerate(SELECTED_CLASSES):
        with cols[i % 5]:
            icon = CLASS_ICONS.get(cls, "🎵")
            st.info(f"{icon} **{cls.capitalize()}**")

    st.markdown("### Key Technical Highlights")
    st.markdown(
        """
        - **Acoustic Preprocessing**: Mono conversion, 22,050 Hz resampling, STFT with 1024-point FFT and 512-point hop, 128 Mel bands, and decibel logarithmic power scaling.
        - **Zero Data Leakage**: Global mean and standard deviation normalization statistics are calculated **strictly from training folds (1–3)** and frozen for inference.
        - **Transfer Learning Strategies**: Systematic comparative evaluation across three paradigms: From-Scratch initialization, Frozen Feature Extraction, and Full Backbone Fine-Tuning with discriminative learning rates.
        - **Inference Reliability**: Supports arbitrary audio duration with sliding-window segmentation, input signal validation, and top-$k$ predictive rankings.
        """
    )


# TAB 2: AUDIO CLASSIFIER
elif app_mode == i18n["classifier"]:
    st.markdown(f'<div class="main-header">{i18n["upload_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{i18n["upload_subtitle"]}</div>', unsafe_allow_html=True)

    if predictor is None:
        st.error(
            "⚠️ **Model Checkpoint Missing**: `models/final/soundvision_resnet18.pth` was not found on disk. "
            "Please ensure the model file is present or pulled from git."
        )

    col_up, col_preview = st.columns([1, 1])

    audio_bytes: Optional[bytes] = None
    fn_label: str = ""

    with col_up:
        input_options = [i18n["method_upload"], i18n["method_sample"]]
        if raw_meta_path.is_file() and raw_audio_dir.is_dir():
            input_options.append(i18n["method_esc50"])
        if hasattr(st, "audio_input"):
            input_options.append(i18n["method_mic"])

        input_choice = st.radio(i18n["input_method"], input_options)

        if input_choice == i18n["method_upload"]:
            uploaded_file = st.file_uploader(i18n["upload_prompt"], type=["wav", "mp3"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.getvalue()
                fn_label = uploaded_file.name

        elif input_choice == i18n["method_sample"]:
            if samples_dir.is_dir():
                sample_files = sorted([f.stem for f in samples_dir.glob("*.wav")])
                selected_sample = st.selectbox(
                    i18n["pick_sample"],
                    sample_files,
                    format_func=lambda s: f"{CLASS_ICONS.get(s, '🎵')} {s.capitalize()}",
                )
                sample_path = samples_dir / f"{selected_sample}.wav"
                if sample_path.is_file():
                    with open(sample_path, "rb") as f:
                        audio_bytes = f.read()
                    fn_label = sample_path.name
            else:
                st.info("No bundled sample clips found in `data/samples/`.")

        elif input_choice == i18n["method_esc50"]:
            df = pd.read_csv(raw_meta_path)
            subset = df[df["category"].isin(SELECTED_CLASSES)]
            sample_filenames = subset["filename"].tolist()
            selected_fn = st.selectbox("Choose sample audio clip from ESC-50:", sample_filenames)
            preset_path = raw_audio_dir / selected_fn
            if preset_path.is_file():
                with open(preset_path, "rb") as f:
                    audio_bytes = f.read()
                fn_label = preset_path.name

        elif hasattr(st, "audio_input") and input_choice == i18n["method_mic"]:
            mic_audio = st.audio_input("Record audio sample")
            if mic_audio is not None:
                audio_bytes = mic_audio.getvalue()
                fn_label = "microphone_recording.wav"

    if audio_bytes is not None:
        try:
            # Shared robust audio decoding (soundfile with librosa fallback)
            data, sr = read_audio(audio_bytes, filename=fn_label)
            duration = len(data) / sr

            # Stereo to Mono conversion
            if data.ndim > 1 and data.shape[1] > 1:
                mono_data = np.mean(data, axis=1)
                channels_count = data.shape[1]
            elif data.ndim > 1:
                mono_data = data[:, 0]
                channels_count = 1
            else:
                mono_data = data
                channels_count = 1

            # Check for silent or empty audio
            if np.max(np.abs(mono_data)) < 1e-4:
                st.warning(f"⚠️ {i18n['silent_warn']}")
            else:
                audio_format = "audio/mpeg" if fn_label.lower().endswith(".mp3") else "audio/wav"

                with col_preview:
                    st.markdown(f"**Filename:** `{fn_label}`")
                    st.markdown(
                        f"**Duration:** {duration:.2f} s | **Sample Rate:** {sr:,} Hz | **Channels:** {channels_count}"
                    )
                    st.audio(audio_bytes, format=audio_format)

                # Resample waveform if necessary to target 22,050 Hz
                wave_tensor = torch.from_numpy(mono_data).float().unsqueeze(0)
                if sr != 22050:
                    wave_tensor = T.Resample(orig_freq=sr, new_freq=22050)(wave_tensor)

                # Preprocess for spectrogram visualization (using standard 5.0s window for preview)
                preprocessor = AudioPreprocessor(sample_rate=22050, target_duration=5.0)
                adjusted_tensor = preprocessor.adjust_duration(wave_tensor)
                mel_db = preprocessor.extract_mel_spectrogram(adjusted_tensor).squeeze(0).numpy()

                # Visualizations
                st.markdown(f"### {i18n['vis_header']}")
                fig = plot_waveform_and_spectrogram(mono_data, sr, mel_db)
                st.pyplot(fig)
                plt.close(fig)

                # Run Prediction
                if predictor is not None:
                    st.markdown(f"### {i18n['pred_header']}")
                    with st.spinner("Analyzing acoustic features through ResNet-18..."):
                        result = predictor.predict(wave_tensor, top_k=3)

                    pred_class = result["predicted_class"]
                    conf = result["confidence"]
                    pred_icon = CLASS_ICONS.get(pred_class, "🎵")

                    # Primary Prediction Card
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <div style="font-size: 0.95rem; color: #4a5568;">{i18n['predicted_label']}</div>
                            <div style="font-size: 2.1rem; font-weight: 800; color: #2b6cb0; text-transform: uppercase; margin-top: 2px;">
                                {pred_icon} {pred_class}
                            </div>
                            <div style="font-size: 1.15rem; font-weight: 600; color: #2d3748; margin-top: 6px;">
                                Confidence: {conf * 100:.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Low-confidence / Out-of-set Warning
                    if conf < CONFIDENCE_THRESHOLD:
                        st.warning(f"⚠️ {i18n['low_conf_warn']}")

                    # Top-3 Predictions Visualization
                    st.markdown(f"**{i18n['top3_label']}:**")
                    col_bar, col_vals = st.columns([1.4, 1])
                    with col_bar:
                        bar_fig = plot_top_predictions_bar(result["top_predictions"])
                        st.pyplot(bar_fig)
                        plt.close(bar_fig)

                    with col_vals:
                        for item in result["top_predictions"]:
                            c_name = item["class"]
                            c_icon = CLASS_ICONS.get(c_name, "🎵")
                            c_prob = item["probability"]
                            st.write(f"{c_icon} **{c_name.capitalize()}**: `{c_prob * 100:.2f}%`")
                            st.progress(float(min(max(c_prob, 0.0), 1.0)))

                    # Sliding-window Timeline for audio longer than 5 seconds
                    if duration > 5.0:
                        st.markdown(f"### {i18n['timeline_header']}")
                        timeline_data = predictor.predict_timeline(wave_tensor, window_sec=5.0, hop_sec=2.5)
                        timeline_fig = plot_timeline(timeline_data)
                        st.pyplot(timeline_fig)
                        plt.close(timeline_fig)

                        with st.expander("Detailed Window Breakdown"):
                            t_df = pd.DataFrame([
                                {
                                    "Segment": f"{w['start_sec']:.1f}s - {w['end_sec']:.1f}s",
                                    "Predicted Class": w["predicted_class"].capitalize(),
                                    "Confidence": f"{w['confidence'] * 100:.1f}%",
                                }
                                for w in timeline_data
                            ])
                            st.dataframe(t_df, use_container_width=True)

                    # Export / Download Buttons
                    st.markdown("---")
                    col_dl_json, col_dl_csv = st.columns(2)
                    with col_dl_json:
                        json_str = json.dumps(
                            {
                                "filename": fn_label,
                                "duration_seconds": round(duration, 2),
                                "sample_rate": sr,
                                "prediction": result,
                            },
                            indent=2,
                        )
                        st.download_button(
                            label="📥 Download Prediction (JSON)",
                            data=json_str,
                            file_name=f"soundvision_{Path(fn_label).stem}_prediction.json",
                            mime="application/json",
                        )

                    with col_dl_csv:
                        pred_df = pd.DataFrame([
                            {
                                "filename": fn_label,
                                "rank": rank + 1,
                                "class": item["class"],
                                "probability": item["probability"],
                            }
                            for rank, item in enumerate(result["top_predictions"])
                        ])
                        st.download_button(
                            label="📥 Download Prediction (CSV)",
                            data=pred_df.to_csv(index=False),
                            file_name=f"soundvision_{Path(fn_label).stem}_prediction.csv",
                            mime="text/csv",
                        )

                    st.caption(f"ℹ️ {result['disclaimer']}")
                else:
                    st.error(i18n["model_missing_err"])

        except Exception as e:
            st.error(f"⚠️ Error processing audio: {e}")


# TAB 3: EVALUATION DASHBOARD
elif app_mode == i18n["eval"]:
    st.markdown('<div class="main-header">Evaluation & Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Empirical benchmark metrics evaluated strictly on held-out test data (Fold 5).</div>',
        unsafe_allow_html=True,
    )

    metrics_file = PROJECT_ROOT / "results" / "metrics" / "test_metrics.json"
    exp_csv_file = PROJECT_ROOT / "results" / "metrics" / "experiment_results.csv"

    if metrics_file.is_file():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{metrics["accuracy"] * 100:.1f}%</div>'
                f'<div class="metric-label">Test Accuracy</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{metrics["macro_f1"] * 100:.1f}%</div>'
                f'<div class="metric-label">Macro F1 Score</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{metrics["weighted_f1"] * 100:.1f}%</div>'
                f'<div class="metric-label">Weighted F1 Score</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{metrics["macro_precision"] * 100:.1f}%</div>'
                f'<div class="metric-label">Macro Precision</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(
            "ℹ️ **Evaluation Note**: Test metrics are evaluated strictly on Fold 5 (~8 audio clips per class, 80 total). "
            "Because sample counts per class are relatively compact, individual per-class metrics can exhibit noticeable variance."
        )

        # Honest Weak Classes Analysis
        if "per_class_f1" in metrics:
            sorted_f1 = sorted(metrics["per_class_f1"].items(), key=lambda x: x[1])
            weakest_1, weakest_2 = sorted_f1[0], sorted_f1[1]
            st.info(
                f"📊 **Observed Class Weaknesses**: Held-out evaluation identified lowest F1 scores on "
                f"**{weakest_1[0].capitalize()}** ({weakest_1[1] * 100:.1f}%) and **{weakest_2[0].capitalize()}** ({weakest_2[1] * 100:.1f}%). "
                f"Acoustic confusion analysis indicates overlapping temporal transients with thunderstorm/train and broadband diffuse energy with wind."
            )

    st.markdown("### Comparative Experiment History")
    if exp_csv_file.is_file():
        exp_df = pd.read_csv(exp_csv_file)
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("No experiment logs found yet. Run `python scripts/train.py` to record experiments.")

    # Figures
    st.markdown("### Performance Visualizations")
    col_cm, col_f1 = st.columns(2)

    cm_img_path = PROJECT_ROOT / "results" / "figures" / "confusion_matrix_test.png"
    f1_img_path = PROJECT_ROOT / "results" / "figures" / "per_class_f1_test.png"

    with col_cm:
        if cm_img_path.is_file():
            st.image(str(cm_img_path), caption="Normalized Confusion Matrix (Test Fold 5)", use_container_width=True)
        else:
            st.info("Confusion matrix figure not generated yet. Run `python scripts/evaluate.py`.")

    with col_f1:
        if f1_img_path.is_file():
            st.image(str(f1_img_path), caption="Per-Class F1 Score Breakdown (Test Fold 5)", use_container_width=True)
        else:
            st.info("Per-class F1 figure not generated yet. Run `python scripts/evaluate.py`.")


# TAB 4: MODEL & METHODOLOGY
elif app_mode == i18n["method"]:
    st.markdown('<div class="main-header">Model Architecture & Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detailed technical specification of feature representations, transfer learning, '
        'and training optimizations.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 1. Acoustic Signal Processing Pipeline
        ```text
        Raw Audio (WAV/MP3) -> Mono Conversion -> Resample (22,050 Hz) -> Windowing / Pad (5.0s)
               │
               ▼
        Short-Time Fourier Transform (STFT) [N_FFT=1024, Hop=512, Hann Window]
               │
               ▼
        Mel Filterbank Projection [128 Mel Bins, f_min=0 Hz, f_max=11,025 Hz]
               │
               ▼
        Logarithmic Power Scaling [AmplitudeToDB, Top_dB=80]
               │
               ▼
        Z-Score Normalization [(x - μ_train) / (σ_train + ε)] (Folds 1-3 only)
               │
               ▼
        Multi-Channel Tensor Expansion [3, 128, 216] -> ResNet-18 Deep Neural Network
        ```

        ### 2. Experimental Paradigms
        - **Experiment A (From Scratch)**: Randomly initialized ResNet-18 trained exclusively on environmental audio.
        - **Experiment B (Feature Extraction)**: ImageNet-pretrained backbone with frozen convolutional layers. BatchNorm layers kept in eval mode to prevent running statistic degradation.
        - **Experiment C (Fine-Tuning)**: Pretrained backbone fine-tuned end-to-end with discriminative learning rates (`1e-4` for backbone, `1e-3` for custom head), SpecAugment, AdamW with weight decay `1e-4`, and Cosine Annealing learning rate scheduling.

        ### 3. Known Limitations & Responsible AI
        - **Closed-Set Assumption**: The model produces softmax probability distributions over its 10 trained categories. Unseen classes (e.g. car horns, human speech) will be assigned to whichever of the 10 classes has the nearest acoustic embedding.
        - **Audio Quality Sensitivity**: Severe microphone clipping, low bitrates, or heavy non-stationary background noise can shift Mel-spectrogram energy distributions.
        """
    )
