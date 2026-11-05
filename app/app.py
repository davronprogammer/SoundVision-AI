"""SoundVision AI — Environmental Sound Intelligence Web Application."""

import json
from pathlib import Path
import sys

# Ensure both project root and app dir are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch

try:
    from components.ui import apply_custom_css, plot_waveform_and_spectrogram
except (ModuleNotFoundError, ImportError):
    from app.components.ui import apply_custom_css, plot_waveform_and_spectrogram

from soundvision.config import load_config
from soundvision.data.dataset import SELECTED_CLASSES
from soundvision.data.preprocessing import AudioPreprocessor
from soundvision.inference.predictor import AudioPredictor

# Page Configuration
st.set_page_config(
    page_title="SoundVision AI — Environmental Sound Intelligence",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Cache model predictor
@st.cache_resource(show_spinner="Loading SoundVision Deep Learning Model...")
def get_predictor(checkpoint_path: str):
    if not Path(checkpoint_path).is_file():
        return None
    return AudioPredictor(checkpoint_path=checkpoint_path, device="cpu")


cfg = load_config()
default_checkpoint = "models/final/soundvision_resnet18.pth"
predictor = get_predictor(default_checkpoint)

# Sidebar
st.sidebar.title("🔊 SoundVision AI")
st.sidebar.caption("v1.0.0 | Environmental Audio Classifier")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    ["Overview", "Audio Classifier", "Evaluation Dashboard", "Model & Methodology"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Target Categories (10 Classes):**")
for cls in SELECTED_CLASSES:
    st.sidebar.markdown(f"- `{cls}`")

st.sidebar.markdown("---")
st.sidebar.caption("Trained on ESC-50 benchmark | PyTorch ResNet-18")


# TAB 1: OVERVIEW
if app_mode == "Overview":
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
    icons = ["🐕", "⛈️", "🎆", "✈️", "🚆", "🐑", "💨", "🐸", "🚁", "😂"]
    for i, cls in enumerate(SELECTED_CLASSES):
        with cols[i % 5]:
            st.info(f"{icons[i]} **{cls.capitalize()}**")

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
elif app_mode == "Audio Classifier":
    st.markdown('<div class="main-header">Audio Classifier & Live Inference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a WAV audio recording or pick an evaluation sample to inspect waveforms, '
        'spectrograms, and model confidence scores.</div>',
        unsafe_allow_html=True,
    )

    if predictor is None:
        st.warning(
            "Final model checkpoint not found at `models/final/soundvision_resnet18.pth`. "
            "Please train or evaluate a model first using `python scripts/train.py`."
        )

    col_up, col_preview = st.columns([1, 1])

    audio_source = None
    uploaded_file = None
    preset_path = None

    with col_up:
        input_choice = st.radio("Choose Input Method", ["Upload WAV Audio", "Select Sample from Dataset"])

        if input_choice == "Upload WAV Audio":
            uploaded_file = st.file_uploader("Upload audio file", type=["wav"])
            if uploaded_file is not None:
                audio_source = uploaded_file
        else:
            audio_dir = Path("data/raw/ESC-50-master/audio")
            meta_path = Path("data/raw/ESC-50-master/meta/esc50.csv")
            if meta_path.is_file() and audio_dir.is_dir():
                df = pd.read_csv(meta_path)
                subset = df[df["category"].isin(SELECTED_CLASSES)]
                sample_filenames = subset["filename"].tolist()
                selected_fn = st.selectbox("Choose sample audio clip:", sample_filenames)
                preset_path = audio_dir / selected_fn
                audio_source = preset_path
            else:
                st.info("Dataset files not detected in `data/raw/`. Please upload a custom WAV file.")

    if audio_source is not None:
        try:
            if uploaded_file:
                audio_bytes = uploaded_file.getvalue()
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
                fn_label = uploaded_file.name
            else:
                with open(preset_path, "rb") as f:
                    audio_bytes = f.read()
                data, sr = sf.read(str(preset_path), dtype="float32", always_2d=True)
                fn_label = preset_path.name

            duration = len(data) / sr
            mono_data = np.mean(data, axis=1)

            with col_preview:
                st.markdown(f"**Filename:** `{fn_label}`")
                st.markdown(f"**Duration:** {duration:.2f} seconds | **Sample Rate:** {sr:,} Hz | **Channels:** {data.shape[1]}")
                st.audio(audio_bytes, format="audio/wav")

            # Extract Mel Spectrogram for preview
            preprocessor = AudioPreprocessor(sample_rate=22050, target_duration=5.0)
            wave_tensor = torch.from_numpy(mono_data).unsqueeze(0)
            if sr != 22050:
                import torchaudio.transforms as T
                wave_tensor = T.Resample(orig_freq=sr, new_freq=22050)(wave_tensor)
            adjusted_tensor = preprocessor.adjust_duration(wave_tensor)
            mel_db = preprocessor.extract_mel_spectrogram(adjusted_tensor).squeeze(0).numpy()

            # Plot Waveform and Spectrogram
            st.markdown("### Acoustic Visualization")
            fig = plot_waveform_and_spectrogram(mono_data, sr, mel_db)
            st.pyplot(fig)
            plt.close(fig)

            # Run Prediction
            if predictor is not None:
                st.markdown("### Model Prediction")
                with st.spinner("Analyzing audio features..."):
                    result = predictor.predict(wave_tensor, top_k=3)

                pred_class = result["predicted_class"]
                conf = result["confidence"]

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div style="font-size: 0.95rem; color: #4a5568;">Predicted Sound Category</div>
                        <div style="font-size: 2rem; font-weight: 800; color: #2b6cb0; text-transform: uppercase;">
                            {pred_class}
                        </div>
                        <div style="font-size: 1.1rem; font-weight: 600; color: #2d3748; margin-top: 4px;">
                            Confidence: {conf * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("**Top-3 Classification Probabilities:**")
                for item in result["top_predictions"]:
                    c_name = item["class"].capitalize()
                    c_prob = item["probability"]
                    st.write(f"**{c_name}**: {c_prob * 100:.1f}%")
                    st.progress(float(c_prob))

                st.caption(f"ℹ️ {result['disclaimer']}")

        except Exception as e:
            st.error(f"Error processing audio: {e}")


# TAB 3: EVALUATION DASHBOARD
elif app_mode == "Evaluation Dashboard":
    st.markdown('<div class="main-header">Evaluation & Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Empirical benchmark metrics evaluated strictly on held-out test data (Fold 5).</div>',
        unsafe_allow_html=True,
    )

    metrics_file = Path("results/metrics/test_metrics.json")
    exp_csv_file = Path("results/metrics/experiment_results.csv")

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

    st.markdown("### Comparative Experiment History")
    if exp_csv_file.is_file():
        exp_df = pd.read_csv(exp_csv_file)
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("No experiment logs found yet. Run `python scripts/train.py` to record experiments.")

    # Figures
    st.markdown("### Performance Visualizations")
    col_cm, col_f1 = st.columns(2)

    cm_img_path = Path("results/figures/confusion_matrix_test.png")
    f1_img_path = Path("results/figures/per_class_f1_test.png")

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
elif app_mode == "Model & Methodology":
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
        Raw Audio (WAV) -> Mono Conversion -> Resample (22,050 Hz) -> Windowing / Pad (5.0s)
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
