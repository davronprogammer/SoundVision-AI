"""Error analysis module for detailed model diagnostic inspection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def perform_error_analysis(
    predictions_df: pd.DataFrame,
    output_csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyzes prediction discrepancies, finds confused pairs, and extracts sample errors."""
    misclassified = predictions_df[~predictions_df["is_correct"]].copy()

    total_samples = len(predictions_df)
    error_count = len(misclassified)
    error_rate = error_count / total_samples if total_samples > 0 else 0.0

    # Confused pairs: (true_class, predicted_class)
    confused_pairs = (
        misclassified.groupby(["true_class", "predicted_class"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    # Class-level error counts
    class_errors = (
        misclassified.groupby("true_class")
        .size()
        .reset_index(name="errors")
        .sort_values(by="errors", ascending=False)
    )

    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        misclassified.to_csv(output_csv_path, index=False)

    return {
        "total_samples": total_samples,
        "error_count": error_count,
        "error_rate": error_rate,
        "confused_pairs": confused_pairs.to_dict(orient="records"),
        "class_errors": class_errors.to_dict(orient="records"),
        "misclassified_samples": misclassified,
    }


def generate_error_markdown(
    analysis_results: Dict[str, Any],
    model_name: str,
) -> str:
    """Generates markdown report content for docs/ERROR_ANALYSIS.md."""
    total = analysis_results["total_samples"]
    errors = analysis_results["error_count"]
    acc = (total - errors) / total * 100 if total > 0 else 0.0

    md = [
        f"# SoundVision AI — Error Analysis Report",
        f"",
        f"**Model Evaluated**: {model_name}  ",
        f"**Test Set Samples**: {total}  ",
        f"**Misclassified Samples**: {errors}  ",
        f"**Empirical Accuracy**: {acc:.2f}%  ",
        f"",
        f"## 1. Top Confused Acoustic Pairs",
        f"",
        f"| True Category | Predicted Category | Error Count | Acoustic Hypothesis |",
        f"|---|---|---:|---|",
    ]

    acoustic_hypotheses = {
        ("airplane", "helicopter"): "Broadband engine drone with harmonic rotor modulation overlapping jet thrust spectra.",
        ("helicopter", "airplane"): "Periodic blade slaps smoothed by distance attenuation into continuous low-frequency roar.",
        ("wind", "thunderstorm"): "Continuous low-pass filtered brown noise lacking sharp lightning transient attacks.",
        ("thunderstorm", "wind"): "Distant thunder rumbles resemble ambient turbulent aerodynamic wind gusts.",
        ("fireworks", "thunderstorm"): "Impulsive explosion transients share rapid rise time and explosive low-frequency decay.",
        ("dog", "sheep"): "Harmonic vocalization with pitch bursts and formant transitions sharing spectral overlap.",
    }

    for item in analysis_results["confused_pairs"][:10]:
        t_cls = item["true_class"]
        p_cls = item["predicted_class"]
        cnt = item["count"]
        hypo = acoustic_hypotheses.get(
            (t_cls, p_cls),
            "Shared acoustic envelope, similar frequency distribution or environmental background noise."
        )
        md.append(f"| `{t_cls}` | `{p_cls}` | {cnt} | {hypo} |")

    md.extend([
        f"",
        f"## 2. Per-Class Error Breakdown",
        f"",
        f"| Category | Total Errors |",
        f"|---|---:|",
    ])

    for item in analysis_results["class_errors"]:
        md.append(f"| `{item['true_class']}` | {item['errors']} |")

    md.extend([
        f"",
        f"## 3. Notable Misclassified Examples",
        f"",
        f"| Filename | True Label | Predicted Label | Confidence | Top-3 Alternatives |",
        f"|---|---|---|---:|---|",
    ])

    for _, row in analysis_results["misclassified_samples"].head(15).iterrows():
        top3_str = ", ".join([f"{x['class']} ({x['probability']:.2f})" for x in row["top_3"]])
        md.append(f"| `{row['filename']}` | `{row['true_class']}` | `{row['predicted_class']}` | {row['confidence']:.3f} | {top3_str} |")

    return "\n".join(md)
