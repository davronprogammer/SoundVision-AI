"""CLI entry point for running audio prediction with SoundVision AI."""

import argparse
import json
from pathlib import Path
from soundvision.inference.predictor import AudioPredictor


def main():
    parser = argparse.ArgumentParser(description="Run SoundVision AI prediction on a WAV file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/final/soundvision_resnet18.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda or cpu)")
    args = parser.parse_args()

    predictor = AudioPredictor(checkpoint_path=args.checkpoint, device=args.device)
    result = predictor.predict(audio_input=args.audio, top_k=args.top_k)

    print("\n" + "=" * 50)
    print(" SOUNDVISION AI — PREDICTION RESULT")
    print("=" * 50)
    print(f"File:            {Path(args.audio).name}")
    print(f"Predicted Class: {result['predicted_class'].upper()}")
    print(f"Confidence:      {result['confidence'] * 100:.2f}%")
    print("\nTop Candidates:")
    for i, pred in enumerate(result["top_predictions"], 1):
        bar = "█" * int(pred["probability"] * 30)
        print(f"  {i}. {pred['class']:<15} {pred['probability'] * 100:6.2f}%  | {bar}")
    print("=" * 50)
    print(f"Note: {result['disclaimer']}\n")


if __name__ == "__main__":
    main()
