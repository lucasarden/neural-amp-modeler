"""
Compare original amp output vs model prediction
"""

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def calculate_metrics(target, prediction):
    """Calculate error metrics"""
    # Ensure same length
    min_len = min(len(target), len(prediction))
    target = target[:min_len]
    prediction = prediction[:min_len]

    # ESR (Error-to-Signal Ratio) - same as training loss
    esr = np.mean((prediction - target) ** 2) / (np.mean(target**2) + 1e-8)

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((prediction - target) ** 2))

    # Correlation
    correlation = np.corrcoef(target, prediction)[0, 1]

    return {
        'esr': esr,
        'rmse': rmse,
        'correlation': correlation
    }


def plot_comparison(target, prediction, sr, duration=2.0, output_file=None):
    """Plot waveforms and spectrograms for comparison"""
    # Get first N seconds
    samples = int(duration * sr)
    target_short = target[:samples]
    pred_short = prediction[:samples]

    time = np.arange(len(target_short)) / sr

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Waveform comparison
    axes[0].plot(time, target_short, label='Target (Real Amp)', alpha=0.7, linewidth=0.8)
    axes[0].plot(time, pred_short, label='Prediction (Model)', alpha=0.7, linewidth=0.8)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Waveform Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Target waveform
    axes[1].plot(time, target_short, color='blue', linewidth=0.8)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Target (Real Amp Output)')
    axes[1].grid(True, alpha=0.3)

    # Prediction waveform
    axes[2].plot(time, pred_short, color='orange', linewidth=0.8)
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Prediction (Model Output)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n[*] Plot saved: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare model output vs real amp")
    parser.add_argument("target", type=str, help="Target audio (real amp output)")
    parser.add_argument("prediction", type=str, help="Prediction audio (model output)")
    parser.add_argument("-p", "--plot", action="store_true", help="Show comparison plot")
    parser.add_argument("-d", "--duration", type=float, default=2.0,
                       help="Duration to plot (seconds)")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Save plot to file")

    args = parser.parse_args()

    # Load audio files
    print(f"\n[*] Loading files...")
    target, sr_target = sf.read(args.target)
    prediction, sr_pred = sf.read(args.prediction)

    # Convert stereo to mono if needed
    if target.ndim == 2:
        print(f"    [!] Target is stereo, converting to mono")
        target = np.mean(target, axis=1)
    if prediction.ndim == 2:
        print(f"    [!] Prediction is stereo, converting to mono")
        prediction = np.mean(prediction, axis=1)

    # Verify sample rates match
    if sr_target != sr_pred:
        print(f"[!] Warning: Sample rate mismatch ({sr_target} vs {sr_pred})")
        return

    print(f"    Sample rate: {sr_target} Hz")
    print(f"    Target length: {len(target):,} samples ({len(target)/sr_target:.2f}s)")
    print(f"    Prediction length: {len(prediction):,} samples ({len(prediction)/sr_pred:.2f}s)")

    # Calculate metrics
    print(f"\n{'='*60}")
    print("COMPARISON METRICS")
    print('='*60)

    metrics = calculate_metrics(target, prediction)

    print(f"ESR (Error-to-Signal Ratio): {metrics['esr']:.6f}")
    print(f"  Lower is better (training loss metric)")

    print(f"\nRMSE (Root Mean Square Error): {metrics['rmse']:.6f}")
    print(f"  Lower is better")

    print(f"\nCorrelation: {metrics['correlation']:.6f}")
    print(f"  Higher is better (1.0 = perfect match)")

    # Qualitative assessment
    if metrics['correlation'] > 0.95:
        quality = "Excellent"
    elif metrics['correlation'] > 0.90:
        quality = "Very Good"
    elif metrics['correlation'] > 0.80:
        quality = "Good"
    elif metrics['correlation'] > 0.70:
        quality = "Fair"
    else:
        quality = "Needs Improvement"

    print(f"\nOverall Quality: {quality}")
    print('='*60)

    # Plot if requested
    if args.plot or args.output:
        print(f"\n[*] Generating comparison plot...")
        plot_comparison(target, prediction, sr_target, args.duration, args.output)


if __name__ == "__main__":
    main()
