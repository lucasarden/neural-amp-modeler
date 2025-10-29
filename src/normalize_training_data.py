"""
Normalize training data to professional audio levels while preserving gain relationship.

This brings both input and output to realistic levels:
- Input: ~-18dB peak (typical guitar DI)
- Output: scaled proportionally to maintain amp's gain structure
"""

import soundfile as sf
import numpy as np
import argparse
from pathlib import Path


def normalize_to_target_level(input_file, output_file, target_input_db=-18.0):
    """
    Normalize training pair to professional levels.

    Args:
        input_file: Path to input (DI) audio
        output_file: Path to output (amp) audio
        target_input_db: Target peak level for input in dB (default: -18dB)
    """
    print("="*70)
    print("NORMALIZING TRAINING DATA TO PROFESSIONAL LEVELS")
    print("="*70)

    # Load files
    print(f"\nLoading input: {input_file}")
    input_audio, sr_in = sf.read(input_file)
    print(f"Loading output: {output_file}")
    output_audio, sr_out = sf.read(output_file)

    # Verify compatibility
    assert sr_in == sr_out, f"Sample rate mismatch: {sr_in} vs {sr_out}"
    assert len(input_audio) == len(output_audio), "Length mismatch"

    # Convert stereo to mono if needed
    if input_audio.ndim == 2:
        input_audio = np.mean(input_audio, axis=1)
        print("  Converted input to mono")
    if output_audio.ndim == 2:
        output_audio = np.mean(output_audio, axis=1)
        print("  Converted output to mono")

    # Calculate current levels
    input_peak = np.abs(input_audio).max()
    output_peak = np.abs(output_audio).max()
    input_rms = np.sqrt(np.mean(input_audio**2))
    output_rms = np.sqrt(np.mean(output_audio**2))

    print(f"\n{'BEFORE NORMALIZATION':-^70}")
    print(f"Input:  Peak = {input_peak:.6f} ({20*np.log10(input_peak):.2f} dB), RMS = {input_rms:.6f}")
    print(f"Output: Peak = {output_peak:.6f} ({20*np.log10(output_peak):.2f} dB), RMS = {output_rms:.6f}")
    print(f"Gain ratio (output/input): {output_peak/input_peak:.2f}x")

    # Calculate target peak value from dB
    target_peak = 10 ** (target_input_db / 20.0)

    # Calculate scale factor based on INPUT reaching target level
    scale_factor = target_peak / input_peak

    print(f"\n{'SCALING CALCULATION':-^70}")
    print(f"Target input level: {target_input_db} dB = {target_peak:.6f} amplitude")
    print(f"Scale factor: {scale_factor:.4f}x")

    # Apply SAME scale factor to both signals
    input_normalized = input_audio * scale_factor
    output_normalized = output_audio * scale_factor

    # Calculate new levels
    input_peak_new = np.abs(input_normalized).max()
    output_peak_new = np.abs(output_normalized).max()
    input_rms_new = np.sqrt(np.mean(input_normalized**2))
    output_rms_new = np.sqrt(np.mean(output_normalized**2))

    print(f"\n{'AFTER NORMALIZATION':-^70}")
    print(f"Input:  Peak = {input_peak_new:.6f} ({20*np.log10(input_peak_new):.2f} dB), RMS = {input_rms_new:.6f}")
    print(f"Output: Peak = {output_peak_new:.6f} ({20*np.log10(output_peak_new):.2f} dB), RMS = {output_rms_new:.6f}")
    print(f"Gain ratio (output/input): {output_peak_new/input_peak_new:.2f}x (PRESERVED!)")

    # Check for clipping
    if output_peak_new > 0.99:
        print(f"\n⚠ WARNING: Output will clip at {output_peak_new:.3f}!")
        print(f"   Consider using a lower target (e.g., -24dB for input)")
        print(f"   Or use --prevent-clipping flag")
        return None, None, sr_in

    # Create output filenames
    input_path = Path(input_file)
    output_path = Path(output_file)

    input_normalized_path = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
    output_normalized_path = output_path.parent / f"{output_path.stem}_normalized{output_path.suffix}"

    print(f"\n{'SAVING':-^70}")
    print(f"Input:  {input_normalized_path}")
    print(f"Output: {output_normalized_path}")

    # Save with same format as input
    sf.write(input_normalized_path, input_normalized, sr_in)
    sf.write(output_normalized_path, output_normalized, sr_out)

    print(f"\n{'SUCCESS':-^70}")
    print("✓ Training data normalized to professional levels!")
    print("✓ Gain relationship preserved!")
    print("\nNext steps:")
    print(f"1. Update your config.yaml to use:")
    print(f"   input_file: {input_normalized_path}")
    print(f"   output_file: {output_normalized_path}")
    print(f"2. Retrain your model")
    print("="*70)

    return input_normalized_path, output_normalized_path, sr_in


def main():
    parser = argparse.ArgumentParser(
        description="Normalize training data to professional audio levels"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input (DI) audio file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output (amp) audio file"
    )
    parser.add_argument(
        "--target-db",
        type=float,
        default=-18.0,
        help="Target input peak level in dB (default: -18.0)"
    )
    parser.add_argument(
        "--prevent-clipping",
        action="store_true",
        help="Automatically reduce target level if output would clip"
    )

    args = parser.parse_args()

    # Try normalization
    result = normalize_to_target_level(args.input, args.output, args.target_db)

    # If clipping would occur and prevent-clipping is enabled, retry with lower target
    if result[0] is None and args.prevent_clipping:
        print("\nRetrying with automatic level adjustment to prevent clipping...")
        # Load to check output level
        input_audio, _ = sf.read(args.input)
        output_audio, _ = sf.read(args.output)
        if input_audio.ndim == 2:
            input_audio = np.mean(input_audio, axis=1)
        if output_audio.ndim == 2:
            output_audio = np.mean(output_audio, axis=1)

        # Calculate safe target
        input_peak = np.abs(input_audio).max()
        output_peak = np.abs(output_audio).max()
        gain_ratio = output_peak / input_peak

        # Target should keep output under 0.95
        safe_target_peak = 0.95 / gain_ratio
        safe_target_db = 20 * np.log10(safe_target_peak)

        print(f"Using safe target: {safe_target_db:.1f} dB")
        normalize_to_target_level(args.input, args.output, safe_target_db)


if __name__ == "__main__":
    main()
