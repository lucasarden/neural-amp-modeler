"""
Inspect audio files to get their properties
"""

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def inspect_audio_file(filepath):
    """Print detailed info about an audio file"""
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    # Load the file
    audio, sample_rate = sf.read(filepath)

    # Basic info
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"Total Samples: {len(audio):,}")
    print(f"Channels: {audio.ndim} ({'mono' if audio.ndim == 1 else 'stereo'})")
    print(f"Data Type: {audio.dtype}")
    print(f"Bit Depth: FP32 (32-bit floating point)")

    # Amplitude info
    print(f"\nAmplitude Stats:")
    print(f"  Min: {audio.min():.6f}")
    print(f"  Max: {audio.max():.6f}")
    print(f"  Mean: {audio.mean():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.6f}")

    # Check if normalized
    max_abs = np.abs(audio).max()
    print(f"  Peak Absolute: {max_abs:.6f}")
    if max_abs <= 1.0:
        print(f"  ✓ Already normalized to [-1, 1]")
    else:
        print(f"  ✗ NOT normalized (exceeds [-1, 1] range)")

    # Suggested segment lengths
    print(f"\nSuggested Segment Lengths:")
    for length in [4096, 8192, 16384]:
        num_segments = len(audio) // length
        print(
            f"  {length} samples ({length/sample_rate*1000:.1f}ms): {num_segments} segments"
        )

    return audio, sample_rate


def compare_files(input_file, output_file):
    """Compare input and output files"""
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    audio_in, sr_in = sf.read(input_file)
    audio_out, sr_out = sf.read(output_file)

    # Check sample rates match
    if sr_in == sr_out:
        print(f"✓ Sample rates match: {sr_in} Hz")
    else:
        print(f"✗ Sample rate mismatch: input={sr_in} Hz, output={sr_out} Hz")

    # Check lengths match
    if len(audio_in) == len(audio_out):
        print(f"✓ Lengths match: {len(audio_in):,} samples")
    else:
        print(f"✗ Length mismatch: input={len(audio_in):,}, output={len(audio_out):,}")
        print(f"  Difference: {abs(len(audio_in) - len(audio_out)):,} samples")

    # Check if they're aligned (correlation)
    if len(audio_in) == len(audio_out):
        correlation = np.corrcoef(audio_in, audio_out)[0, 1]
        print(f"\nCorrelation: {correlation:.4f}")
        if correlation > 0.5:
            print("✓ Files appear to be aligned (good correlation)")
        else:
            print("⚠ Low correlation - files might not be aligned")


def plot_waveforms(input_file, output_file, duration=2.0):
    """Plot first few seconds of both waveforms"""
    audio_in, sr_in = sf.read(input_file)
    audio_out, sr_out = sf.read(output_file)

    # Get first N seconds
    samples = int(duration * sr_in)
    audio_in = audio_in[:samples]
    audio_out = audio_out[:samples]

    # Time axis
    time = np.arange(len(audio_in)) / sr_in

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.plot(time, audio_in, linewidth=0.5)
    ax1.set_title("Input (Clean DI)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, audio_out, linewidth=0.5)
    ax2.set_title("Output (Processed)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("waveform_comparison.png", dpi=150)
    print(f"\n✓ Waveform plot saved to: waveform_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Inspect both files
    input_file = "data/raw/flstudio_test1_in.wav"
    output_file = "data/raw/flstudio_test1_out.wav"

    print("\n" + "#" * 60)
    print("# AUDIO FILE INSPECTION")
    print("#" * 60)

    # Inspect input
    audio_in, sr_in = inspect_audio_file(input_file)

    # Inspect output
    audio_out, sr_out = inspect_audio_file(output_file)

    # Compare them
    compare_files(input_file, output_file)

    # Plot (optional - comment out if you don't want the plot)
    print("\nGenerating waveform comparison plot...")
    try:
        plot_waveforms(input_file, output_file, duration=2.0)
    except Exception as e:
        print(f"Could not generate plot: {e}")
