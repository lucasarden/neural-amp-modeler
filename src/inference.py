"""
Inference script for running the trained model
"""

import torch
import soundfile as sf
import yaml
import os
import argparse
import numpy as np
from pathlib import Path
from model import WaveNetAmp, StreamingWaveNetAmp


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(config, model_name=None, device="cuda"):
    """Load a trained model"""
    # Use model name from config if not specified
    if model_name is None:
        model_name = config["model"]["name"]

    # Build model path
    model_path = config["paths"]["best_model"].format(model_name=model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Initialize model architecture
    causal = config["model"].get("causal", False)
    model = WaveNetAmp(
        channels=config["model"]["channels"],
        num_layers=config["model"]["num_layers"],
        kernel_size=config["model"]["kernel_size"],
        dilation_base=config["model"]["dilation_base"],
        causal=causal,
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    print(f"[*] Loaded model: {model_path}")
    print(f"    Mode: {'Causal (Real-time)' if causal else 'Non-causal (Offline)'}")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    Receptive field: {model.receptive_field()} samples")

    if causal:
        latency_ms = model.receptive_field() / config["data"]["sample_rate"] * 1000
        print(f"    Latency: {latency_ms:.2f}ms @ {config['data']['sample_rate']}Hz")

    return model, model_path


def process_audio_file(
    model, input_file, output_file, device="cpu", chunk_size=None, streaming=False
):
    """Process an audio file through the model

    Args:
        model: WaveNetAmp model
        input_file: Path to input audio file
        output_file: Path to output audio file
        device: torch device ('cuda' or 'cpu')
        chunk_size: Size of chunks for processing (optional)
        streaming: If True, use streaming mode with buffer management (requires causal model)
    """
    print(f"\n[*] Processing: {input_file}")

    # Load audio
    audio, sr = sf.read(input_file)

    # Handle stereo files - convert to mono
    if audio.ndim == 2:
        print(
            f"    [!] Stereo file detected, converting to mono (averaging channels)"
        )
        audio = np.mean(audio, axis=1)

    print(f"    Sample rate: {sr} Hz")
    print(f"    Duration: {len(audio)/sr:.2f} seconds")
    print(f"    Samples: {len(audio):,}")

    # Convert to tensor and add batch + channel dimensions
    # Shape: [samples] -> [1, 1, samples]
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    audio_tensor = audio_tensor.to(device)

    # Process through model
    print(
        f"    Processing on {device}... (mode: {'streaming' if streaming else 'batch'})"
    )
    model.eval()

    with torch.no_grad():
        if streaming:
            # Use streaming mode with buffer management
            if not model.causal:
                raise ValueError(
                    "Streaming mode requires a causal model. "
                    "Set causal=true in config and retrain."
                )

            # Initialize streaming processor
            streamer = StreamingWaveNetAmp(model, device=device)

            # Default chunk size for streaming (512 samples ~= 11.6ms @ 44.1kHz)
            if chunk_size is None:
                chunk_size = 512

            print(f"    Chunk size: {chunk_size} samples (~{chunk_size/sr*1000:.1f}ms)")
            print(f"    Latency: {streamer.get_latency_ms(sr):.2f}ms")

            # Process in chunks with buffer management
            output_chunks = []
            num_chunks = int(np.ceil(audio_tensor.shape[-1] / chunk_size))

            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, audio_tensor.shape[-1])
                chunk = audio_tensor[:, :, start:end]
                output_chunk = streamer.process_chunk(chunk)
                output_chunks.append(output_chunk)

            output = torch.cat(output_chunks, dim=-1)

        elif chunk_size is not None:
            # Process in chunks to avoid memory issues with very long files
            # (but without streaming buffer management)
            output_chunks = []
            num_chunks = int(np.ceil(audio_tensor.shape[-1] / chunk_size))

            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, audio_tensor.shape[-1])
                chunk = audio_tensor[:, :, start:end]
                output_chunk = model(chunk)
                output_chunks.append(output_chunk)

            output = torch.cat(output_chunks, dim=-1)
        else:
            # Process entire file at once
            output = model(audio_tensor)

    # Convert back to numpy
    output_audio = output.squeeze().cpu().numpy()

    # Normalize to prevent clipping
    max_val = np.abs(output_audio).max()
    if max_val > 1.0:
        output_audio = output_audio / max_val
        print(f"    [!] Normalized output (peak was {max_val:.2f})")

    # Save output
    sf.write(output_file, output_audio, sr)
    print(f"[*] Saved: {output_file}")

    return output_audio, sr


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained amp model"
    )
    parser.add_argument("input", type=str, help="Input audio file (clean DI)")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file (default: input_processed.wav)",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None, help="Model name (default: from config)"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=None,
        help="Process in chunks of N samples (for large files or streaming)",
    )
    parser.add_argument(
        "-s",
        "--streaming",
        action="store_true",
        help="Use streaming mode with buffer management (requires causal model)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load model
    model, model_path = load_trained_model(config, args.model, device)

    # Determine output filename
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(
            input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
        )

    # Process audio
    process_audio_file(
        model, args.input, args.output, device, args.chunk_size, args.streaming
    )

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
