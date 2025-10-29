"""
Real-time audio processing demo using PyAudio

This script demonstrates real-time processing of audio input through a causal model.
Requires: pip install pyaudio

WARNING: This is a proof-of-concept demo. For production use, build a proper VST plugin.
"""

import torch
import numpy as np
import argparse
import yaml
from model import WaveNetAmp, StreamingWaveNetAmp

try:
    import pyaudio
except ImportError:
    print("ERROR: PyAudio not installed")
    print("Install with: pip install pyaudio")
    print(
        "Note: On Windows, you may need: pip install pipwin && pipwin install pyaudio"
    )
    exit(1)


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def audio_callback(in_data, frame_count, time_info, status):
    """PyAudio callback for real-time processing"""
    global streamer, device, input_channel_select, input_channels, output_channels

    # Convert input bytes to numpy array
    audio_input = np.frombuffer(in_data, dtype=np.float32)

    # Reshape to (samples, channels) based on stream configuration
    stream_channels = max(input_channels, output_channels)
    audio_input = audio_input.reshape(-1, stream_channels)

    # Select the specified input channel
    audio_input = audio_input[:, input_channel_select]

    # Convert to tensor: [samples] -> [1, 1, samples]
    audio_tensor = torch.from_numpy(audio_input).float().unsqueeze(0).unsqueeze(0)

    # Process through model
    with torch.no_grad():
        output = streamer.process_chunk(audio_tensor.to(device))

    # Convert back to numpy (mono output)
    audio_output = output.squeeze().cpu().numpy()

    # Duplicate mono signal to match output channels
    if output_channels > 1:
        # Duplicate mono to all output channels
        audio_output = np.tile(audio_output[:, np.newaxis], (1, output_channels))
    else:
        # Keep as mono, but reshape for consistency
        audio_output = audio_output[:, np.newaxis]

    # If stream needs more channels than output, pad with zeros
    if stream_channels > output_channels:
        padding = np.zeros((audio_output.shape[0], stream_channels - output_channels), dtype=np.float32)
        audio_output = np.concatenate([audio_output, padding], axis=1)

    # Flatten and convert to bytes
    audio_output = audio_output.flatten()
    return (audio_output.astype(np.float32).tobytes(), pyaudio.paContinue)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio processing demo (PROOF OF CONCEPT)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained causal model (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_realtime.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Audio buffer size (default: 512)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--input-device", type=int, default=None, help="Input device index"
    )
    parser.add_argument(
        "--output-device", type=int, default=None, help="Output device index"
    )
    parser.add_argument(
        "--input-channels",
        type=int,
        default=1,
        help="Number of input channels (1=mono, 2=stereo, etc.)",
    )
    parser.add_argument(
        "--input-channel",
        type=int,
        default=0,
        help="Which input channel to use (0=left, 1=right for stereo input)",
    )
    parser.add_argument(
        "--output-channels",
        type=int,
        default=2,
        help="Number of output channels (1=mono, 2=stereo headphones)",
    )

    args = parser.parse_args()

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # List devices if requested
    if args.list_devices:
        print("\nAvailable Audio Devices:")
        print("=" * 60)
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"{i}: {info['name']}")
            print(f"   Max Input Channels: {info['maxInputChannels']}")
            print(f"   Max Output Channels: {info['maxOutputChannels']}")
            print(f"   Default Sample Rate: {info['defaultSampleRate']}")
            print()
        p.terminate()
        return

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Check if model is causal
    if not config["model"].get("causal", False):
        print("ERROR: Model must be causal for real-time processing")
        print("Set causal: true in config and retrain the model")
        p.terminate()
        return

    # Set device and channel globals for callback
    global device, input_channel_select, input_channels, output_channels
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_channels = args.input_channels
    input_channel_select = args.input_channel
    output_channels = args.output_channels

    print(f"Using device: {device}")

    # Validate input channel selection
    if input_channel_select >= input_channels:
        print(f"ERROR: input-channel ({input_channel_select}) must be less than input-channels ({input_channels})")
        p.terminate()
        return

    # Load model
    print(f"Loading model from: {args.model}")
    model = WaveNetAmp(
        channels=config["model"]["channels"],
        num_layers=config["model"]["num_layers"],
        kernel_size=config["model"]["kernel_size"],
        dilation_base=config["model"]["dilation_base"],
        causal=True,
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Receptive field: {model.receptive_field()} samples")

    # Initialize streaming processor
    global streamer
    streamer = StreamingWaveNetAmp(model, device=device)

    sample_rate = config["data"]["sample_rate"]
    latency_ms = streamer.get_latency_ms(sample_rate)

    print(f"\nReal-time Configuration:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Input channels: {input_channels}")
    print(f"  Selected input channel: {input_channel_select} ({'Left' if input_channel_select == 0 else 'Right' if input_channel_select == 1 else str(input_channel_select)})")
    print(f"  Output channels: {output_channels} ({'Mono' if output_channels == 1 else 'Stereo' if output_channels == 2 else str(output_channels) + ' channels'})")
    print(f"  Output mode: Mono signal {'duplicated to stereo' if output_channels == 2 else 'to all channels' if output_channels > 2 else ''}")
    print(f"  Buffer size: {args.chunk_size} samples")
    print(f"  Buffer duration: {args.chunk_size / sample_rate * 1000:.2f}ms")
    print(f"  Model latency: {latency_ms:.2f}ms")
    print(f"  Total latency: ~{latency_ms + args.chunk_size / sample_rate * 1000:.2f}ms")

    # Open audio stream
    print("\n" + "=" * 60)
    print("Starting real-time audio processing...")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        # Use max of input/output channels for the stream
        stream_channels = max(input_channels, output_channels)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=stream_channels,
            rate=sample_rate,
            input=True,
            output=True,
            input_device_index=args.input_device,
            output_device_index=args.output_device,
            frames_per_buffer=args.chunk_size,
            stream_callback=audio_callback,
        )

        stream.start_stream()

        # Keep stream running
        while stream.is_active():
            import time

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        if "stream" in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

    print("Done!")


if __name__ == "__main__":
    main()
