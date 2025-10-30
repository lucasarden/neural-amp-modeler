"""
CPU performance benchmarking for trained models

Measures inference time, real-time factor, and resource usage to evaluate
whether models can run in real-time on CPU.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from model import WaveNetAmp


def count_parameters(model: torch.nn.Module) -> int:
    """Count actual number of trainable parameters

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculate model size in megabytes

    Args:
        model: PyTorch model

    Returns:
        Model size in MB (assuming float32)
    """
    param_count = count_parameters(model)
    # 4 bytes per float32 parameter
    size_bytes = param_count * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def benchmark_inference(model: torch.nn.Module,
                       chunk_size: int = 512,
                       num_chunks: int = 100,
                       warmup_chunks: int = 10,
                       sample_rate: int = 44100,
                       device: str = 'cpu') -> Dict:
    """Benchmark model inference performance

    Measures how fast the model can process audio on CPU, which determines
    whether it can run in real-time.

    Args:
        model: Trained WaveNetAmp model
        chunk_size: Number of samples per chunk (typical real-time buffer size)
        num_chunks: Number of chunks to process for benchmarking
        warmup_chunks: Number of warmup iterations before timing
        sample_rate: Audio sample rate in Hz
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Dictionary with benchmark results:
            - mean_ms: Mean processing time per chunk (milliseconds)
            - std_ms: Standard deviation of processing time
            - min_ms: Minimum processing time
            - max_ms: Maximum processing time
            - rtf: Real-time factor (processing_time / audio_duration)
            - can_realtime: Boolean, True if RTF < 0.8 (20% safety margin)
            - throughput_samples_per_sec: Samples processed per second
            - chunk_size: Chunk size used
            - sample_rate: Sample rate used
    """
    # Move model to device and set to eval mode
    model.to(device)
    model.eval()

    # Generate random audio chunks for testing
    # Use randn to simulate realistic audio signal distribution
    dummy_chunk = torch.randn(1, 1, chunk_size, device=device)

    # Warmup: get model loaded into cache, JIT compiled, etc.
    print(f"Warming up ({warmup_chunks} chunks)...")
    with torch.no_grad():
        for _ in range(warmup_chunks):
            _ = model(dummy_chunk)

    # Synchronize if using CUDA to ensure warmup is complete
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark: measure processing time for many chunks
    print(f"Benchmarking ({num_chunks} chunks of {chunk_size} samples)...")
    times_ms = []

    with torch.no_grad():
        for i in range(num_chunks):
            # Generate new random chunk each iteration
            chunk = torch.randn(1, 1, chunk_size, device=device)

            # Time the forward pass
            start = time.perf_counter()
            _ = model(chunk)

            # Synchronize to ensure computation is complete
            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()

            # Convert to milliseconds
            elapsed_ms = (end - start) * 1000
            times_ms.append(elapsed_ms)

            # Progress indicator every 20 chunks
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{num_chunks} chunks...")

    # Calculate statistics
    times_array = np.array(times_ms)
    mean_ms = np.mean(times_array)
    std_ms = np.std(times_array)
    min_ms = np.min(times_array)
    max_ms = np.max(times_array)

    # Calculate real-time factor
    # RTF = processing_time / audio_duration
    # RTF < 1.0 means can process faster than real-time
    chunk_duration_ms = (chunk_size / sample_rate) * 1000
    rtf = mean_ms / chunk_duration_ms

    # Calculate throughput (samples per second)
    mean_sec = mean_ms / 1000
    throughput = chunk_size / mean_sec if mean_sec > 0 else 0

    # Safety margin: consider real-time capable if RTF < 0.8
    # This allows for headroom for other processing, OS overhead, etc.
    can_realtime = rtf < 0.8

    results = {
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'rtf': rtf,
        'can_realtime': can_realtime,
        'throughput_samples_per_sec': throughput,
        'chunk_size': chunk_size,
        'sample_rate': sample_rate,
        'device': device,
        'chunk_duration_ms': chunk_duration_ms
    }

    return results


def benchmark_model_file(model_path: str,
                         chunk_size: int = 512,
                         num_chunks: int = 100,
                         device: str = 'cpu') -> Dict:
    """Benchmark a saved model file

    Loads model from checkpoint and runs full benchmark suite.

    Args:
        model_path: Path to saved model (.pt file)
        chunk_size: Chunk size for inference benchmark
        num_chunks: Number of chunks to benchmark
        device: Device to run on

    Returns:
        Dictionary with all benchmark results including model info
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        raise ValueError(f"Checkpoint missing 'model_config'. Cannot reconstruct model.")

    # Create model
    model = WaveNetAmp(
        channels=model_config['channels'],
        num_layers=model_config['num_layers'],
        kernel_size=model_config['kernel_size'],
        dilation_base=model_config['dilation_base'],
        causal=model_config['causal']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Count parameters and measure size
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    receptive_field = model.receptive_field()
    latency_ms = (receptive_field / 44100) * 1000

    print(f"Model Architecture:")
    print(f"  Channels: {model_config['channels']}")
    print(f"  Layers: {model_config['num_layers']}")
    print(f"  Kernel size: {model_config['kernel_size']}")
    print(f"  Causal: {model_config['causal']}")
    print(f"  Parameters: {params:,}")
    print(f"  Model size: {size_mb:.2f} MB")
    print(f"  Receptive field: {receptive_field} samples")
    print(f"  Latency: {latency_ms:.2f} ms @ 44.1kHz\n")

    # Run inference benchmark
    inference_results = benchmark_inference(
        model=model,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        device=device
    )

    # Combine all results
    results = {
        'model_path': model_path,
        'model_name': Path(model_path).parent.name,
        'architecture': model_config,
        'parameters': params,
        'size_mb': size_mb,
        'receptive_field': receptive_field,
        'latency_ms': latency_ms,
        **inference_results
    }

    # Print results
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Inference Performance (chunk_size={chunk_size}, device={device}):")
    print(f"  Mean time: {inference_results['mean_ms']:.3f} ± {inference_results['std_ms']:.3f} ms")
    print(f"  Min time: {inference_results['min_ms']:.3f} ms")
    print(f"  Max time: {inference_results['max_ms']:.3f} ms")
    print(f"  Chunk duration: {inference_results['chunk_duration_ms']:.3f} ms")
    print(f"  Real-time factor: {inference_results['rtf']:.3f}x")
    print(f"  Throughput: {inference_results['throughput_samples_per_sec']:,.0f} samples/sec")

    if inference_results['can_realtime']:
        print(f"  [OK] Real-time capable on {device.upper()} (RTF < 0.8)")
    else:
        print(f"  [X] NOT real-time capable on {device.upper()} (RTF >= 0.8)")

    print(f"{'='*70}\n")

    return results


def main():
    """Main function for benchmarking from command line"""
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark model CPU performance')
    parser.add_argument('model_path', type=str,
                       help='Path to saved model (.pt file)')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Chunk size for inference (default: 512)')
    parser.add_argument('--num-chunks', type=int, default=100,
                       help='Number of chunks to benchmark (default: 100)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to benchmark on (default: cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_model_file(
        model_path=args.model_path,
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
        device=args.device
    )

    # Save to JSON if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
