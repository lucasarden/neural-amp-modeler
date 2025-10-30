"""
Search space generation for Neural Architecture Search

Generates all model configurations in the search space, calculates latency,
estimates CPU cost, applies constraints, and saves individual config files.
"""

import os
import yaml
import itertools
from pathlib import Path
from typing import Dict, List, Tuple


def calculate_receptive_field(num_layers: int, kernel_size: int, dilation_base: int) -> int:
    """Calculate receptive field without building the model

    This matches the implementation in model.py WaveNetAmp.receptive_field()

    Args:
        num_layers: Number of residual blocks
        kernel_size: Convolution kernel size
        dilation_base: Base for exponential dilation pattern (2^i)

    Returns:
        Receptive field in samples
    """
    rf = 1
    for i in range(num_layers):
        dilation = dilation_base ** i
        rf += (kernel_size - 1) * dilation
    return rf


def calculate_latency_ms(num_layers: int, kernel_size: int, dilation_base: int,
                         sample_rate: int = 44100) -> float:
    """Calculate latency in milliseconds

    Args:
        num_layers: Number of residual blocks
        kernel_size: Convolution kernel size
        dilation_base: Base for exponential dilation pattern
        sample_rate: Audio sample rate in Hz

    Returns:
        Latency in milliseconds
    """
    rf = calculate_receptive_field(num_layers, kernel_size, dilation_base)
    latency_ms = (rf / sample_rate) * 1000
    return latency_ms


def estimate_parameter_count(channels: int, num_layers: int, kernel_size: int) -> int:
    """Estimate model parameter count without building the model

    Architecture:
        - Input conv: 1 -> channels (kernel=1)
        - Residual blocks: channels -> channels (kernel=kernel_size) × num_layers
        - Output conv: channels -> 1 (kernel=1)

    Args:
        channels: Number of channels in residual blocks
        num_layers: Number of residual blocks
        kernel_size: Convolution kernel size

    Returns:
        Estimated total parameters (including biases)
    """
    # Input conv1d: (in_channels * out_channels * kernel) + bias
    # 1 * channels * 1 + channels
    input_params = channels + channels

    # Each residual block: Conv1d layer
    # (channels * channels * kernel_size) + channels
    block_params = (channels * channels * kernel_size) + channels
    residual_params = num_layers * block_params

    # Output conv1d: (channels * 1 * 1) + 1
    output_params = channels + 1

    total = input_params + residual_params + output_params
    return total


def generate_config_name(config_id: int, arch_params: Dict) -> str:
    """Generate a descriptive model name from architecture parameters

    Format: search_{id:03d}_L{layers}_C{channels}_K{kernel}
    Example: search_001_L6_C16_K3

    Args:
        config_id: Configuration ID number
        arch_params: Dictionary with num_layers, channels, kernel_size

    Returns:
        Model name string
    """
    return (f"search_{config_id:03d}_"
            f"L{arch_params['num_layers']}_"
            f"C{arch_params['channels']}_"
            f"K{arch_params['kernel_size']}")


def generate_search_space(search_config: Dict) -> List[Dict]:
    """Generate all configurations in the search space

    Creates all combinations of hyperparameters, calculates metrics,
    and applies constraints.

    Args:
        search_config: Loaded search configuration dictionary

    Returns:
        List of valid configuration dictionaries with metadata
    """
    space = search_config['search_space']
    constraints = search_config['constraints']
    sample_rate = search_config['data']['sample_rate']

    # Generate all combinations
    all_configs = []
    config_id = 1

    combinations = itertools.product(
        space['num_layers'],
        space['channels'],
        space['kernel_size']
    )

    for num_layers, channels, kernel_size in combinations:
        # Calculate metrics
        rf = calculate_receptive_field(num_layers, kernel_size, space['dilation_base'])
        latency_ms = calculate_latency_ms(num_layers, kernel_size,
                                          space['dilation_base'], sample_rate)
        params = estimate_parameter_count(channels, num_layers, kernel_size)

        # Create config dict
        arch_params = {
            'num_layers': num_layers,
            'channels': channels,
            'kernel_size': kernel_size,
            'dilation_base': space['dilation_base'],
            'causal': space['causal']
        }

        config = {
            'id': config_id,
            'name': generate_config_name(config_id, arch_params),
            'architecture': arch_params,
            'metrics': {
                'receptive_field': rf,
                'latency_ms': latency_ms,
                'estimated_params': params
            }
        }

        # Apply constraints
        valid = True
        rejection_reasons = []

        if latency_ms > constraints['max_latency_ms']:
            valid = False
            rejection_reasons.append(
                f"latency {latency_ms:.2f}ms > {constraints['max_latency_ms']}ms"
            )

        if params > constraints['max_params']:
            valid = False
            rejection_reasons.append(
                f"params {params} > {constraints['max_params']}"
            )

        if params < constraints['min_params']:
            valid = False
            rejection_reasons.append(
                f"params {params} < {constraints['min_params']}"
            )

        config['valid'] = valid
        config['rejection_reasons'] = rejection_reasons

        if valid:
            all_configs.append(config)
        else:
            print(f"  [X] {config['name']}: {', '.join(rejection_reasons)}")

        config_id += 1

    return all_configs


def create_model_config(base_config: Dict, model_config: Dict, phase: str = 'phase2') -> Dict:
    """Create a complete training config for a specific model

    Args:
        base_config: Base search configuration
        model_config: Model-specific configuration from generate_search_space()
        phase: Training phase ('phase1' for quick scan, 'phase2' for full training)

    Returns:
        Complete configuration dictionary ready to save as YAML
    """
    # Start with base data settings
    config = {
        'data': base_config['data'].copy(),
        'model': {
            'name': model_config['name'],
            **model_config['architecture']
        },
        'loss': base_config['loss'].copy(),
    }

    # Add phase-specific training settings
    phase_config = base_config[phase]
    config['training'] = {
        'batch_size': phase_config['batch_size'],
        'num_epochs': phase_config['num_epochs'],
        'learning_rate': phase_config['learning_rate'],
        'weight_decay': phase_config['weight_decay'],
        'patience': phase_config['patience'],
        'checkpoint_every': phase_config['checkpoint_every']
    }

    # Add paths (will be formatted with model name)
    config['paths'] = {
        'model_dir': f"models/{model_config['name']}",
        'checkpoint_dir': f"models/{model_config['name']}/checkpoints",
        'best_model': f"models/{model_config['name']}/best_model.pt",
        'tensorboard_dir': f"runs/search/{model_config['name']}"
    }

    # Add metadata for tracking
    config['metadata'] = {
        'config_id': model_config['id'],
        'receptive_field': model_config['metrics']['receptive_field'],
        'latency_ms': model_config['metrics']['latency_ms'],
        'estimated_params': model_config['metrics']['estimated_params'],
        'training_phase': phase
    }

    return config


def save_configs(configs: List[Dict], base_config: Dict, output_dir: str,
                 phase: str = 'phase2') -> List[str]:
    """Save individual YAML config files for each model

    Args:
        configs: List of model configurations
        base_config: Base search configuration
        output_dir: Directory to save config files
        phase: Training phase ('phase1' or 'phase2')

    Returns:
        List of saved config file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for model_config in configs:
        # Create complete training config
        full_config = create_model_config(base_config, model_config, phase)

        # Save as YAML
        filename = f"{model_config['name']}.yaml"
        filepath = output_path / filename

        with open(filepath, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        saved_paths.append(str(filepath))

    return saved_paths


def print_search_space_summary(configs: List[Dict], sample_rate: int = 44100):
    """Print a summary of the search space

    Args:
        configs: List of valid model configurations
        sample_rate: Audio sample rate for latency calculations
    """
    print(f"\n{'='*70}")
    print("SEARCH SPACE SUMMARY")
    print(f"{'='*70}")
    print(f"Total valid configurations: {len(configs)}\n")

    # Extract metrics
    latencies = [c['metrics']['latency_ms'] for c in configs]
    params = [c['metrics']['estimated_params'] for c in configs]
    layers = [c['architecture']['num_layers'] for c in configs]
    channels = [c['architecture']['channels'] for c in configs]

    print("Latency Range:")
    print(f"  Min: {min(latencies):.2f}ms ({min(latencies) * sample_rate / 1000:.0f} samples)")
    print(f"  Max: {max(latencies):.2f}ms ({max(latencies) * sample_rate / 1000:.0f} samples)")
    print(f"  Mean: {sum(latencies)/len(latencies):.2f}ms")

    print("\nParameter Count Range:")
    print(f"  Min: {min(params):,} parameters")
    print(f"  Max: {max(params):,} parameters")
    print(f"  Mean: {int(sum(params)/len(params)):,} parameters")

    print("\nArchitecture Ranges:")
    print(f"  Layers: {min(layers)} - {max(layers)}")
    print(f"  Channels: {min(channels)} - {max(channels)}")

    print(f"\n{'='*70}")
    print("CONFIGURATION DETAILS")
    print(f"{'='*70}")
    print(f"{'ID':<4} {'Name':<25} {'Latency':<12} {'Params':<10} {'Layers':<7} {'Channels'}")
    print(f"{'-'*70}")

    for config in sorted(configs, key=lambda x: x['metrics']['latency_ms']):
        print(f"{config['id']:<4} "
              f"{config['name']:<25} "
              f"{config['metrics']['latency_ms']:>6.2f}ms    "
              f"{config['metrics']['estimated_params']:>8,}  "
              f"{config['architecture']['num_layers']:>7}  "
              f"{config['architecture']['channels']:>7}")

    print(f"{'='*70}\n")


def main():
    """Main function for testing search space generation"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Neural Architecture Search space')
    parser.add_argument('--config', type=str,
                       default='configs/search_config.yaml',
                       help='Path to search configuration file')
    parser.add_argument('--output-dir', type=str,
                       default='configs/search',
                       help='Directory to save generated configs')
    parser.add_argument('--phase', type=str, default='phase2',
                       choices=['phase1', 'phase2'],
                       help='Training phase (phase1=quick scan, phase2=full training)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only print summary, do not save configs')

    args = parser.parse_args()

    # Load search config
    print(f"Loading search configuration from {args.config}...")
    with open(args.config, 'r') as f:
        search_config = yaml.safe_load(f)

    # Generate search space
    print("Generating search space...")
    configs = generate_search_space(search_config)

    # Print summary
    print_search_space_summary(configs, search_config['data']['sample_rate'])

    if not args.dry_run:
        # Save configs
        print(f"Saving configuration files to {args.output_dir}/...")
        saved_paths = save_configs(configs, search_config, args.output_dir, args.phase)
        print(f"[OK] Saved {len(saved_paths)} configuration files")
    else:
        print("Dry run - no files saved")


if __name__ == "__main__":
    main()
