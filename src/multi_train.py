"""
Multi-model training orchestration for Neural Architecture Search

Trains multiple model configurations, filters by performance, and evaluates
the Pareto frontier of quality/latency/efficiency trade-offs.
"""

import os
import sys
import time
import yaml
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

from model import WaveNetAmp
from dataset import AmpDataset
from search_space import generate_search_space, save_configs, print_search_space_summary
from benchmark import benchmark_inference, count_parameters
from export_plugin_weights import export_for_plugin

# Import loss functions from train.py
from train import esr_loss, spectral_loss, hybrid_loss, get_loss_function


def train_single_model(config_path: str, phase: str = 'phase2',
                       device: Optional[torch.device] = None) -> Dict:
    """Train a single model configuration

    Args:
        config_path: Path to model configuration YAML file
        phase: Training phase ('phase1' for quick scan, 'phase2' for full training)
        device: PyTorch device (defaults to CUDA if available)

    Returns:
        Dictionary with training results and metrics
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    print(f"\n{'='*70}")
    print(f"Training: {model_name} (Phase: {phase})")
    print(f"{'='*70}")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup directories
    model_dir = Path(config['paths']['model_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    tensorboard_dir = Path(config['paths']['tensorboard_dir'])

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = AmpDataset(
        input_file=config['data']['input_file'],
        output_file=config['data']['output_file'],
        segment_length=config['data']['segment_length'],
        sample_rate=config['data']['sample_rate']
    )

    # Split into train/val
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Consistent splits
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues on Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    model = WaveNetAmp(
        channels=config['model']['channels'],
        num_layers=config['model']['num_layers'],
        kernel_size=config['model']['kernel_size'],
        dilation_base=config['model']['dilation_base'],
        causal=config['model']['causal']
    ).to(device)

    params = count_parameters(model)
    receptive_field = model.receptive_field()
    latency_ms = (receptive_field / config['data']['sample_rate']) * 1000

    print(f"Architecture:")
    print(f"  Channels: {config['model']['channels']}")
    print(f"  Layers: {config['model']['num_layers']}")
    print(f"  Kernel: {config['model']['kernel_size']}")
    print(f"  Causal: {config['model']['causal']}")
    print(f"  Parameters: {params:,}")
    print(f"  Receptive field: {receptive_field} samples")
    print(f"  Latency: {latency_ms:.2f}ms\n")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=15,
        min_lr=1e-6
    )

    # Get loss function from config
    loss_fn = get_loss_function(config)

    # TensorBoard writer
    writer = SummaryWriter(str(tensorboard_dir))

    # Training loop
    num_epochs = config['training']['num_epochs']
    patience = config['training']['patience']
    checkpoint_every = config['training']['checkpoint_every']

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    best_epoch = 0

    print(f"Starting training for up to {num_epochs} epochs (patience={patience})...")

    for epoch in range(num_epochs):
        # Train epoch
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for input_audio, target_audio in train_pbar:
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)

            # Forward pass
            output = model(input_audio)
            loss = loss_fn(output, target_audio)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            for input_audio, target_audio in val_pbar:
                input_audio = input_audio.to(device)
                target_audio = target_audio.to(device)

                output = model(input_audio)
                loss = loss_fn(output, target_audio)
                total_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        val_loss = total_val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"train={train_loss:.6f}, val={val_loss:.6f}, lr={current_lr:.2e}")

        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': config['model']
            }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch + 1

            # Save best model with full metadata
            best_model_path = model_dir / "best_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config['model'],
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch + 1,
                'receptive_field': receptive_field,
                'latency_ms': latency_ms,
                'parameters': params
            }, best_model_path)

            print(f"  [OK] New best model! (val_loss: {val_loss:.6f})")

            # Auto-export for VST plugin (only if causal)
            if config['model']['causal'] and phase == 'phase2':
                json_path = model_dir / f"{model_name}.json"
                try:
                    export_for_plugin(str(best_model_path), config_path, str(json_path))
                    print(f"  [OK] VST export: {json_path}")
                except Exception as e:
                    print(f"  [X] Export failed: {e}")

            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            break

    writer.close()
    training_time = time.time() - start_time

    # Create results dictionary
    results = {
        'model_name': model_name,
        'config_id': config.get('metadata', {}).get('config_id', 0),
        'phase': phase,
        'architecture': config['model'],
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'training_time_sec': training_time,
        'parameters': params,
        'receptive_field': receptive_field,
        'latency_ms': latency_ms,
        'best_model_path': str(best_model_path),
        'config_path': config_path
    }

    print(f"\n[OK] Training complete!")
    print(f"  Best val_loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Training time: {training_time/60:.1f} minutes")

    return results


def run_multi_train(search_config_path: str, phase: str = 'both',
                   configs_dir: Optional[str] = None):
    """Run multi-model training

    Args:
        search_config_path: Path to search configuration YAML
        phase: 'phase1', 'phase2', or 'both'
        configs_dir: Optional directory with pre-generated configs (for phase2 after filtering)
    """
    print(f"\n{'='*70}")
    print("NEURAL ARCHITECTURE SEARCH - MULTI-MODEL TRAINING")
    print(f"{'='*70}\n")

    # Load search config
    print(f"Loading search configuration: {search_config_path}")
    with open(search_config_path, 'r') as f:
        search_config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Create results directory
    results_dir = Path(search_config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Phase 1: Quick Scan
    if phase in ['phase1', 'both']:
        print(f"\n{'='*70}")
        print("PHASE 1: QUICK SCAN")
        print(f"{'='*70}\n")

        # Generate search space if not provided
        if configs_dir is None:
            print("Generating search space...")
            model_configs = generate_search_space(search_config)
            print_search_space_summary(model_configs, search_config['data']['sample_rate'])

            # Save configs
            config_output_dir = search_config['paths']['config_dir']
            print(f"Saving configurations to {config_output_dir}/...")
            config_paths = save_configs(model_configs, search_config,
                                       config_output_dir, phase='phase1')
            print(f"[OK] Saved {len(config_paths)} configuration files\n")
        else:
            # Load existing configs
            config_paths = list(Path(configs_dir).glob("*.yaml"))
            print(f"Loading {len(config_paths)} existing configs from {configs_dir}\n")

        # Train all models (Phase 1)
        phase1_results = []
        for i, config_path in enumerate(config_paths, 1):
            print(f"\n[{i}/{len(config_paths)}] Training {Path(config_path).stem}...")
            try:
                result = train_single_model(str(config_path), phase='phase1', device=device)
                phase1_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"[X] Training failed: {e}")
                continue

        # Save Phase 1 results
        phase1_csv = results_dir / "phase1_results.csv"
        save_results_csv(phase1_results, str(phase1_csv))
        print(f"\n[OK] Phase 1 results saved to {phase1_csv}")

        # Filter configurations
        if phase == 'both':
            keep_percent = search_config['phase1']['keep_top_percent']
            filtered_configs = filter_top_models(phase1_results, keep_percent)

            # Save filtered configs for Phase 2
            filtered_dir = Path(search_config['paths']['config_dir_filtered'])
            filtered_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nFiltering: Keeping top {keep_percent}% by validation loss...")
            print(f"Selected {len(filtered_configs)}/{len(phase1_results)} models for Phase 2\n")

            # Copy/regenerate configs for Phase 2
            filtered_paths = []
            for result in filtered_configs:
                # Load original config and update for Phase 2
                with open(result['config_path'], 'r') as f:
                    config = yaml.safe_load(f)

                # Update training settings for Phase 2
                config['training'] = {
                    'batch_size': search_config['phase2']['batch_size'],
                    'num_epochs': search_config['phase2']['num_epochs'],
                    'learning_rate': search_config['phase2']['learning_rate'],
                    'weight_decay': search_config['phase2']['weight_decay'],
                    'patience': search_config['phase2']['patience'],
                    'checkpoint_every': search_config['phase2']['checkpoint_every']
                }
                config['metadata']['training_phase'] = 'phase2'

                # Save to filtered directory
                filtered_path = filtered_dir / Path(result['config_path']).name
                with open(filtered_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                filtered_paths.append(str(filtered_path))

            configs_dir = str(filtered_dir)

    # Phase 2: Full Training
    if phase in ['phase2', 'both']:
        print(f"\n{'='*70}")
        print("PHASE 2: FULL TRAINING")
        print(f"{'='*70}\n")

        # Get config paths
        if configs_dir is None:
            raise ValueError("Must specify configs_dir for Phase 2 training")

        config_paths = list(Path(configs_dir).glob("*.yaml"))
        print(f"Training {len(config_paths)} models to full convergence...\n")

        # Train all models (Phase 2)
        phase2_results = []
        for i, config_path in enumerate(config_paths, 1):
            print(f"\n[{i}/{len(config_paths)}] Training {Path(config_path).stem}...")
            try:
                result = train_single_model(str(config_path), phase='phase2', device=device)
                phase2_results.append(result)
                all_results.append(result)

                # Benchmark CPU performance (after training)
                print(f"\nBenchmarking CPU performance...")
                # Extract only WaveNetAmp parameters (exclude 'name')
                arch_params = {k: v for k, v in result['architecture'].items()
                              if k in ['channels', 'num_layers', 'kernel_size', 'dilation_base', 'causal']}
                bench_result = benchmark_inference(
                    model=WaveNetAmp(**arch_params).cpu(),
                    chunk_size=search_config['benchmark']['chunk_size'],
                    num_chunks=search_config['benchmark']['num_chunks'],
                    device='cpu'
                )
                # Add benchmark metrics to result
                result.update({
                    'cpu_mean_ms': bench_result['mean_ms'],
                    'cpu_rtf': bench_result['rtf'],
                    'cpu_realtime_capable': bench_result['can_realtime']
                })

            except Exception as e:
                print(f"[X] Training failed: {e}")
                continue

        # Save Phase 2 results
        phase2_csv = results_dir / "phase2_results.csv"
        save_results_csv(phase2_results, str(phase2_csv))
        print(f"\n[OK] Phase 2 results saved to {phase2_csv}")

    # Save all results
    all_csv = results_dir / "all_results.csv"
    save_results_csv(all_results, str(all_csv))

    print(f"\n{'='*70}")
    print("MULTI-TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total models trained: {len(all_results)}")
    print(f"Results saved to: {results_dir}")
    print(f"\nNext step: Run analyze_results.py to find Pareto frontier")
    print(f"{'='*70}\n")


def filter_top_models(results: List[Dict], keep_percent: float) -> List[Dict]:
    """Filter top models by validation loss

    Args:
        results: List of training results
        keep_percent: Percentage of models to keep (0-100)

    Returns:
        Filtered list of top models
    """
    # Sort by validation loss (lower is better)
    sorted_results = sorted(results, key=lambda x: x['best_val_loss'])

    # Keep top N%
    num_keep = max(1, int(len(sorted_results) * keep_percent / 100))
    filtered = sorted_results[:num_keep]

    return filtered


def save_results_csv(results: List[Dict], csv_path: str):
    """Save results to CSV file

    Args:
        results: List of training results dictionaries
        csv_path: Path to save CSV file
    """
    if not results:
        print("No results to save")
        return

    # Define columns
    columns = [
        'model_name', 'config_id', 'phase',
        'num_layers', 'channels', 'kernel_size', 'dilation_base', 'causal',
        'parameters', 'receptive_field', 'latency_ms',
        'best_train_loss', 'best_val_loss', 'best_epoch', 'total_epochs',
        'training_time_sec',
        'cpu_mean_ms', 'cpu_rtf', 'cpu_realtime_capable',
        'best_model_path', 'config_path'
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for result in results:
            # Flatten architecture dict
            row = {
                **result,
                'num_layers': result['architecture']['num_layers'],
                'channels': result['architecture']['channels'],
                'kernel_size': result['architecture']['kernel_size'],
                'dilation_base': result['architecture']['dilation_base'],
                'causal': result['architecture']['causal'],
            }
            # Add missing columns with default values
            for col in columns:
                if col not in row:
                    row[col] = ''
            writer.writerow(row)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-model Neural Architecture Search training')
    parser.add_argument('--search-config', type=str,
                       default='configs/search_config.yaml',
                       help='Path to search configuration file')
    parser.add_argument('--phase', type=str, default='both',
                       choices=['phase1', 'phase2', 'both'],
                       help='Training phase to run')
    parser.add_argument('--configs-dir', type=str, default=None,
                       help='Directory with pre-generated configs (for phase2 only)')

    args = parser.parse_args()

    run_multi_train(
        search_config_path=args.search_config,
        phase=args.phase,
        configs_dir=args.configs_dir
    )


if __name__ == "__main__":
    main()
