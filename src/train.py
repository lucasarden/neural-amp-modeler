"""
Training script for amp modeler
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
import os
from tqdm import tqdm

from model import WaveNetAmp
from dataset import AmpDataset
from export_plugin_weights import export_for_plugin


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary directories for the model"""
    model_name = config["model"]["name"]

    # Format paths with model name
    model_dir = config["paths"]["model_dir"].format(model_name=model_name)
    checkpoint_dir = config["paths"]["checkpoint_dir"].format(model_name=model_name)
    tensorboard_dir = config["paths"]["tensorboard_dir"].format(model_name=model_name)

    # Create directories
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

    print(f"Model directory: {model_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Tensorboard directory: {tensorboard_dir}")

    return model_dir, checkpoint_dir, tensorboard_dir


def esr_loss(pred, target):
    """
    Error-to-Signal Ratio loss

    WARNING: ESR is time-domain only and doesn't correlate well with perceptual quality!
    Use spectral_loss or hybrid_loss for better audio quality.
    """
    return torch.mean((pred - target) ** 2) / (torch.mean(target**2) + 1e-8)


def spectral_loss(pred, target, fft_sizes=[512, 1024, 2048], sample_rate=44100):
    """
    Multi-scale spectral loss using STFT at multiple resolutions

    This loss captures frequency-domain information that ESR misses:
    - Harmonic structure
    - Phase relationships
    - Transient preservation

    Args:
        pred: Predicted audio (batch, 1, samples)
        target: Target audio (batch, 1, samples)
        fft_sizes: List of FFT sizes for multi-resolution analysis
        sample_rate: Audio sample rate (for hop length calculation)

    Returns:
        Combined spectral loss across all scales
    """
    total_loss = 0.0

    # Squeeze channel dimension for torch.stft
    pred = pred.squeeze(1)      # (batch, samples)
    target = target.squeeze(1)  # (batch, samples)

    for fft_size in fft_sizes:
        # Hop length: 25% overlap (common for audio)
        hop_length = fft_size // 4

        # Compute STFT (Short-Time Fourier Transform)
        # Returns complex tensor: (batch, freq_bins, time_frames)
        pred_stft = torch.stft(
            pred,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=fft_size,
            window=torch.hann_window(fft_size).to(pred.device),
            return_complex=True
        )

        target_stft = torch.stft(
            target,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=fft_size,
            window=torch.hann_window(fft_size).to(target.device),
            return_complex=True
        )

        # Magnitude loss (L1 distance between spectrograms)
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        mag_loss = torch.mean(torch.abs(pred_mag - target_mag))

        # Log magnitude loss (perceptual weighting - humans hear logarithmically)
        log_mag_loss = torch.mean(
            torch.abs(
                torch.log(pred_mag + 1e-7) - torch.log(target_mag + 1e-7)
            )
        )

        # Combine magnitude losses
        total_loss += mag_loss + log_mag_loss

    # Average across all FFT sizes
    return total_loss / len(fft_sizes)


def pre_emphasis_loss(pred, target, coef=0.97):
    """
    Pre-emphasis loss - weights high frequencies more heavily

    Helps prevent the model from ignoring high-frequency detail.
    Common in speech/audio ML to improve clarity.

    Args:
        pred: Predicted audio (batch, 1, samples)
        target: Target audio (batch, 1, samples)
        coef: Pre-emphasis coefficient (0.95-0.97 typical)

    Returns:
        MSE loss on pre-emphasized signal
    """
    # Pre-emphasis filter: y[n] = x[n] - coef * x[n-1]
    # This boosts high frequencies
    pred_preemph = pred[:, :, 1:] - coef * pred[:, :, :-1]
    target_preemph = target[:, :, 1:] - coef * target[:, :, :-1]

    return torch.mean((pred_preemph - target_preemph) ** 2)


def hybrid_loss(pred, target, sample_rate=44100, esr_weight=1.0, spectral_weight=0.5, preemph_weight=0.1):
    """
    Hybrid loss combining time-domain and frequency-domain objectives

    This is the RECOMMENDED loss function for amp modeling!

    Combines:
    1. ESR (time-domain energy matching)
    2. Multi-scale spectral (frequency-domain structure)
    3. Pre-emphasis (high-frequency detail)

    Args:
        pred: Predicted audio (batch, 1, samples)
        target: Target audio (batch, 1, samples)
        sample_rate: Audio sample rate
        esr_weight: Weight for ESR loss
        spectral_weight: Weight for spectral loss
        preemph_weight: Weight for pre-emphasis loss

    Returns:
        Weighted combination of losses
    """
    loss_esr = esr_loss(pred, target)
    loss_spectral = spectral_loss(pred, target, sample_rate=sample_rate)
    loss_preemph = pre_emphasis_loss(pred, target)

    total = (
        esr_weight * loss_esr +
        spectral_weight * loss_spectral +
        preemph_weight * loss_preemph
    )

    return total


def get_loss_function(config):
    """
    Get loss function based on config

    Supports:
    - 'esr': Classic ESR (fast but poor perceptual quality)
    - 'spectral': Multi-scale spectral only
    - 'hybrid': Combined ESR + spectral + pre-emphasis (RECOMMENDED)

    Returns:
        Loss function: (pred, target) -> loss
    """
    loss_type = config.get('loss', {}).get('type', 'esr')
    sample_rate = config['data']['sample_rate']

    if loss_type == 'esr':
        print("Using ESR loss (WARNING: May produce 'funky' sound)")
        return lambda pred, target: esr_loss(pred, target)

    elif loss_type == 'spectral':
        print("Using multi-scale spectral loss")
        return lambda pred, target: spectral_loss(pred, target, sample_rate=sample_rate)

    elif loss_type == 'hybrid':
        # Get weights from config (with defaults)
        loss_config = config.get('loss', {})
        esr_w = loss_config.get('esr_weight', 1.0)
        spec_w = loss_config.get('spectral_weight', 0.5)
        pre_w = loss_config.get('preemph_weight', 0.1)

        print(f"Using hybrid loss (esr={esr_w}, spectral={spec_w}, preemph={pre_w})")

        return lambda pred, target: hybrid_loss(
            pred, target,
            sample_rate=sample_rate,
            esr_weight=esr_w,
            spectral_weight=spec_w,
            preemph_weight=pre_w
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'esr', 'spectral', or 'hybrid'")


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename):
    """Save a training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def train_epoch(model, dataloader, optimizer, device, loss_fn):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for input_audio, target_audio in pbar:
        input_audio = input_audio.to(device)
        target_audio = target_audio.to(device)

        # Forward pass
        output = model(input_audio)
        loss = loss_fn(output, target_audio)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, device, loss_fn):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for input_audio, target_audio in pbar:
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)

            output = model(input_audio)
            loss = loss_fn(output, target_audio)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def main():
    import sys

    # Parse command line arguments for config file
    config_path = "configs/config.yaml"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config" and len(sys.argv) > 2:
            config_path = sys.argv[2]

    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Setup directories
    model_dir, checkpoint_dir, tensorboard_dir = setup_directories(config)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = AmpDataset(
        input_file=config['data']['input_file'],
        output_file=config['data']['output_file'],
        segment_length=config['data']['segment_length'],
        sample_rate=config['data']['sample_rate']
    )

    # Split into train/val
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train segments: {len(train_dataset)}")
    print(f"Val segments: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize model
    print("\nInitializing model...")
    causal = config['model'].get('causal', False)  # Default to False for backward compatibility
    model = WaveNetAmp(
        channels=config['model']['channels'],
        num_layers=config['model']['num_layers'],
        kernel_size=config['model']['kernel_size'],
        dilation_base=config['model']['dilation_base'],
        causal=causal
    ).to(device)

    print(f"Model type: {'Causal (Real-time)' if causal else 'Non-causal (Offline)'}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Receptive field: {model.receptive_field()} samples")
    if causal:
        latency_ms = model.receptive_field() / config['data']['sample_rate'] * 1000
        print(f"Latency: {latency_ms:.2f}ms @ {config['data']['sample_rate']}Hz")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,        # Reduce LR by 30% when plateau detected (gentler)
        patience=15,       # Wait 15 epochs before reducing (more patient)
        min_lr=1e-6       # Don't go below this
    )

    # Get loss function from config
    print("\nSetting up loss function...")
    loss_fn = get_loss_function(config)

    # TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logs: {tensorboard_dir}")

    # Training loop
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    print("="*60)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        print(f"  Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, device, loss_fn)
        print(f"  Val Loss:   {val_loss:.6f}")

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Save checkpoint every N epochs
        if (epoch + 1) % config['training']['checkpoint_every'] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] New best model! (val_loss: {val_loss:.6f})")

            # Auto-export for VST plugin (only if causal)
            if causal:
                json_path = os.path.join(model_dir, f"{config['model']['name']}.json")
                print(f"  [*] Auto-exporting for VST plugin...")
                try:
                    export_for_plugin(best_model_path, config_path, json_path)
                    print(f"  [*] VST plugin export: {json_path}")
                except Exception as e:
                    print(f"  [!] Export failed: {e}")

            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['training']['patience']}")

        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    writer.close()

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {best_model_path}")
    print(f"View training with: tensorboard --logdir={tensorboard_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
