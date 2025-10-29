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
    """Error-to-Signal Ratio loss"""
    return torch.mean((pred - target) ** 2) / (torch.mean(target**2) + 1e-8)


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


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for input_audio, target_audio in pbar:
        input_audio = input_audio.to(device)
        target_audio = target_audio.to(device)

        # Forward pass
        output = model(input_audio)
        loss = esr_loss(output, target_audio)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for input_audio, target_audio in pbar:
            input_audio = input_audio.to(device)
            target_audio = target_audio.to(device)

            output = model(input_audio)
            loss = esr_loss(output, target_audio)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / len(dataloader)


def main():
    # Load config
    config = load_config()

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
    model = WaveNetAmp(
        channels=config['model']['channels'],
        num_layers=config['model']['num_layers'],
        kernel_size=config['model']['kernel_size'],
        dilation_base=config['model']['dilation_base']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Receptive field: {model.receptive_field()} samples")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

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
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"  Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"  Val Loss:   {val_loss:.6f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint every N epochs
        if (epoch + 1) % config['training']['checkpoint_every'] == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] New best model! (val_loss: {val_loss:.6f})")
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
