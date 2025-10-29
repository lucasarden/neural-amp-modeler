# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural amplifier modeler using WaveNet-style architecture to emulate guitar amplifier tone. Currently in data collection phase. The project uses PyTorch with CUDA support to train deep learning models that learn the transfer function between clean DI guitar input and amplified/processed output.

## Common Commands

### Environment Setup
```powershell
# Activate virtual environment (already configured)
.\venv\Scripts\activate

# Verify CUDA installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Testing
```powershell
# Test model architecture
python src\model.py

# Test dataset loading (requires audio files in data\raw\)
python src\dataset.py
```

### Training
```powershell
# Train the model (after data collection is complete)
python src\train.py
```

### Inference
```powershell
# Process audio through trained model
python src\inference.py input.wav -o output.wav

# Process on CPU
python src\inference.py input.wav -d cpu

# Process large files in chunks
python src\inference.py long_file.wav -c 441000
```

### Comparison and Analysis
```powershell
# Compare model output vs real amp output
python src\compare_audio.py data/raw/ts9_test1_out_FP32.wav test_output.wav

# With visualization
python src\compare_audio.py target.wav prediction.wav -p

# Save plot
python src\compare_audio.py target.wav prediction.wav -o comparison.png
```

## Architecture

### Model Structure
- **WaveNet-style architecture** (`src/model.py`): Causal convolutional neural network with residual blocks
  - ResidualBlock: Dilated convolution with tanh activation and residual connections
  - WaveNetAmp: Stacks multiple ResidualBlocks with exponentially increasing dilation rates (2^i pattern)
  - Receptive field grows exponentially with layer depth to capture long-term audio dependencies
  - Input: 1 channel mono audio → channels (16 default) → 1 channel output

### Data Pipeline
- **AmpDataset** (`src/dataset.py`): Loads paired input/output audio files and creates training segments
  - Normalizes audio to [-1, 1] range
  - Splits long recordings into fixed-length segments (default 8192 samples)
  - Returns paired tensors with shape (1, segment_length) for input/output

### Training Framework
- **Loss function**: ESR (Error-to-Signal Ratio) - specialized for audio applications
- **Early stopping**: Monitors validation loss with configurable patience
- **Checkpointing**: Saves periodic checkpoints + best model based on validation loss
- **Configuration**: All hyperparameters in `configs/config.yaml`

### Configuration System
All settings in `configs/config.yaml`:
- Data: sample rate, file paths, segment length, train/val split
- Model: architecture params (channels, layers, kernel size, dilation base)
- Training: batch size, epochs, learning rate, weight decay, early stopping patience
- Paths: dynamically formatted with model name using `{model_name}` placeholder

## Project Structure

```
src/
├── model.py         # WaveNet architecture
├── dataset.py       # Audio data loading and preprocessing
├── train.py         # Training loop with checkpointing
└── inference.py     # Model inference on new audio

models/{model_name}/
├── best_model.pt    # Best model saved during training (tracked in git)
└── checkpoints/     # Periodic training checkpoints (not tracked)

data/raw/            # Raw audio recordings (input/output pairs)
configs/config.yaml  # Central configuration file
runs/{model_name}/   # TensorBoard logs
```

## Current State

**Fully functional for offline/batch processing!** The model can be trained and used to process audio files. However, it is **not real-time ready** - see `REALTIME.md` for details on what's needed for real-time guitar processing.

Key limitations for real-time:
- Architecture uses non-causal convolutions (padding="same") - requires retraining with causal padding
- No streaming/buffering infrastructure for chunk-based processing
- See `REALTIME.md` for comprehensive analysis and required changes

## Model Naming Convention

Models are organized by name (e.g., "ts9_pushed", "vox_ac30"). The model name in `configs/config.yaml` determines:
- Model save directory: `models/{model_name}/`
- Checkpoint directory: `models/{model_name}/checkpoints/`
- Best model path: `models/{model_name}/best_model.pt`
- TensorBoard logs: `runs/{model_name}/`

When working with different amp models, update the `model.name` field in config.yaml.
