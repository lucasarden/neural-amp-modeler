# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural amplifier modeler using WaveNet-style architecture to emulate guitar amplifier tone. The project uses PyTorch with CUDA support to train deep learning models that learn the transfer function between clean DI guitar input and amplified/processed output.

**Supports both offline and real-time processing:**
- **Offline mode**: Non-causal models for batch processing audio files
- **Real-time mode**: Causal models for live guitar processing and VST plugins

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
# Train offline model (non-causal)
python src\train.py

# Train real-time model (causal, low latency)
python src\train.py --config configs/config_realtime.yaml
```

### Inference
```powershell
# Offline processing (batch mode)
python src\inference.py input.wav -o output.wav

# Process on CPU
python src\inference.py input.wav -d cpu

# Process large files in chunks
python src\inference.py long_file.wav -c 441000

# Real-time streaming mode (requires causal model)
python src\inference.py input.wav --streaming -c 512

# Use real-time config
python src\inference.py input.wav --config configs/config_realtime.yaml --streaming
```

### Real-Time Processing
```powershell
# List audio devices first
python src\realtime_demo.py --list-devices

# Live audio processing demo (requires PyAudio)
python src\realtime_demo.py --model models/realtime_6layer/best_model.pt

# Stereo interface: select left channel for guitar, output to stereo headphones
python src\realtime_demo.py --model models/realtime_6layer/best_model.pt --input-channels 2 --input-channel 0 --output-channels 2

# Specify audio interface and channels
python src\realtime_demo.py --model models/realtime_6layer/best_model.pt --input-device 1 --output-device 1 --input-channels 2 --input-channel 0
```

### Export for VST3 Plugin
```powershell
# Models auto-export to JSON during training!
# To manually export a model:
python src\export_plugin_weights.py --model models/realtime_6layer/best_model.pt

# Batch export all trained models
python src\batch_export_models.py

# Exported models are saved as: models/{model_name}/{model_name}.json
```

### Build VST3 Plugin (ONE TIME)
```powershell
# Build the universal loader VST3 plugin (Windows)
cd plugin
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Install to FL Studio
copy "NeuralVoxModeler_artefacts\Release\VST3\Neural Vox Modeler.vst3" "C:\Program Files\Common Files\VST3\"

# That's it! Now drag & drop ANY .json model file into the plugin
# No rebuild needed for new models!
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
- **WaveNet-style architecture** (`src/model.py`): Convolutional neural network with residual blocks
  - **ResidualBlock**: Dilated convolution with tanh activation and residual connections
    - **Non-causal mode** (offline): Uses centered padding, can see future samples
    - **Causal mode** (real-time): Uses left-only padding, no future samples
  - **WaveNetAmp**: Stacks multiple ResidualBlocks with exponentially increasing dilation rates (2^i pattern)
    - Receptive field grows exponentially with layer depth to capture long-term audio dependencies
    - Input: 1 channel mono audio → channels (16 default) → 1 channel output
    - Supports both causal and non-causal operation via `causal` parameter
  - **StreamingWaveNetAmp**: Wrapper for real-time chunk processing with buffer management
    - Maintains context buffer between chunks for causal models
    - Enables processing in small chunks (128-512 samples) for real-time use

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
- **Data**: sample rate, file paths, segment length, train/val split
- **Model**: architecture params (channels, layers, kernel size, dilation base, **causal** mode)
- **Training**: batch size, epochs, learning rate, weight decay, early stopping patience
- **Paths**: dynamically formatted with model name using `{model_name}` placeholder

**Configuration files:**
- `config.yaml`: Default non-causal config for offline processing
- `config_realtime.yaml`: Causal config optimized for low-latency real-time (~4.3ms)

## Project Structure

```
src/
├── model.py                   # WaveNet architecture (causal & non-causal) + StreamingWaveNetAmp
├── dataset.py                 # Audio data loading and preprocessing
├── train.py                   # Training loop with auto-export to JSON
├── inference.py               # Model inference (batch & streaming modes)
├── export_plugin_weights.py  # Export model weights to JSON for VST plugin
├── batch_export_models.py    # Batch export all trained models
└── realtime_demo.py          # Live audio processing demo (PyAudio)

plugin/                        # C++ VST3 Plugin (Universal Loader!)
├── Source/
│   ├── PluginProcessor.h/cpp  # Audio processing + dynamic model loading
│   ├── PluginEditor.h/cpp     # GUI with drag & drop + Load Model button
│   └── ModelLoader.h          # Parse JSON weights
├── CMakeLists.txt             # Build configuration (JUCE + CMake)
└── README.md                  # Build instructions

configs/
├── config.yaml                # Default config (non-causal)
└── config_realtime.yaml       # Real-time config (causal, low latency)

models/{model_name}/
├── best_model.pt              # Best model saved during training
├── {model_name}.json          # Auto-exported for VST plugin
└── checkpoints/               # Periodic training checkpoints

data/raw/                      # Raw audio recordings (input/output pairs)
runs/{model_name}/             # TensorBoard logs

REALTIME.md                    # Real-time processing analysis & requirements
PLUGIN_GUIDE.md                # Universal loader VST plugin guide (JUCE-based)
PLUGIN_USAGE.md                # End-user guide for guitarists
README_REALTIME.md             # Quick start for real-time processing
```

## Current State

**Fully functional for both offline and real-time processing!**

### Offline Mode (Non-Causal)
- ✅ Train models with `config.yaml` (non-causal architecture)
- ✅ Process entire audio files at once
- ✅ Best quality (can see future samples)
- ✅ Use for: batch processing, testing, highest quality output

### Real-Time Mode (Causal)
- ✅ Train models with `config_realtime.yaml` (causal architecture)
- ✅ Streaming inference with buffer management
- ✅ Low latency (~4.3ms with 6-layer config)
- ✅ Export to ONNX for VST plugins
- ✅ Live audio processing demo
- ✅ Use for: VST plugins, live guitar processing, real-time applications

### VST Plugin Development
- ✅ **Complete C++ VST3 plugin with universal model loader!** (JUCE-based)
- ✅ **Build once, load any model** - Drag & drop .json files at runtime
- ✅ Professional GUI with drag & drop, file browser, and status indicators
- ✅ Controls: Input/Output Gain, Mix, Bypass
- ✅ **Per-project model persistence** - DAW remembers which model you used
- ✅ Ultra-low latency (~3-5ms) native C++ performance
- ✅ Cross-platform: Windows, macOS, Linux
- ✅ Ready for FL Studio, Ableton, Reaper, all major DAWs
- ✅ Full source code in `plugin/` directory
- 📚 See `PLUGIN_GUIDE.md` for build and usage instructions

## Real-Time Performance

### Latency Calculation
Model latency is determined by the receptive field:

```
receptive_field = 1 + sum((kernel_size - 1) * dilation_base^i for i in range(num_layers))
latency_ms = receptive_field / sample_rate * 1000
```

**Examples:**
- 10 layers, kernel=3, dilation=2: 2047 samples = **46.4ms** @ 44.1kHz
- 6 layers, kernel=3, dilation=2: 191 samples = **4.3ms** @ 44.1kHz ✅
- 5 layers, kernel=3, dilation=2: 95 samples = **2.2ms** @ 44.1kHz ✅✅

### Best Practices for Real-Time

1. **Always use causal models** - Set `causal: true` in config for real-time
2. **Start with 6 layers** - Good balance between quality and latency
3. **Test streaming mode** - Use `--streaming` flag to test real-time behavior
4. **Optimize if needed** - Reduce layers or channels if processing is too slow
5. **Build plugin once** - The universal loader VST lets you load any model at runtime
6. **Auto-export enabled** - Models automatically export to JSON during training

### Quality vs Latency Trade-off
- **More layers** = larger receptive field = better quality but higher latency
- **Fewer layers** = smaller receptive field = lower latency but may reduce quality
- Sweet spot for guitar: **6-8 layers** (~4-20ms latency)

## Model Naming Convention

Models are organized by name (e.g., "ts9_pushed", "vox_ac30"). The model name in `configs/config.yaml` determines:
- Model save directory: `models/{model_name}/`
- Checkpoint directory: `models/{model_name}/checkpoints/`
- Best model path: `models/{model_name}/best_model.pt`
- TensorBoard logs: `runs/{model_name}/`

When working with different amp models, update the `model.name` field in config.yaml.
