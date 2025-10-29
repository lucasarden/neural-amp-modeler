# VST Plugin Development Guide

Turn your trained neural amp models into professional VST3 plugins for FL Studio!

## Overview

The Neural Vox Modeler includes a complete **C++ VST3 plugin** using the industry-standard **JUCE framework** with **universal model loading** - build once, load any model!

### Why This Approach?

- ✅ **Professional Quality** - JUCE is used by top plugin companies (Ableton, Native Instruments, etc.)
- ✅ **Ultra-Low Latency** - Native C++ performance (~3-5ms total latency)
- ✅ **Universal Loader** - Load ANY model at runtime (no rebuild needed!)
- ✅ **Cross-Platform** - Works on Windows, macOS, and Linux
- ✅ **Full Source Code** - Complete implementation included, customize as needed
- ✅ **FL Studio Ready** - Works in FL Studio, Ableton, Reaper, and all major DAWs

### New Workflow

**Build the plugin ONCE, then:**
- Train models with automatic JSON export
- Load any model via file browser or drag & drop
- Switch between models instantly in your DAW
- No rebuild or reinstall required!

---

## 🚀 Quick Start (3 Steps!)

### Step 1: Build the VST3 Plugin (ONE TIME)

```powershell
cd plugin
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**First time?** See detailed build instructions in `plugin/README.md`

### Step 2: Install in FL Studio (ONE TIME)

```powershell
# Copy VST3 to system folder
copy "NeuralVoxModeler_artefacts\Release\VST3\Neural Vox Modeler.vst3" "C:\Program Files\Common Files\VST3\"
```

Open FL Studio → Options → Manage Plugins → Rescan

### Step 3: Train Models (Auto-exports for VST!)

```powershell
python src/train.py --config configs/config_realtime.yaml
```

**That's it!** Your model is automatically exported to JSON when training completes.

The plugin saves `models/{model_name}/{model_name}.json` - just drag & drop this file into the plugin in FL Studio!

**Critical**: Make sure your config has `causal: true`!

---

## 🎛️ Using the Plugin

### Loading Models

**Method 1: Drag & Drop** (Easiest!)
1. Open the Neural Vox Modeler plugin in your DAW
2. Drag your `.json` model file onto the plugin window
3. Done! The model is loaded and ready to rock 🎸

**Method 2: File Browser**
1. Click the "LOAD MODEL" button in the plugin
2. Navigate to your model's `.json` file (e.g., `models/realtime_6layer/realtime_6layer.json`)
3. Select it and click Open
4. Done!

### Model Status

- **Green** "Model Name": Model loaded successfully ✅
- **Orange** "No Model Loaded": Click Load Model or drag & drop a file ⚠️
- **Error Dialog**: Model failed to load (check it's causal and valid JSON) ❌

### Model Persistence

The plugin remembers which model you loaded for each DAW project! When you reopen a project, it automatically loads the same model you were using.

---

## 📋 Prerequisites

### Required Software

**Windows:**
- Visual Studio 2019/2022 (Community Edition is free)
  - Download: https://visualstudio.microsoft.com/downloads/
  - Select "Desktop development with C++"
- CMake 3.15+
  - Download: https://cmake.org/download/

**macOS:**
- Xcode (from App Store)
- CMake via Homebrew: `brew install cmake`

**Linux:**
```bash
sudo apt-get install cmake build-essential libasound2-dev libfreetype6-dev libx11-dev libxinerama-dev libxrandr-dev libxcursor-dev libgl1-mesa-dev
```

---

## 🛠️ Detailed Build Process

### Windows (Visual Studio)

```powershell
# 1. Navigate to plugin directory
cd plugin

# 2. Create build directory
mkdir build
cd build

# 3. Configure (downloads JUCE automatically)
cmake ..

# 4. Build
cmake --build . --config Release

# Or open in Visual Studio:
start NeuralVoxModeler.sln
# Then: Build → Build Solution
```

**Build Output:**
```
plugin/build/NeuralVoxModeler_artefacts/Release/VST3/Neural Vox Modeler.vst3
```

### macOS

```bash
cd plugin
mkdir build
cd build
cmake ..
make -j8
```

**Build Output:**
```
plugin/build/NeuralVoxModeler_artefacts/Release/VST3/Neural Vox Modeler.vst3
```

### Linux

```bash
cd plugin
mkdir build
cd build
cmake ..
make -j$(nproc)
```

---

## 📁 Plugin Architecture

### Source Code Structure

```
plugin/Source/
├── PluginProcessor.h/cpp    # Audio processing + neural network
├── PluginEditor.h/cpp        # GUI with controls
└── ModelLoader.h             # Load JSON weights
```

### Key Components

**PluginProcessor**:
- Loads model weights from JSON
- Implements neural network inference in C++
- Manages audio buffer processing
- Handles parameter controls

**PluginEditor**:
- Clean, professional GUI with drag & drop support
- "Load Model" button with file browser
- Real-time model status display (green/orange/red indicators)
- Input Gain, Output Gain, Mix, Bypass controls
- Dark theme design

**ModelLoader**:
- Parses JSON weight file
- Loads architecture and weights
- Validates model structure

---

## 🎛️ Plugin Features

### Parameters

- **Input Gain**: -12dB to +12dB
  - Adjust input level to match your guitar
- **Output Gain**: -12dB to +12dB
  - Control overall volume
- **Mix**: 0% to 100%
  - Blend dry (clean) and wet (processed) signals
- **Bypass**: On/Off
  - Toggle processing

### Performance

- **CPU Usage**: ~1-2% (very efficient!)
- **Latency**: Model receptive field (~3-5ms) + DAW buffer
- **Memory**: ~50KB (tiny!)
- **Real-time Safe**: Yes, lock-free processing

---

## 🐛 Troubleshooting

### "No Model Loaded" Warning

**Problem**: Plugin shows orange "No Model Loaded" warning

**Solution**:
1. Click "LOAD MODEL" button or drag & drop a `.json` file
2. Make sure you've trained a causal model: `python src/train.py --config configs/config_realtime.yaml`
3. The model JSON is auto-exported to `models/{model_name}/{model_name}.json`

### "Model is not causal" Error

**Problem**: Plugin rejects model with error dialog

**Solution**:
1. Model must be trained with `causal: true` in config
2. Use `configs/config_realtime.yaml` for training
3. Non-causal models cannot be used for real-time processing

### Plugin Not Showing in FL Studio

**Solutions**:
1. Ensure built in **Release** mode: `cmake --build . --config Release`
2. Copy to correct folder: `C:\Program Files\Common Files\VST3\`
3. Rescan plugins in FL Studio: Options → Manage Plugins → Find Plugins
4. Check FL Studio supports VST3 (all recent versions do)

### Audio Crackling/Glitches

**Solutions**:
1. **Increase buffer size** in FL Studio (256-512 samples)
2. Ensure model is **causal** (trained with `causal: true`)
3. Check JSON file is valid (don't edit manually)
4. Try reducing model layers (6 instead of 10)

### Build Errors

**CMake can't find JUCE**:
- Check internet connection (JUCE downloads automatically)
- Delete `build/` folder and try again
- Check CMake version: `cmake --version` (need 3.15+)

**Compiler errors**:
- Visual Studio 2019+ required (Windows)
- Xcode 12+ required (macOS)
- GCC 9+ or Clang 10+ required (Linux)

---

## ⚡ Performance Optimization

### Reduce Latency

Train with fewer layers:
```yaml
# configs/config_ultrafast.yaml
model:
  num_layers: 5  # Instead of 6
  # Latency: ~1.5ms!
```

### Optimize Build

Add to CMakeLists.txt:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
```

### Profile in DAW

1. Load plugin in FL Studio
2. Check CPU meter
3. If high: reduce model size or increase buffer

---

## 🎨 Customization

### Change Plugin Name

Edit `plugin/CMakeLists.txt`:
```cmake
COMPANY_NAME "YourCompany"
PRODUCT_NAME "Your Custom Name"
```

### Add More Parameters

Edit `PluginProcessor.cpp` → `createParameterLayout()`:
```cpp
params.push_back(std::make_unique<juce::AudioParameterFloat>(
    "drive", "Drive",
    juce::NormalisableRange<float>(1.0f, 10.0f, 0.1f),
    1.0f
));
```

### Modify GUI

Edit `PluginEditor.cpp` → `paint()` and `resized()` methods.

---

## 📊 Model Requirements

Your model **MUST** be:
- ✅ **Causal** - Trained with `causal: true`
- ✅ **Mono** - Single channel input/output
- ✅ **6-10 layers** - For good latency
- ✅ **44.1kHz or 48kHz** - Standard sample rates

---

## 🎉 Success Checklist

**One-Time Setup:**
- [ ] Plugin built in **Release** mode
- [ ] VST3 copied to system plugins folder (`C:\Program Files\Common Files\VST3\`)
- [ ] FL Studio rescanned plugins
- [ ] Plugin shows in FL Studio mixer

**For Each Model:**
- [ ] Model trained with `causal: true` (use `configs/config_realtime.yaml`)
- [ ] Training auto-exported `.json` file (in `models/{model_name}/`)
- [ ] Drag & drop `.json` into plugin or use Load Model button
- [ ] Plugin shows green "Model Loaded" status
- [ ] Guitar sounds amazing! 🎸🔥

**Switch Models Anytime:**
- Just drag & drop a different `.json` file - no rebuild needed!

---

## 📚 Additional Resources

- **Complete build guide**: `plugin/README.md`
- **JUCE Documentation**: https://docs.juce.com/
- **JUCE Tutorials**: https://docs.juce.com/master/tutorial_create_projucer_basic_plugin.html
- **JUCE Forum**: https://forum.juce.com/

---

## 🔥 Pro Tips

1. **Organize your models**: Keep all `.json` files in a single folder for easy access
2. **Batch export existing models**: Use `python src/batch_export_models.py` to export all trained models at once
3. **Test in standalone first**: Build includes a standalone app for testing
4. **Check latency compensation**: Enable in DAW for proper timing
5. **Backup your models**: Keep trained `.pt` AND `.json` files safe!
6. **Train multiple variants**: Try different layer counts (5, 6, 8 layers) for different latency/quality tradeoffs
7. **Per-project models**: The plugin remembers which model you used in each DAW project

---

Enjoy your professional neural amp plugin in FL Studio! 🎸✨
