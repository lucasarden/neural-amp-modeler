# Neural Vox Modeler - VST3 Plugin

Professional C++ VST3 plugin for real-time neural guitar amp modeling using JUCE.

## 🎸 Features

- ✅ Real-time neural network inference (C++ performance)
- ✅ Professional VST3 plugin for FL Studio, Ableton, Reaper, etc.
- ✅ Ultra-low latency (~3-5ms depending on model)
- ✅ Clean GUI with Input Gain, Output Gain, Mix, and Bypass controls
- ✅ Uses your exact trained model weights
- ✅ Automatic latency compensation
- ✅ Cross-platform (Windows, macOS, Linux)

---

## 📋 Prerequisites

### Required
- **CMake** (3.15 or higher): https://cmake.org/download/
- **C++ Compiler**:
  - **Windows**: Visual Studio 2019/2022 (Community Edition is free)
  - **macOS**: Xcode
  - **Linux**: GCC/Clang

### Optional
- **Git** (if you want to clone JUCE automatically)

---

## 🚀 Quick Start

### Step 1: Export Your Model Weights

From the project root directory:

```powershell
# Export your trained causal model to JSON
python src/export_plugin_weights.py --model models/realtime_6layer/best_model.pt
```

This creates `models/realtime_6layer/realtime_6layer_weights.json`

### Step 2: Build the Plugin

```powershell
# Navigate to plugin directory
cd plugin

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build (choose one):

# Windows - Build with Visual Studio
cmake --build . --config Release

# macOS/Linux - Build with make
make -j8

# Or open in Visual Studio and build from IDE (Windows)
start NeuralVoxModeler.sln
```

### Step 3: Copy Model Weights

```powershell
# Copy your exported JSON to the plugin's Resources folder
copy ..\..\models\realtime_6layer\realtime_6layer_weights.json NeuralVoxModeler_artefacts\Release\VST3\Neural Vox Modeler.vst3\Contents\Resources\model.json

# On macOS/Linux:
# cp ../../models/realtime_6layer/realtime_6layer_weights.json NeuralVoxModeler_artefacts/Release/VST3/Neural\ Vox\ Modeler.vst3/Contents/Resources/model.json
```

### Step 4: Install VST3

```powershell
# Windows: Copy to VST3 folder
copy "NeuralVoxModeler_artefacts\Release\VST3\Neural Vox Modeler.vst3" "C:\Program Files\Common Files\VST3\"

# macOS:
# cp -r "NeuralVoxModeler_artefacts/Release/VST3/Neural Vox Modeler.vst3" ~/Library/Audio/Plug-Ins/VST3/

# Linux:
# cp -r "NeuralVoxModeler_artefacts/Release/VST3/Neural Vox Modeler.vst3" ~/.vst3/
```

### Step 5: Use in FL Studio

1. Open FL Studio
2. Go to **Options → Manage Plugins → Find Plugins**
3. Rescan for new plugins
4. Find "Neural Vox Modeler" in your effects list
5. Add to mixer channel → Process guitar! 🎸

---

## 📁 Project Structure

```
plugin/
├── CMakeLists.txt              # Build configuration
├── Source/
│   ├── PluginProcessor.h       # Audio processing logic
│   ├── PluginProcessor.cpp     # Implementation
│   ├── PluginEditor.h          # GUI header
│   ├── PluginEditor.cpp        # GUI implementation
│   └── ModelLoader.h           # Load JSON weights
├── build/                      # Build directory (created by you)
│   └── NeuralVoxModeler_artefacts/
│       └── Release/VST3/
│           └── Neural Vox Modeler.vst3/
│               └── Contents/Resources/
│                   └── model.json  # Your model weights go here!
└── README.md                   # This file
```

---

## 🛠️ Detailed Build Instructions

### Windows (Visual Studio)

1. **Install Visual Studio 2022** (Community Edition):
   - Download: https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"

2. **Install CMake**:
   - Download: https://cmake.org/download/
   - Add to PATH during installation

3. **Build**:
   ```powershell
   cd plugin
   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release
   ```

4. **Or use Visual Studio IDE**:
   ```powershell
   cmake ..
   start NeuralVoxModeler.sln
   # Build → Build Solution (Ctrl+Shift+B)
   ```

### macOS (Xcode)

1. **Install Xcode** from App Store

2. **Install CMake** via Homebrew:
   ```bash
   brew install cmake
   ```

3. **Build**:
   ```bash
   cd plugin
   mkdir build
   cd build
   cmake ..
   make -j$(sysctl -n hw.ncpu)
   ```

### Linux

1. **Install dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake build-essential libasound2-dev libfreetype6-dev libx11-dev libxinerama-dev libxrandr-dev libxcursor-dev libgl1-mesa-dev

   # Fedora
   sudo dnf install cmake gcc-c++ alsa-lib-devel freetype-devel libX11-devel libXinerama-devel libXrandr-devel libXcursor-dev mesa-libGL-devel
   ```

2. **Build**:
   ```bash
   cd plugin
   mkdir build
   cd build
   cmake ..
   make -j$(nproc)
   ```

---

## 🎛️ Plugin Controls

- **Input Gain**: -12dB to +12dB - Adjust input level to match your guitar
- **Output Gain**: -12dB to +12dB - Adjust output volume
- **Mix**: 0% to 100% - Blend between dry (clean) and wet (processed) signal
- **Bypass**: Toggle processing on/off

---

## ⚡ Performance

- **CPU Usage**: Very low (~1-2% on modern CPUs)
- **Latency**: 3-5ms (model receptive field) + buffer size
- **Memory**: ~50KB (tiny!)
- **Real-time Safe**: Yes, all processing is lock-free

---

## 🐛 Troubleshooting

### "Model file not found" error

**Problem**: Plugin can't find `model.json`

**Solution**: Make sure to copy your exported JSON file to:
```
plugin/build/NeuralVoxModeler_artefacts/Release/VST3/Neural Vox Modeler.vst3/Contents/Resources/model.json
```

### Plugin doesn't load in DAW

**Solutions**:
1. Make sure you built in **Release** mode: `cmake --build . --config Release`
2. Check VST3 is in correct folder: `C:\Program Files\Common Files\VST3\`
3. Rescan plugins in your DAW
4. Check DAW supports VST3 (FL Studio, Ableton, Reaper all do)

### Crashing or audio glitches

**Solutions**:
1. Make sure your model is **causal** (trained with `causal: true`)
2. Increase buffer size in your DAW (256-512 samples)
3. Check model file is valid JSON (use export script, don't edit manually)

### Build errors

**CMake can't find JUCE**:
- Make sure you have internet connection (JUCE downloads automatically)
- Try deleting `build/` folder and rebuild from scratch

**Compiler errors**:
- Make sure you have C++17 support
- Visual Studio 2019+ on Windows
- Xcode 12+ on macOS
- GCC 9+ or Clang 10+ on Linux

---

## 🔧 Advanced Configuration

### Change Plugin Name

Edit `CMakeLists.txt`:
```cmake
COMPANY_NAME "YourCompany"
PRODUCT_NAME "Your Plugin Name"
```

### Build Multiple Formats

Edit `CMakeLists.txt`:
```cmake
set(FORMATS VST3 AU Standalone)  # Add AU for macOS
```

### Optimize for Size

Add to `CMakeLists.txt`:
```cmake
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Os")  # Optimize for size
```

---

## 📊 Model Requirements

Your model **MUST** be:
- ✅ **Causal** - Trained with `causal: true` in config
- ✅ **6-10 layers** - For reasonable latency
- ✅ **Mono** - Single channel input/output
- ✅ **44.1kHz or 48kHz** - Standard sample rates

---

## 📚 Additional Resources

- **JUCE Documentation**: https://docs.juce.com/
- **JUCE Forum**: https://forum.juce.com/
- **CMake Tutorial**: https://cmake.org/cmake/help/latest/guide/tutorial/

---

## 🎉 Success!

If you see "Model Loaded | Latency: X.Xms" in the plugin GUI, you're ready to rock! 🎸🔥

Add the plugin to a mixer channel in FL Studio, plug in your guitar, and enjoy your neural amp model in real-time!

---

## 📄 License

This plugin code is part of the Neural Vox Modeler project.
JUCE is licensed under GPLv3 or commercial license.
