# Real-Time Neural Amp Modeling - Quick Start Guide

Your neural amp modeler now supports **real-time processing** for live guitar use and VST plugins! 🎸

## Quick Start

### 1. Train a Real-Time Model

```powershell
# Train a causal model optimized for low latency (~4.3ms)
python src/train.py --config configs/config_realtime.yaml
```

The real-time config uses 6 layers instead of 10, reducing latency from 46ms to 4.3ms while maintaining quality.

### 2. Test Streaming Mode

```powershell
# Test your trained model in streaming mode (simulates real-time)
python src/inference.py input_guitar.wav --streaming --config configs/config_realtime.yaml
```

### 3. Try Live Processing (Optional)

```powershell
# Install PyAudio first
pip install pyaudio

# List your audio devices first
python src/realtime_demo.py --list-devices

# Run live audio demo (stereo input, select left channel, mono output to stereo headphones)
python src/realtime_demo.py --model models/realtime_6layer/best_model.pt --input-channels 2 --input-channel 0

# Use specific audio interface (replace device numbers with yours)
python src/realtime_demo.py --model models/realtime_6layer/best_model.pt --input-device 1 --output-device 1 --input-channels 2 --input-channel 0
```

### 4. Use in VST Plugin

**No export needed!** Models are automatically exported to JSON when training completes.

```powershell
# Find your model JSON at:
models/realtime_6layer/realtime_6layer.json
```

Just drag & drop this file into the Neural Vox Modeler plugin in your DAW!

---

## What's New?

### Two Operating Modes

**Offline Mode (Original)**
- Train with `config.yaml`
- Non-causal (can see future samples)
- Best quality for batch processing
- ~46ms latency (doesn't matter for offline)

**Real-Time Mode (NEW!)**
- Train with `config_realtime.yaml`
- Causal (no future samples)
- Low latency (~2-5ms)
- Perfect for live guitar and plugins

### Key Features

✅ **Causal Architecture** - No future samples, real-time ready
✅ **Streaming Inference** - Process audio in chunks with buffer management
✅ **Low Latency** - As low as 2.88ms with 6-layer config
✅ **Universal VST Loader** - Build plugin once, load any model at runtime
✅ **Auto-Export** - Models automatically export to JSON on training completion
✅ **Live Demo** - Test with real audio input/output
✅ **Backward Compatible** - All existing features still work

---

## Understanding Latency

Latency depends on the model's receptive field:

| Layers | Receptive Field | Latency @ 44.1kHz | Use Case |
|--------|-----------------|-------------------|-----------|
| 10     | 2047 samples    | 46.4ms           | Offline processing |
| 8      | 511 samples     | 11.6ms           | Looser real-time |
| 6      | 127 samples     | 2.88ms           | **Recommended** ✅ |
| 5      | 63 samples      | 1.43ms           | Ultra low latency |

**Sweet spot:** 6-8 layers gives great quality with imperceptible latency.

---

## File Structure

```
configs/
├── config.yaml              # Original (non-causal, offline)
└── config_realtime.yaml     # NEW: Causal, low-latency

src/
├── model.py                   # Updated: Causal support + StreamingWaveNetAmp
├── train.py                   # Updated: Auto-exports JSON on training completion
├── inference.py               # Updated: Streaming mode (--streaming flag)
├── export_plugin_weights.py  # NEW: Export models to JSON for VST plugin
├── batch_export_models.py    # NEW: Batch export all trained models
└── realtime_demo.py          # NEW: Live audio processing

PLUGIN_GUIDE.md               # NEW: How to build universal loader VST plugin
PLUGIN_USAGE.md               # NEW: End-user guide for guitarists
REALTIME_IMPLEMENTATION.md    # NEW: Technical implementation details
```

---

## Next Steps

### For Offline Processing
Continue using `config.yaml` and existing workflow. Nothing changed!

### For Real-Time / VST Plugin

1. **Build VST plugin (ONE TIME):**
   ```powershell
   # See PLUGIN_GUIDE.md for full instructions
   cd plugin
   mkdir build && cd build
   cmake .. -G "Visual Studio 17 2022" -A x64
   cmake --build . --config Release

   # Install to system plugins folder
   copy "NeuralVoxModeler_artefacts\Release\VST3\Neural Vox Modeler.vst3" "C:\Program Files\Common Files\VST3\"
   ```

2. **Train causal model (auto-exports!):**
   ```powershell
   python src/train.py --config configs/config_realtime.yaml
   # Your model is automatically exported to JSON!
   ```

3. **Test quality in streaming mode:**
   ```powershell
   python src/inference.py test.wav --streaming --config configs/config_realtime.yaml
   ```

4. **Load in DAW:**
   - Open Neural Vox Modeler plugin in FL Studio/Ableton/etc.
   - Drag & drop `models/realtime_6layer/realtime_6layer.json` into plugin
   - Rock! 🎸🔥

**Train more models? Just drag & drop the new JSON - no plugin rebuild needed!**

---

## Configuration Options

### config_realtime.yaml

```yaml
model:
  name: "realtime_6layer"
  channels: 16          # Internal channel count
  num_layers: 6         # Reduced for low latency
  kernel_size: 3
  dilation_base: 2
  causal: true          # CRITICAL for real-time!
```

**Tuning Tips:**
- **Reduce latency:** Decrease `num_layers` (5-6)
- **Increase quality:** Increase `num_layers` (7-8) or `channels` (24-32)
- **Always keep** `causal: true` for real-time

---

## Channel Configuration for Live Processing

The `realtime_demo.py` script supports flexible input/output channel configuration:

### Input Channel Selection

```powershell
# Stereo audio interface - select left channel (guitar input)
--input-channels 2 --input-channel 0

# Stereo audio interface - select right channel
--input-channels 2 --input-channel 1

# Mono audio interface
--input-channels 1 --input-channel 0  # (default if not specified)
```

### Output Configuration

```powershell
# Stereo headphones (mono signal duplicated to both ears) - DEFAULT
--output-channels 2

# True mono output (single channel)
--output-channels 1
```

### Common Scenarios

**Guitar plugged into left input of stereo interface, listening on stereo headphones:**
```powershell
python src/realtime_demo.py --model models/realtime_6layer/best_model.pt \
  --input-channels 2 --input-channel 0 --output-channels 2
```

**Mono audio interface, mono headphones:**
```powershell
python src/realtime_demo.py --model models/realtime_6layer/best_model.pt \
  --input-channels 1 --input-channel 0 --output-channels 1
```

**4-channel interface, select input 3:**
```powershell
python src/realtime_demo.py --model models/realtime_6layer/best_model.pt \
  --input-channels 4 --input-channel 2
```

## Troubleshooting

### "Model must be causal for streaming mode"
- Make sure your config has `causal: true`
- Retrain the model with the causal config

### "Processing too slow for real-time"
- Reduce `num_layers` (try 5)
- Reduce `channels` (try 12)
- Use GPU (CUDA) instead of CPU

### "Quality degraded with causal model"
- Normal - causal models can't see future
- Try 7-8 layers instead of 6
- Ensure sufficient training data
- Compare ESR loss between causal and non-causal

### "PyAudio installation fails"
Windows:
```powershell
pip install pipwin
pipwin install pyaudio
```

macOS:
```bash
brew install portaudio
pip install pyaudio
```

Linux:
```bash
sudo apt-get install python3-pyaudio
```

---

## Performance Benchmarks

**RTX 2070 SUPER:**
- 6-layer model: 7,889 parameters
- Chunk size: 512 samples (11.6ms)
- Processing time: ~1-2ms
- Real-time capable: ✅ Yes (plenty of headroom)

**Latency Budget:**
- Model latency: 2.88ms (6 layers)
- Buffer latency: 11.6ms (512 samples @ 44.1kHz)
- **Total:** ~14ms (imperceptible for guitar)

---

## Resources

- **PLUGIN_GUIDE.md** - Complete VST plugin development guide with universal loader
- **PLUGIN_USAGE.md** - End-user guide for guitarists
- **REALTIME_IMPLEMENTATION.md** - Technical implementation details
- **REALTIME.md** - Original analysis (now implemented!)
- **CLAUDE.md** - Complete project documentation

### External Links
- [JUCE](https://juce.com/) - Cross-platform audio plugin framework (used in our plugin)
- [JUCE Forum](https://forum.juce.com/) - Community support for plugin development

---

## FAQ

**Q: Can I use my existing trained models?**
A: Non-causal models can't be used for real-time. You need to retrain with `causal: true`.

**Q: Will quality be worse with causal models?**
A: Slightly, since they can't see future samples. But with proper training and 6-8 layers, the difference is minimal.

**Q: How do I make a VST plugin?**
A: Build the included JUCE plugin once, then load any trained model at runtime! See PLUGIN_GUIDE.md for step-by-step instructions.

**Q: Can I run this on CPU?**
A: Yes, but GPU is recommended for real-time. Test with `realtime_demo.py` to check performance.

**Q: What's the smallest latency possible?**
A: Theoretically ~1-2ms with 5 layers, but quality drops. 6 layers (~3ms) is the sweet spot.

---

## Support

For issues, questions, or contributions:
- Check existing documentation (PLUGIN_GUIDE.md, REALTIME_IMPLEMENTATION.md)
- Review CLAUDE.md for full project overview
- Test with the demo scripts before building plugins

---

**Happy modeling! 🎸🔥**
