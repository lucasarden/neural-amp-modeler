# Neural Vox Modeler - User Guide

Professional neural amp modeling plugin for guitarists! 🎸

## What is This?

Neural Vox Modeler is a VST3 plugin that uses AI to replicate the sound of guitar amplifiers and pedals. Instead of modeling circuits or components, it learns the exact tone of your gear through deep learning.

**Think of it like:** Taking a sonic photograph of your favorite amp!

---

## Getting Started

### System Requirements

- **Windows**: Windows 10/11
- **macOS**: macOS 10.13+
- **Linux**: Ubuntu 20.04+ or equivalent
- **DAW**: Any DAW that supports VST3 (FL Studio, Ableton, Reaper, Logic, etc.)

### Installation

1. Download the latest release
2. Copy `Neural Vox Modeler.vst3` to your system plugins folder:
   - **Windows**: `C:\Program Files\Common Files\VST3\`
   - **macOS**: `/Library/Audio/Plug-Ins/VST3/`
   - **Linux**: `~/.vst3/`
3. Rescan plugins in your DAW
4. Done!

---

## How to Use

### 1. Loading the Plugin

In your DAW:
1. Insert Neural Vox Modeler on a guitar track
2. The plugin opens showing "No Model Loaded" in orange

### 2. Loading a Model

You have **two ways** to load a model:

**Method 1: Drag & Drop** (Easiest!)
- Simply drag a `.json` model file onto the plugin window
- The file name turns **green** when loaded successfully

**Method 2: File Browser**
- Click the blue **"LOAD MODEL"** button
- Navigate to your model file (`.json`)
- Click Open

### 3. Plugin Controls

Once a model is loaded, you'll see:

**MODEL NAME** (in green) - Shows which amp/pedal is loaded

**Model Info** - Shows technical details (layers, channels, latency)

**INPUT GAIN** - Adjust how hot the signal hits the model (-12dB to +12dB)
- Turn up for more drive/saturation
- Turn down if clipping

**OUTPUT GAIN** - Control the output volume (-12dB to +12dB)
- Match the volume with your clean signal
- Compensate for quieter amps

**MIX** - Blend between dry (clean) and wet (processed) signal (0% to 100%)
- 100% = Full amp tone (typical use)
- 50% = Half clean, half processed
- 0% = Completely clean (bypass effect)

**BYPASS** - Toggle processing on/off
- Red when bypassed
- Black when active

---

## Understanding Model Files

### What are .json Model Files?

Model files (`.json`) contain the trained neural network that replicates a specific amp or pedal. Each file represents one amp tone.

**Example model names:**
- `vox_ac30_clean.json` - Vox AC30 clean channel
- `ts9_pushed.json` - Tube Screamer pedal driven hard
- `mesa_high_gain.json` - Mesa Boogie high gain tone

### Where to Get Models

**Option 1: Download Pre-trained Models**
- Check the releases page for community models
- Download and drag into the plugin

**Option 2: Train Your Own**
- Record your own amp with the training tools
- See `PLUGIN_GUIDE.md` for how to train models

---

## Common Scenarios

### Using in FL Studio

1. **Insert on Mixer Track:**
   - Right-click mixer track → Insert → Neural Vox Modeler

2. **Load Your Tone:**
   - Drag your `.json` file onto the plugin

3. **Dial In:**
   - Set Input Gain to match your guitar level
   - Leave Mix at 100% for full amp tone
   - Adjust Output Gain to match volume

### Using in Ableton Live

1. **Insert on Audio Track:**
   - Drag plugin from Browser to track

2. **Load Model:**
   - Click LOAD MODEL button, select `.json` file

3. **Enable Delay Compensation:**
   - Options → Delay Compensation → On
   - This keeps your guitar in time with the track

### Recording Guitar

**Recommended signal chain:**
```
Guitar → Audio Interface → DAW Track → Neural Vox Modeler → Output
```

**Tips:**
- Set input buffer to 128-256 samples for low latency
- Use ASIO drivers on Windows for best performance
- Enable delay compensation in your DAW

---

## Model Status Indicators

The plugin shows the current status with color-coded text:

🟢 **Green "Model Name"** = Model loaded successfully, ready to rock!

🟠 **Orange "No Model Loaded"** = No model loaded yet, click Load Model or drag & drop

🔴 **Red Error Dialog** = Model failed to load, possible reasons:
   - Not a valid model file
   - Model is not causal (can't be used in real-time)
   - File is corrupted

---

## Switching Between Models

**Want to try a different amp?** Super easy!

1. Just drag & drop a different `.json` file onto the plugin
2. The new model loads instantly
3. No need to close or reopen anything!

**Pro tip:** Keep all your model files in one folder for quick access.

---

## Performance Tips

### Reducing Latency

**Low latency is crucial for playing live!**

1. **Lower buffer size in your DAW** (128-256 samples)
2. **Use ASIO drivers** (Windows) or CoreAudio (Mac)
3. **Close other programs** to free up CPU
4. **Use models with fewer layers** (check model info - 5-6 layers = lowest latency)

### If Audio is Crackling

1. **Increase buffer size** to 256 or 512 samples
2. **Close background apps** (browsers, Discord, etc.)
3. **Try a different model** with fewer layers

---

## Troubleshooting

### "No Model Loaded" Won't Go Away

**Problem:** You dropped a file but it stays orange

**Solutions:**
- Make sure the file is a `.json` model file
- Check the file isn't corrupted (try re-downloading)
- Look for error dialogs that might explain why

### Model Sounds Wrong or Distorted

**Problem:** Tone doesn't sound right

**Solutions:**
- **Too quiet?** Increase Input Gain
- **Too loud/distorted?** Decrease Input Gain
- **Too processed?** Lower Mix control
- **Wrong model?** Load a different model

### Latency is Too High (Delay When Playing)

**Problem:** You hear a delay between playing and hearing sound

**Solutions:**
1. Lower buffer size in DAW settings (try 128 or 256 samples)
2. Enable ASIO drivers (Windows) in DAW audio settings
3. Check model latency in plugin (shown under model name)
4. Use models with 5-6 layers (lowest latency)

### Plugin Not Showing in DAW

**Problem:** Can't find Neural Vox Modeler in plugin list

**Solutions:**
1. Check you copied to the right folder (see Installation above)
2. Rescan plugins in your DAW (usually in preferences/settings)
3. Make sure your DAW supports VST3

---

## Workflow Tips

### Organizing Your Models

Create a dedicated folder for your amp models:
```
C:\Users\YourName\Documents\Amp Models\
├── Clean\
│   ├── vox_ac30_clean.json
│   └── fender_twin_clean.json
├── Crunch\
│   ├── marshall_plexi.json
│   └── mesa_crunch.json
└── High Gain\
    ├── mesa_high_gain.json
    └── 5150_lead.json
```

### A/B Testing Amps

Want to compare different amps on the same riff?

1. Record your guitar with the plugin
2. Drag & drop different models to hear how each sounds
3. The plugin remembers which model you used when you save the project!

### Matching Levels

When switching between models:
1. Play a test note
2. Adjust Output Gain so all models have similar volume
3. This makes A/B comparisons fair

---

## Advanced Tips

### Using Mix Control Creatively

- **80-90% Mix**: Blend a bit of clean signal for clarity
- **50% Mix**: "Reamp" effect - half processed, half raw
- **30% Mix**: Add a hint of amp coloration to clean tone

### Stacking with Other Effects

**Recommended order:**
```
Guitar → [Compressor] → [Drive Pedals] → Neural Vox Modeler → [Time Effects] → Output
```

- Put drive/distortion **before** Neural Vox Modeler
- Put reverb/delay **after** Neural Vox Modeler

### Per-Project Model Memory

The plugin remembers which model you used in each project!

- Open a song → plugin auto-loads the model you used
- No need to re-load manually each time
- Makes collaboration easy (share model files with bandmates)

---

## Frequently Asked Questions

**Q: Can I use this live?**
A: Yes! The latency is very low (3-5ms). Use a low buffer size (128 samples) in your interface for best results.

**Q: Does it work with bass?**
A: Yes! Any model trained with bass will work. The plugin is mono, perfect for bass.

**Q: Can I run multiple instances?**
A: Yes! Each instance can load a different model. Perfect for dual guitar tones.

**Q: How much CPU does it use?**
A: Very little! Usually 1-2% CPU per instance. You can run many instances without issues.

**Q: Where can I get more models?**
A: Check the releases page, community forums, or train your own! See PLUGIN_GUIDE.md for training instructions.

**Q: Can I use this on vocals/synths/drums?**
A: You can, but it's designed for guitar/bass. Results may vary with other instruments.

**Q: Is this better than cab sims?**
A: Different! This models the entire amp (preamp + power amp + cab). It's learning the complete tone.

**Q: Does it replace my physical amp?**
A: That's up to you! Many players use this for recording and switch to real amps for live gigs. Some go full digital.

---

## Getting Help

- **Technical Issues**: See PLUGIN_GUIDE.md
- **Training Your Own Models**: See PLUGIN_GUIDE.md and README_REALTIME.md
- **Community**: Check the GitHub discussions

---

## Quick Reference

| Control | Range | Purpose |
|---------|-------|---------|
| **Load Model** | Button | Open file browser to load model |
| **Drag & Drop** | Anywhere on plugin | Quick model loading |
| **Input Gain** | -12 to +12 dB | Control signal level into model |
| **Output Gain** | -12 to +12 dB | Control output volume |
| **Mix** | 0% to 100% | Blend dry/wet signals |
| **Bypass** | On/Off | Disable processing |

---

**Rock on! 🎸🔥**

For developers and advanced users, see:
- `PLUGIN_GUIDE.md` - Building and training
- `README_REALTIME.md` - Real-time processing details
- `CLAUDE.md` - Complete technical documentation
