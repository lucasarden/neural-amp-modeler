# Real-Time Processing Analysis

## Current Status: NOT REAL-TIME READY

Your model can process audio files offline, but **cannot run in real-time** as currently implemented. Here's why and what needs to change.

---

## How to Test Your Model (Offline)

### Basic Usage
```powershell
# Process a guitar recording
python src/inference.py path/to/clean_guitar.wav

# Specify output file
python src/inference.py input.wav -o output.wav

# Process on CPU
python src/inference.py input.wav -d cpu

# Process large files in chunks
python src/inference.py long_file.wav -c 441000
```

The model successfully processes audio files and outputs the amp-modeled result!

---

## Why It's NOT Real-Time Ready

### 1. **Non-Causal Architecture** (CRITICAL ISSUE)
**Problem**: Your model uses `padding="same"` in convolutions (see `src/model.py:16`)

```python
self.conv = nn.Conv1d(channels, channels, kernel_size, padding="same", dilation=dilation)
```

- `padding="same"` **centers** the convolution kernel
- This means the model looks at **future samples** to process current samples
- Real-time = you can't see the future!

**Example**: With kernel_size=3, padding="same" looks at [past, present, future]
- Real-time needs: [past, past, present] only

### 2. **Batch Processing Architecture**
**Current**: Processes entire audio file at once
```python
output = model(entire_audio_tensor)  # Process all 186 seconds at once
```

**Real-time needs**: Process small chunks continuously (e.g., 128-512 samples at a time)

### 3. **No State Management**
Real-time streaming requires maintaining context between chunks:
- Your receptive field is **2047 samples** (~46ms at 44.1kHz)
- Each chunk needs context from previous chunks
- Current implementation has no mechanism for this

---

## Performance Analysis

### Current Model Specs
- **Parameters**: 7,889 (very lightweight!)
- **Receptive field**: 2047 samples (46.4ms latency)
- **Processing speed**: Processed 186 seconds in ~1-2 seconds on RTX 2070 SUPER

### Real-Time Requirements (44.1kHz)
- Must process 44,100 samples per second
- Typical audio buffer: 128-512 samples (2.9-11.6ms chunks)
- Must process each chunk in < chunk duration

**Good news**: Your model is small enough (7,889 params) that real-time is likely achievable with proper architecture!

---

## What Changes Are Needed for Real-Time

### 1. **Make Architecture Causal**

**Change in `src/model.py`**:

```python
# CURRENT (non-causal)
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding="same",  # ❌ This looks into the future
            dilation=dilation
        )
```

**NEEDED (causal)**:

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        # Calculate left-only padding for causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=0,  # We'll pad manually
            dilation=dilation
        )

    def forward(self, x):
        # Pad only on the left (past samples only)
        x = F.pad(x, (self.padding, 0))
        residual = x[:, :, self.padding:]  # Align for residual
        out = self.conv(x)
        out = self.activation(out)
        return out + residual
```

**Impact**: You'll need to **retrain** the model with causal architecture.

### 2. **Implement Streaming Inference**

Create a new class for real-time processing:

```python
class StreamingWaveNetAmp:
    def __init__(self, model, receptive_field):
        self.model = model
        self.receptive_field = receptive_field
        self.buffer = torch.zeros(1, 1, receptive_field)  # Context buffer

    def process_chunk(self, chunk):
        """Process a small chunk with context from buffer"""
        # Concatenate buffer with new chunk
        input_with_context = torch.cat([self.buffer, chunk], dim=-1)

        # Process
        output = self.model(input_with_context)

        # Update buffer (keep last receptive_field samples)
        self.buffer = input_with_context[:, :, -self.receptive_field:]

        # Return only the new samples (not buffered context)
        return output[:, :, -chunk.shape[-1]:]

    def reset(self):
        """Clear buffer between sessions"""
        self.buffer.zero_()
```

### 3. **Optimize for Low Latency**

**Current latency**: ~46ms (receptive field of 2047 samples)

Options to reduce latency:
- **Reduce layers**: 10 → 6-8 layers (reduces receptive field)
- **Smaller dilation base**: 2 → 1 or custom pattern
- **Trade-off**: Less context = may reduce quality

Example smaller architecture:
```yaml
# configs/config_realtime.yaml
model:
  channels: 16
  num_layers: 6      # Reduced from 10
  kernel_size: 3
  dilation_base: 2
  # Receptive field: ~191 samples (~4.3ms latency)
```

### 4. **Create Plugin Interface**

For actual guitar use, you need an audio plugin (VST/AU/LV2):

**Option A**: Use existing Python plugin frameworks
- [DPF-Python](https://github.com/falkTX/DPF-Plugins) - DISTRHO Plugin Framework
- [neutone](https://github.com/QosmoInc/neutone_sdk) - AI audio plugin SDK

**Option B**: Export to ONNX and use C++ plugin
```python
# Export trained model
torch.onnx.export(model, dummy_input, "ts9_model.onnx")
```
Then load in C++ VST wrapper for minimal latency

**Option C**: Use existing NAM ecosystem
- Export to [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) format
- Use their real-time plugin infrastructure

---

## Recommended Path Forward

### For Testing (Current Model)
✅ **You can use it now** for offline processing:
```powershell
python src/inference.py my_guitar.wav -o processed.wav
```

### For Real-Time (Requires Changes)

**Phase 1: Causal Architecture** (Essential)
1. Modify `ResidualBlock` to use causal padding
2. Retrain the model from scratch
3. Verify offline that results are similar quality

**Phase 2: Streaming Infrastructure** (Required)
1. Implement `StreamingWaveNetAmp` class
2. Benchmark processing time per chunk
3. Optimize if needed (reduce layers/channels)

**Phase 3: Plugin Development** (For guitar use)
1. Choose plugin framework (recommend neutone or ONNX export)
2. Implement audio callback with streaming model
3. Test latency and CPU usage

---

## Quick Latency Calculation

```python
# Receptive field determines minimum latency
receptive_field_samples = (kernel_size - 1) * sum(dilation_base^i for i in range(num_layers))

# Current config
# kernel=3, dilation_base=2, layers=10
# receptive_field = 2047 samples
# latency = 2047 / 44100 = 46.4ms

# Smaller config (for low latency)
# kernel=3, dilation_base=2, layers=6
# receptive_field = 191 samples
# latency = 191 / 44100 = 4.3ms ✅ Good for real-time!
```

---

## Summary

| Aspect | Current | Real-Time Ready |
|--------|---------|-----------------|
| Architecture | Non-causal ❌ | Causal ✅ |
| Processing | Full file | Streaming chunks |
| State | Stateless | Buffered context |
| Latency | N/A (offline) | <10ms target |
| Interface | Python script | VST/AU plugin |

**Bottom Line**:
- ✅ Your model works great for offline processing right now!
- ❌ Real-time requires architectural changes and retraining
- ✅ But it's totally achievable - the model is small and fast enough!

The biggest change needed is making the convolutions causal and retraining.
