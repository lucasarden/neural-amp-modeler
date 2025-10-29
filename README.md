# Neural Amp Modeler - Pushed TS9

Deep learning-based amplifier modeling using WaveNet architecture.

## Project Status

**In Progress** - Currently in data collection phase

## Setup (Windows 10)

1. Virtual environment already set up ✓

2. Verify CUDA installation:

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

3. Test model:

```powershell
python src\model.py
```

## Workflow

### Phase 1: Data Collection

- [ ] Record clean DI guitar signal
- [ ] Record AC30 output (SM57 mic'd)
- [ ] Place files in `data\raw\`

### Phase 2: Training

- [ ] Preprocess audio data
- [ ] Train model
- [ ] Evaluate results

### Phase 3: Inference

- [ ] Implement real-time processing
- [ ] Create standalone application

## Project Structure

```
neural-amp-modeler/
├── data/                  # Audio recordings
├── src/                   # Source code
├── models/
│   └── ts9_pushed/         # Model files
│       ├── best_model.pt # Best model (saved in git)
│       └── checkpoints/  # Training checkpoints (not in git)
├── configs/              # Configuration files
└── notebooks/            # Experiments
```

## Model Management

- Best models are saved to: `models/ts9_pushed/best_model.pt`
- Training checkpoints: `models/ts9_pushed/checkpoints/`
- Only best_model.pt is tracked in git (checkpoints are ignored)

## Resources

- [NAM Project](https://github.com/sdatkinson/neural-amp-modeler)
- [WaveNet Paper](https://arxiv.org/abs/1609.03499)
