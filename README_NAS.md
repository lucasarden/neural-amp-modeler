# Neural Architecture Search for Neural Vox Modeler

This document describes the multi-model training system that automatically explores different model architectures to find optimal trade-offs between **audio quality**, **latency**, and **CPU efficiency** for real-time guitar amp modeling.

## Overview

The Neural Architecture Search (NAS) system trains multiple model configurations in parallel, evaluates their performance, and identifies the **Pareto frontier** - the set of models that represent optimal trade-offs where no other model is strictly better in all objectives.

### Key Features

- **Automated architecture exploration**: Trains 10-84 model variants automatically
- **Two-phase training**: Quick scan eliminates poor architectures early, then full training on promising models
- **Pre-computed latency**: Calculates latency without training (instant filtering)
- **CPU benchmarking**: Measures real-world performance on CPU
- **Pareto frontier analysis**: Identifies objectively best models
- **Auto-export**: All real-time models export to JSON for VST plugin use
- **Rich visualizations**: Scatter plots, 3D plots, and summary reports

## Quick Start

### 1. Run the complete search (Proof of Concept - 12 models)

```powershell
# Train 12 models with two-phase approach (~12-18 hours)
python src/multi_train.py --search-config configs/search_config.yaml
```

This will:
- **Phase 1 (Quick Scan)**: Train all 12 configs for 50 epochs (~2-3 hours)
- **Filter**: Keep top 50% by validation loss
- **Phase 2 (Full Training)**: Train filtered models to convergence (~10-15 hours)
- **Benchmark**: Measure CPU performance for each model
- **Export**: Save results to `results/search/`

### 2. Analyze results

```powershell
# Find Pareto frontier and generate visualizations
python src/analyze_results.py results/search/all_results.csv
```

This creates:
- `pareto_frontier.csv` - Optimal models only
- `summary.md` - Comprehensive report with recommendations
- `*.png` - Visualization plots

### 3. Select your model

Review `results/search/summary.md` for recommended models:
- **Best Quality**: Lowest validation loss (best audio)
- **Best Latency**: Fastest model meeting quality threshold
- **Most Efficient**: Best quality/parameters ratio
- **Best Balance**: Pareto-optimal, closest to ideal

## Architecture

### New Files

```
configs/
  search_config.yaml           # Search space and training configuration

src/
  search_space.py              # Configuration generation and latency calculation
  multi_train.py               # Main orchestration script
  benchmark.py                 # CPU performance benchmarking
  analyze_results.py           # Pareto frontier analysis and visualization

results/search/
  phase1_results.csv           # Quick scan results
  phase2_results.csv           # Full training results
  all_results.csv              # Combined results
  pareto_frontier.csv          # Optimal models only
  summary.md                   # Comprehensive report
  *.png                        # Visualization plots
```

## Search Space Configuration

### Current Configuration (Proof of Concept)

**`configs/search_config.yaml`** defines:

```yaml
search_space:
  num_layers: [5, 6, 7, 8]      # 4 options
  channels: [16, 24, 32]         # 3 options
  kernel_size: [3]               # 1 option
  # Total: 4 × 3 × 1 = 12 configurations

constraints:
  max_latency_ms: 25.0           # Remove configs >25ms
  max_params: 150000             # Remove very large models
  min_params: 5000               # Remove very small models
```

**Latency range**: 1.4ms - 11.6ms (all under 25ms target)
**Parameter range**: 5.5K - 25K parameters

### Expanding the Search Space

To explore more configurations, edit `search_config.yaml`:

```yaml
search_space:
  num_layers: [4, 5, 6, 7, 8, 9, 10]  # 7 options
  channels: [12, 16, 20, 24, 32, 48]   # 6 options
  kernel_size: [3, 5]                  # 2 options
  # Total: 7 × 6 × 2 = 84 configurations
```

## Training Phases

### Phase 1: Quick Scan

**Goal**: Eliminate poor architectures early

- **Duration**: 50 epochs per model
- **Early stopping**: Patience = 10 epochs
- **Purpose**: Quick evaluation to filter out bad configs
- **Keeps**: Top 50% by validation loss

### Phase 2: Full Training

**Goal**: Train promising models to convergence

- **Duration**: Up to 2000 epochs per model
- **Early stopping**: Patience = 100 epochs
- **Purpose**: Get best possible performance from good architectures
- **Benchmarks**: CPU performance after training
- **Auto-export**: JSON for VST plugin

## Running Different Phases

### Phase 1 only (quick scan)

```powershell
python src/multi_train.py --search-config configs/search_config.yaml --phase phase1
```

### Phase 2 only (full training on pre-filtered configs)

```powershell
python src/multi_train.py --search-config configs/search_config.yaml --phase phase2 --configs-dir configs/search/filtered
```

### Both phases (default)

```powershell
python src/multi_train.py --search-config configs/search_config.yaml --phase both
```

## Understanding the Results

### Metrics Tracked

For each model, we track:

1. **Quality metrics**:
   - `best_val_loss`: Validation loss (ESR) - lower is better
   - `best_train_loss`: Training loss
   - `best_epoch`: Epoch where best validation loss occurred

2. **Latency metrics**:
   - `latency_ms`: Model latency in milliseconds
   - `receptive_field`: Receptive field in samples

3. **Efficiency metrics**:
   - `parameters`: Total trainable parameters
   - `cpu_mean_ms`: Mean CPU processing time per chunk
   - `cpu_rtf`: Real-time factor (processing_time / audio_duration)
   - `cpu_realtime_capable`: Boolean (True if RTF < 0.8)

4. **Architecture**:
   - `num_layers`: Number of residual blocks
   - `channels`: Internal feature dimension
   - `kernel_size`: Convolution kernel size
   - `causal`: Always True (real-time mode)

### Pareto Frontier

A model is **Pareto-optimal** if no other model is better in ALL objectives simultaneously.

**Example**: Consider 3 models:
- Model A: Loss=0.004, Latency=5ms, Params=10K
- Model B: Loss=0.003, Latency=12ms, Params=20K (Better quality, worse latency/efficiency)
- Model C: Loss=0.005, Latency=3ms, Params=8K (Better latency/efficiency, worse quality)

Models A, B, and C are all Pareto-optimal because each has a trade-off. Model A is balanced, B prioritizes quality, C prioritizes speed.

### Visualizations

The analysis generates several plots:

1. **`quality_vs_latency.png`**: Most important! Shows audio quality vs latency trade-off
2. **`quality_vs_params.png`**: Quality vs model size
3. **`latency_vs_params.png`**: Speed vs size trade-off
4. **`quality_vs_cpu_rtf.png`**: Quality vs CPU performance (if benchmarked)
5. **`pareto_3d.png`**: 3D view of all three objectives

Red stars indicate Pareto-optimal models.

## Customization

### Adjusting Training Hyperparameters

Edit `configs/search_config.yaml`:

```yaml
phase1:
  num_epochs: 50              # Quick scan duration
  patience: 10                # Early stopping patience
  batch_size: 48
  learning_rate: 0.003

phase2:
  num_epochs: 2000            # Full training duration
  patience: 100               # Early stopping patience
  batch_size: 48
  learning_rate: 0.003
```

### Changing Filtering Criteria

```yaml
phase1:
  keep_top_percent: 50        # Keep best 50% after Phase 1

constraints:
  max_latency_ms: 15.0        # Stricter latency constraint
  max_params: 100000          # Smaller model size limit
```

### Using Different Training Data

```yaml
data:
  input_file: "data/raw/your_input.wav"
  output_file: "data/raw/your_output.wav"
  segment_length: 8192
  train_split: 0.85
```

## Time Estimates

Based on current hardware (RTX 2070 SUPER):

### Proof of Concept (12 configs)
- Phase 1: ~2-3 hours (12 models × 12 min)
- Phase 2: ~10-15 hours (6 models × 90 min)
- **Total: ~12-18 hours**

### Focused Search (25 configs)
- Phase 1: ~5 hours (25 models × 12 min)
- Phase 2: ~22.5 hours (12 models × 90 min)
- **Total: ~28 hours**

### Conservative Search (84 configs)
- Phase 1: ~17 hours (84 models × 12 min)
- Phase 2: ~63 hours (42 models × 90 min)
- **Total: ~80 hours** (~3.3 days)

**Tip**: Start overnight with POC (12 configs) to validate approach.

## Advanced Usage

### Generate configs without training

```powershell
# Preview search space
python src/search_space.py --config configs/search_config.yaml --dry-run

# Generate config files for manual inspection
python src/search_space.py --config configs/search_config.yaml --output-dir configs/search
```

### Benchmark an existing model

```powershell
python src/benchmark.py models/your_model/best_model.pt --chunk-size 512 --num-chunks 100
```

### Re-analyze results with different metrics

```powershell
# After editing analyze_results.py for custom metrics
python src/analyze_results.py results/search/all_results.csv --output-dir results/custom_analysis
```

## Latency Calculation Formula

Latency is determined by the receptive field:

```
receptive_field = 1 + sum((kernel_size - 1) * dilation_base^i for i in range(num_layers))
latency_ms = receptive_field / sample_rate * 1000
```

**Examples @ 44.1kHz**:
- 5 layers, k=3, d=2: 63 samples = 1.43ms
- 6 layers, k=3, d=2: 127 samples = 2.88ms
- 7 layers, k=3, d=2: 255 samples = 5.78ms
- 8 layers, k=3, d=2: 511 samples = 11.6ms
- 9 layers, k=3, d=2: 1023 samples = 23.2ms

## Recommendations

### For VST Plugin Use
- **Target**: Latency <10ms, RTF <0.5x
- **Search space**: layers=[5,6,7], channels=[16,24,32]
- **Priority**: Balance quality and CPU efficiency

### For Batch Processing
- **Target**: Best quality regardless of latency
- **Search space**: layers=[8,9,10], channels=[32,48,64]
- **Priority**: Minimize validation loss

### For Mobile/Low-Power
- **Target**: Minimal parameters, RTF <0.3x
- **Search space**: layers=[4,5,6], channels=[12,16,20]
- **Priority**: Efficiency (params and CPU)

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size` in `search_config.yaml`
- Train fewer models in parallel (currently sequential)

### Training too slow
- Start with POC (12 models) first
- Reduce `num_epochs` in Phase 1
- Increase `patience` to allow early stopping

### All models have high loss
- Check training data quality (`src/inspect_audio.py`)
- Verify data is normalized (`src/normalize_training_data.py`)
- Review TensorBoard logs in `runs/search/`

### Unicode errors (Windows)
- Already fixed! Using `[OK]` and `[X]` instead of unicode symbols
- If issues persist, ensure terminal supports UTF-8

## What's Next?

After completing the NAS:

1. **Select your model**: Review `results/search/summary.md` recommendations
2. **Test the model**: `python src/inference.py input.wav --model models/selected_model/best_model.pt`
3. **Real-time demo**: `python src/realtime_demo.py --model models/selected_model/best_model.pt`
4. **Load in VST**: Drag & drop `.json` file into Neural Vox Modeler plugin
5. **Iterate**: Adjust search space based on results and re-run

## Scientific Background

This approach is grounded in:

- **Neural Architecture Search (NAS)**: Automated ML model design
- **Multi-objective optimization**: Balancing competing objectives
- **Pareto efficiency**: Mathematical framework for optimal trade-offs
- **WaveNet architecture**: Proven for audio generation and modeling

## References

- **WaveNet**: van den Oord et al. (2016) - "WaveNet: A Generative Model for Raw Audio"
- **ENAS**: Pham et al. (2018) - "Efficient Neural Architecture Search via Parameter Sharing"
- **Multi-Objective NAS**: Lu et al. (2019) - "Multi-Objective Neural Architecture Search"
- **Neural Amp Modeler**: Steven Atkinson - https://github.com/sdatkinson/neural-amp-modeler

---

**Happy model searching!** 🎸🔍

For questions or issues, consult the main README or check the TensorBoard logs in `runs/search/`.
