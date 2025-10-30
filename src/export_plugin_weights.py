"""
Export trained model weights for C++ VST plugin

Exports weights in JSON format that can be loaded by the JUCE plugin.
"""

import torch
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from model import WaveNetAmp


def export_for_plugin(model_path, config_path, output_path):
    """Export model weights and architecture for C++ plugin

    Args:
        model_path: Path to trained .pt model
        config_path: Path to config.yaml
        output_path: Output path for JSON file
    """
    print("=" * 60)
    print("Exporting Model for C++ VST Plugin")
    print("=" * 60)

    # Load config
    print(f"\n[1/5] Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check if causal
    if not config["model"].get("causal", False):
        print("[X] ERROR: Model must be causal for real-time VST plugin")
        print("   Use config_realtime.yaml or set causal: true and retrain")
        return False

    # Load model
    print(f"[2/5] Loading trained model from: {model_path}")
    model = WaveNetAmp(
        channels=config["model"]["channels"],
        num_layers=config["model"]["num_layers"],
        kernel_size=config["model"]["kernel_size"],
        dilation_base=config["model"]["dilation_base"],
        causal=True,
    )

    # Load checkpoint - handle both old and new formats
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format from multi_train.py (with metadata)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Old format (direct state_dict)
        model.load_state_dict(checkpoint)
    model.eval()

    receptive_field = model.receptive_field()
    latency_ms = receptive_field / config["data"]["sample_rate"] * 1000

    print(f"   [OK] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   [OK] Receptive field: {receptive_field} samples")
    print(f"   [OK] Latency: {latency_ms:.2f}ms @ {config['data']['sample_rate']}Hz")

    # Extract weights
    print(f"[3/5] Extracting weights and biases...")
    state_dict = model.state_dict()

    # Build JSON structure
    model_json = {
        "model_info": {
            "name": config["model"]["name"],
            "version": "1.0.0",
            "architecture": "WaveNet",
            "causal": True,
            "sample_rate": config["data"]["sample_rate"],
            "receptive_field": receptive_field,
            "latency_ms": round(latency_ms, 2),
        },
        "architecture": {
            "channels": config["model"]["channels"],
            "num_layers": config["model"]["num_layers"],
            "kernel_size": config["model"]["kernel_size"],
            "dilation_base": config["model"]["dilation_base"],
            "dilations": [
                config["model"]["dilation_base"] ** i
                for i in range(config["model"]["num_layers"])
            ],
        },
        "weights": {},
    }

    # Input layer
    model_json["weights"]["input_conv"] = {
        "weight": state_dict["input_conv.weight"].squeeze().tolist(),
        "bias": state_dict["input_conv.bias"].tolist(),
    }

    # Residual blocks
    model_json["weights"]["residual_blocks"] = []
    for i in range(config["model"]["num_layers"]):
        block_weights = {
            "layer": i,
            "dilation": config["model"]["dilation_base"] ** i,
            "conv_weight": state_dict[f"residual_blocks.{i}.conv.weight"]
            .numpy()
            .tolist(),
            "conv_bias": state_dict[f"residual_blocks.{i}.conv.bias"].tolist(),
        }
        model_json["weights"]["residual_blocks"].append(block_weights)

    # Output layer
    model_json["weights"]["output_conv"] = {
        "weight": state_dict["output_conv.weight"].squeeze().tolist(),
        "bias": state_dict["output_conv.bias"].tolist(),
    }

    # Save JSON
    print(f"[4/5] Saving to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(model_json, f, indent=2)

    # Calculate file size
    file_size_kb = output_path.stat().st_size / 1024

    # Verify by loading back
    print(f"[5/5] Verifying export...")
    with open(output_path, "r") as f:
        loaded = json.load(f)

    assert loaded["architecture"]["num_layers"] == config["model"]["num_layers"]
    assert len(loaded["weights"]["residual_blocks"]) == config["model"]["num_layers"]

    print("\n" + "=" * 60)
    print("[OK] Export Complete!")
    print("=" * 60)
    print(f"\nOutput file: {output_path}")
    print(f"File size: {file_size_kb:.1f} KB")
    print(f"Latency: {latency_ms:.2f}ms")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Copy the JSON file to plugin/Resources/")
    print("2. Build the C++ plugin (see plugin/README.md)")
    print("3. Install VST3 to FL Studio")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export model weights for C++ VST plugin"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pt file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_realtime.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output", type=str, help="Output path for JSON file (default: auto)"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        model_path = Path(args.model)
        output_path = model_path.parent / f"{model_path.parent.name}_weights.json"
    else:
        output_path = Path(args.output)

    success = export_for_plugin(args.model, args.config, output_path)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
