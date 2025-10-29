"""
Batch export all trained models to JSON format for VST plugin

Scans the models/ directory and exports all trained models to JSON format
that can be loaded by the C++ VST plugin.
"""

import os
import yaml
from pathlib import Path
from export_plugin_weights import export_for_plugin


def find_config_for_model(model_name, configs_dir="configs"):
    """
    Try to find the appropriate config file for a model

    Args:
        model_name: Name of the model directory
        configs_dir: Directory containing config files

    Returns:
        Path to config file or None if not found
    """
    # Try exact match first
    exact_match = Path(configs_dir) / f"config_{model_name}.yaml"
    if exact_match.exists():
        return str(exact_match)

    # Try to match based on keywords in model name
    config_files = list(Path(configs_dir).glob("*.yaml"))

    for config_file in config_files:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            # Check if this config's model name matches
            if config.get("model", {}).get("name") == model_name:
                return str(config_file)

    # Check if model is causal by looking for "realtime" or "causal" in name
    if "realtime" in model_name.lower() or "causal" in model_name.lower():
        realtime_config = Path(configs_dir) / "config_realtime.yaml"
        if realtime_config.exists():
            return str(realtime_config)

    return None


def batch_export_models(models_dir="models", configs_dir="configs", force_config=None):
    """
    Export all trained models to JSON format

    Args:
        models_dir: Directory containing model subdirectories
        configs_dir: Directory containing config files
        force_config: Optional config file to use for all models
    """
    print("=" * 80)
    print("BATCH EXPORT MODELS FOR VST PLUGIN")
    print("=" * 80)
    print()

    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Error: Models directory not found: {models_dir}")
        return

    # Find all model directories
    model_dirs = [d for d in models_path.iterdir() if d.is_dir()]

    if not model_dirs:
        print(f"❌ No model directories found in {models_dir}")
        return

    print(f"Found {len(model_dirs)} model directories:")
    for d in model_dirs:
        print(f"  • {d.name}")
    print()

    # Export each model
    results = {
        "success": [],
        "skipped": [],
        "failed": []
    }

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_file = model_dir / "best_model.pt"

        print("-" * 80)
        print(f"Processing: {model_name}")
        print("-" * 80)

        # Check if model file exists
        if not model_file.exists():
            print(f"  ⚠️  Skipped: No best_model.pt found")
            results["skipped"].append((model_name, "No best_model.pt"))
            print()
            continue

        # Find config
        if force_config:
            config_path = force_config
            print(f"  Using forced config: {config_path}")
        else:
            config_path = find_config_for_model(model_name, configs_dir)
            if config_path:
                print(f"  Found config: {config_path}")
            else:
                print(f"  ⚠️  Skipped: Could not find matching config")
                print(f"     Run with --config to specify config manually")
                results["skipped"].append((model_name, "No matching config"))
                print()
                continue

        # Check if config is causal
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config.get("model", {}).get("causal", False):
            print(f"  ⚠️  Skipped: Model is not causal (cannot use for real-time VST)")
            print(f"     Only causal models can be exported for VST plugins")
            results["skipped"].append((model_name, "Not causal"))
            print()
            continue

        # Export
        output_path = model_dir / f"{model_name}.json"

        try:
            success = export_for_plugin(
                model_path=str(model_file),
                config_path=config_path,
                output_path=str(output_path)
            )

            if success:
                results["success"].append((model_name, str(output_path)))
                print(f"  ✅ Exported: {output_path}")
            else:
                results["failed"].append((model_name, "Export function returned False"))
                print(f"  ❌ Failed to export")

        except Exception as e:
            results["failed"].append((model_name, str(e)))
            print(f"  ❌ Export failed: {e}")

        print()

    # Print summary
    print("=" * 80)
    print("BATCH EXPORT SUMMARY")
    print("=" * 80)
    print()

    print(f"✅ Successfully exported: {len(results['success'])}")
    for model_name, output_path in results["success"]:
        print(f"   • {model_name} → {output_path}")
    print()

    if results["skipped"]:
        print(f"⚠️  Skipped: {len(results['skipped'])}")
        for model_name, reason in results["skipped"]:
            print(f"   • {model_name}: {reason}")
        print()

    if results["failed"]:
        print(f"❌ Failed: {len(results['failed'])}")
        for model_name, error in results["failed"]:
            print(f"   • {model_name}: {error}")
        print()

    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Copy exported .json files to any location")
    print("2. Build the C++ VST plugin (see plugin/README.md)")
    print("3. Install VST3 to your DAW's plugin folder")
    print("4. In the DAW, use 'Load Model' or drag & drop to load any .json file")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch export all trained models to JSON format for VST plugin"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing model subdirectories (default: models)"
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Directory containing config files (default: configs)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Force use of this config file for all models"
    )

    args = parser.parse_args()

    batch_export_models(
        models_dir=args.models_dir,
        configs_dir=args.configs_dir,
        force_config=args.config
    )


if __name__ == "__main__":
    main()
