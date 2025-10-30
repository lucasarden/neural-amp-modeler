"""
Analyze multi-model training results

Performs Pareto frontier analysis and generates visualizations to help
select optimal models balancing quality, latency, and CPU efficiency.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import seaborn as sns


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results from CSV file

    Args:
        csv_path: Path to results CSV

    Returns:
        Pandas DataFrame with results
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    return df


def find_pareto_frontier(df: pd.DataFrame,
                        objectives: List[str] = ['best_val_loss', 'latency_ms', 'parameters'],
                        minimize: List[bool] = [True, True, True]) -> pd.DataFrame:
    """Find Pareto-optimal models (non-dominated solutions)

    A model is Pareto-optimal if no other model is better in ALL objectives.

    Args:
        df: DataFrame with results
        objectives: List of objective column names
        minimize: List of booleans indicating whether to minimize each objective

    Returns:
        DataFrame with only Pareto-optimal models
    """
    print(f"\nFinding Pareto frontier for objectives: {objectives}")

    # Convert to numpy array for efficient computation
    values = df[objectives].values
    n = len(values)

    # Flip sign for maximization objectives
    for i, (obj, mini) in enumerate(zip(objectives, minimize)):
        if not mini:
            values[:, i] = -values[:, i]

    # Find non-dominated solutions
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if is_pareto[i]:
            # Check if any other point dominates this one
            # A point j dominates i if it's better or equal in all objectives
            # and strictly better in at least one
            dominates = np.all(values[i] >= values, axis=1) & np.any(values[i] > values, axis=1)
            is_pareto[dominates] = False

    pareto_df = df[is_pareto].copy()
    pareto_df['pareto_optimal'] = True

    print(f"Found {len(pareto_df)} Pareto-optimal models out of {len(df)} total")

    return pareto_df


def plot_2d_scatter(df: pd.DataFrame, pareto_df: pd.DataFrame,
                   x_col: str, y_col: str, output_path: str,
                   x_label: str = None, y_label: str = None,
                   title: str = None):
    """Create 2D scatter plot with Pareto frontier highlighted

    Args:
        df: DataFrame with all results
        pareto_df: DataFrame with Pareto-optimal results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        output_path: Path to save figure
        x_label: Optional custom x-axis label
        y_label: Optional custom y-axis label
        title: Optional custom title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all points
    ax.scatter(df[x_col], df[y_col], alpha=0.5, s=100, label='All models', color='gray')

    # Highlight Pareto frontier
    ax.scatter(pareto_df[x_col], pareto_df[y_col],
              alpha=0.8, s=150, label='Pareto optimal', color='red', marker='*', edgecolors='black')

    # Labels
    ax.set_xlabel(x_label or x_col, fontsize=12)
    ax.set_ylabel(y_label or y_col, fontsize=12)
    ax.set_title(title or f'{y_label or y_col} vs {x_label or x_col}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_3d_scatter(df: pd.DataFrame, pareto_df: pd.DataFrame,
                   output_path: str):
    """Create 3D scatter plot of all three objectives

    Args:
        df: DataFrame with all results
        pareto_df: DataFrame with Pareto-optimal results
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(df['latency_ms'], df['parameters'], df['best_val_loss'],
              alpha=0.5, s=50, label='All models', color='gray')

    # Highlight Pareto frontier
    ax.scatter(pareto_df['latency_ms'], pareto_df['parameters'], pareto_df['best_val_loss'],
              alpha=0.9, s=150, label='Pareto optimal', color='red', marker='*', edgecolors='black')

    # Labels
    ax.set_xlabel('Latency (ms)', fontsize=11)
    ax.set_ylabel('Parameters', fontsize=11)
    ax.set_zlabel('Validation Loss (ESR)', fontsize=11)
    ax.set_title('3D Pareto Frontier: Quality vs Latency vs Complexity', fontsize=13)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D plot: {output_path}")
    plt.close()


def generate_summary_report(df: pd.DataFrame, pareto_df: pd.DataFrame,
                           output_path: str):
    """Generate markdown summary report

    Args:
        df: DataFrame with all results
        pareto_df: DataFrame with Pareto-optimal results
        output_path: Path to save markdown file
    """
    report = []
    report.append("# Neural Architecture Search - Results Summary\n")
    report.append(f"**Total models trained:** {len(df)}\n")
    report.append(f"**Pareto-optimal models:** {len(pareto_df)}\n\n")

    # Statistics
    report.append("## Overall Statistics\n")
    report.append(f"- **Validation Loss:** {df['best_val_loss'].min():.6f} - {df['best_val_loss'].max():.6f}\n")
    report.append(f"- **Latency:** {df['latency_ms'].min():.2f}ms - {df['latency_ms'].max():.2f}ms\n")
    report.append(f"- **Parameters:** {df['parameters'].min():,} - {df['parameters'].max():,}\n")
    if 'cpu_rtf' in df.columns:
        report.append(f"- **CPU Real-time Factor:** {df['cpu_rtf'].min():.3f}x - {df['cpu_rtf'].max():.3f}x\n")
    report.append("\n")

    # Best models by different criteria
    report.append("## Recommended Models\n\n")

    # Best quality (lowest val_loss)
    best_quality = df.loc[df['best_val_loss'].idxmin()]
    report.append("### 1. Best Quality (Lowest Validation Loss)\n")
    report.append(f"- **Model:** {best_quality['model_name']}\n")
    report.append(f"- **Validation Loss:** {best_quality['best_val_loss']:.6f}\n")
    report.append(f"- **Latency:** {best_quality['latency_ms']:.2f}ms\n")
    report.append(f"- **Parameters:** {best_quality['parameters']:,}\n")
    report.append(f"- **Architecture:** L{best_quality['num_layers']} C{best_quality['channels']} K{best_quality['kernel_size']}\n")
    if 'cpu_rtf' in best_quality and pd.notna(best_quality['cpu_rtf']):
        report.append(f"- **CPU RTF:** {best_quality['cpu_rtf']:.3f}x\n")
    report.append("\n")

    # Best latency (meeting quality threshold)
    quality_threshold = df['best_val_loss'].quantile(0.25)  # Top 25% quality
    low_latency_df = df[df['best_val_loss'] <= quality_threshold]
    if not low_latency_df.empty:
        best_latency = low_latency_df.loc[low_latency_df['latency_ms'].idxmin()]
        report.append("### 2. Best Latency (Among Top 25% Quality)\n")
        report.append(f"- **Model:** {best_latency['model_name']}\n")
        report.append(f"- **Validation Loss:** {best_latency['best_val_loss']:.6f}\n")
        report.append(f"- **Latency:** {best_latency['latency_ms']:.2f}ms\n")
        report.append(f"- **Parameters:** {best_latency['parameters']:,}\n")
        report.append(f"- **Architecture:** L{best_latency['num_layers']} C{best_latency['channels']} K{best_latency['kernel_size']}\n")
        if 'cpu_rtf' in best_latency and pd.notna(best_latency['cpu_rtf']):
            report.append(f"- **CPU RTF:** {best_latency['cpu_rtf']:.3f}x\n")
        report.append("\n")

    # Most efficient (best quality/params ratio)
    df['efficiency'] = 1.0 / (df['best_val_loss'] * df['parameters'])
    most_efficient = df.loc[df['efficiency'].idxmax()]
    report.append("### 3. Most Efficient (Best Quality/Parameters Ratio)\n")
    report.append(f"- **Model:** {most_efficient['model_name']}\n")
    report.append(f"- **Validation Loss:** {most_efficient['best_val_loss']:.6f}\n")
    report.append(f"- **Latency:** {most_efficient['latency_ms']:.2f}ms\n")
    report.append(f"- **Parameters:** {most_efficient['parameters']:,}\n")
    report.append(f"- **Architecture:** L{most_efficient['num_layers']} C{most_efficient['channels']} K{most_efficient['kernel_size']}\n")
    if 'cpu_rtf' in most_efficient and pd.notna(most_efficient['cpu_rtf']):
        report.append(f"- **CPU RTF:** {most_efficient['cpu_rtf']:.3f}x\n")
    report.append("\n")

    # Balanced (Pareto optimal with good balance)
    if len(pareto_df) > 0:
        # Normalize objectives and find closest to ideal point
        pareto_norm = pareto_df[['best_val_loss', 'latency_ms', 'parameters']].copy()
        for col in pareto_norm.columns:
            pareto_norm[col] = (pareto_norm[col] - pareto_norm[col].min()) / (pareto_norm[col].max() - pareto_norm[col].min())
        pareto_norm['distance'] = np.sqrt((pareto_norm ** 2).sum(axis=1))
        balanced_idx = pareto_df.iloc[pareto_norm['distance'].idxmin()]

        report.append("### 4. Best Balance (Pareto-optimal, closest to ideal)\n")
        report.append(f"- **Model:** {balanced_idx['model_name']}\n")
        report.append(f"- **Validation Loss:** {balanced_idx['best_val_loss']:.6f}\n")
        report.append(f"- **Latency:** {balanced_idx['latency_ms']:.2f}ms\n")
        report.append(f"- **Parameters:** {balanced_idx['parameters']:,}\n")
        report.append(f"- **Architecture:** L{balanced_idx['num_layers']} C{balanced_idx['channels']} K{balanced_idx['kernel_size']}\n")
        if 'cpu_rtf' in balanced_idx and pd.notna(balanced_idx['cpu_rtf']):
            report.append(f"- **CPU RTF:** {balanced_idx['cpu_rtf']:.3f}x\n")
        report.append("\n")

    # Pareto frontier table
    report.append("## Pareto Frontier Models\n\n")
    report.append("All non-dominated models (optimal trade-offs):\n\n")

    # Create formatted table
    table_cols = ['model_name', 'num_layers', 'channels', 'best_val_loss',
                  'latency_ms', 'parameters']
    if 'cpu_rtf' in pareto_df.columns:
        table_cols.append('cpu_rtf')

    pareto_table = pareto_df[table_cols].sort_values('best_val_loss')

    report.append("| Model | Layers | Channels | Val Loss | Latency (ms) | Params | CPU RTF |\n")
    report.append("|-------|--------|----------|----------|--------------|--------|---------||\n")

    for _, row in pareto_table.iterrows():
        rtf_str = f"{row['cpu_rtf']:.3f}x" if 'cpu_rtf' in row and pd.notna(row['cpu_rtf']) else "N/A"
        report.append(f"| {row['model_name']} | {row['num_layers']} | {row['channels']} | "
                     f"{row['best_val_loss']:.6f} | {row['latency_ms']:.2f} | "
                     f"{row['parameters']:,} | {rtf_str} |\n")

    report.append("\n")

    # All models table
    report.append("## All Models\n\n")
    report.append("Complete results sorted by validation loss:\n\n")

    all_table = df[table_cols].sort_values('best_val_loss')

    report.append("| Model | Layers | Channels | Val Loss | Latency (ms) | Params | CPU RTF |\n")
    report.append("|-------|--------|----------|----------|--------------|--------|---------||\n")

    for _, row in all_table.iterrows():
        rtf_str = f"{row['cpu_rtf']:.3f}x" if 'cpu_rtf' in row and pd.notna(row['cpu_rtf']) else "N/A"
        report.append(f"| {row['model_name']} | {row['num_layers']} | {row['channels']} | "
                     f"{row['best_val_loss']:.6f} | {row['latency_ms']:.2f} | "
                     f"{row['parameters']:,} | {rtf_str} |\n")

    # Write report
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"Saved summary report: {output_path}")


def analyze_results(csv_path: str, output_dir: str = None):
    """Main analysis function

    Args:
        csv_path: Path to results CSV file
        output_dir: Optional directory for outputs (defaults to same dir as CSV)
    """
    print(f"\n{'='*70}")
    print("NEURAL ARCHITECTURE SEARCH - RESULTS ANALYSIS")
    print(f"{'='*70}\n")

    # Load results
    df = load_results(csv_path)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find Pareto frontier
    pareto_df = find_pareto_frontier(df)

    # Save Pareto frontier to separate CSV
    pareto_csv = output_dir / "pareto_frontier.csv"
    pareto_df.to_csv(pareto_csv, index=False)
    print(f"Saved Pareto frontier: {pareto_csv}\n")

    # Generate visualizations
    print("Generating visualizations...")

    # 2D plots
    plot_2d_scatter(df, pareto_df, 'latency_ms', 'best_val_loss',
                   str(output_dir / 'quality_vs_latency.png'),
                   x_label='Latency (ms)', y_label='Validation Loss (ESR)',
                   title='Model Quality vs Latency')

    plot_2d_scatter(df, pareto_df, 'parameters', 'best_val_loss',
                   str(output_dir / 'quality_vs_params.png'),
                   x_label='Parameters', y_label='Validation Loss (ESR)',
                   title='Model Quality vs Complexity')

    plot_2d_scatter(df, pareto_df, 'latency_ms', 'parameters',
                   str(output_dir / 'latency_vs_params.png'),
                   x_label='Latency (ms)', y_label='Parameters',
                   title='Latency vs Model Complexity')

    # 3D plot
    try:
        plot_3d_scatter(df, pareto_df, str(output_dir / 'pareto_3d.png'))
    except Exception as e:
        print(f"Warning: Could not create 3D plot: {e}")

    # CPU performance plot (if available)
    if 'cpu_rtf' in df.columns and df['cpu_rtf'].notna().any():
        plot_2d_scatter(df, pareto_df, 'cpu_rtf', 'best_val_loss',
                       str(output_dir / 'quality_vs_cpu_rtf.png'),
                       x_label='CPU Real-time Factor (lower is better)',
                       y_label='Validation Loss (ESR)',
                       title='Model Quality vs CPU Performance')

    # Generate summary report
    print("\nGenerating summary report...")
    summary_md = output_dir / "summary.md"
    generate_summary_report(df, pareto_df, str(summary_md))

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"  - Pareto frontier CSV: pareto_frontier.csv")
    print(f"  - Summary report: summary.md")
    print(f"  - Visualization plots: *.png")
    print(f"{'='*70}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze NAS results and find Pareto frontier')
    parser.add_argument('csv_path', type=str,
                       help='Path to results CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis (default: same as CSV)')

    args = parser.parse_args()

    analyze_results(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()
