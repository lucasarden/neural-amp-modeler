"""
Quick diagnostic to verify training data normalization
"""
import yaml
import sys
sys.path.append('src')
from dataset import AmpDataset

# Load config
with open('configs/config_realtime.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
print("Loading dataset with CURRENT normalization code...")
dataset = AmpDataset(
    input_file=config['data']['input_file'],
    output_file=config['data']['output_file'],
    segment_length=config['data']['segment_length'],
    sample_rate=config['data']['sample_rate']
)

# Get first segment
input_seg, output_seg = dataset[0]

print("\n" + "="*60)
print("VERIFICATION RESULTS:")
print("="*60)
print(f"Input segment shape: {input_seg.shape}")
print(f"Output segment shape: {output_seg.shape}")
print(f"\nInput stats:")
print(f"  Min: {input_seg.min():.6f}")
print(f"  Max: {input_seg.max():.6f}")
print(f"  Mean: {input_seg.mean():.6f}")
print(f"  RMS: {(input_seg**2).mean().sqrt():.6f}")
print(f"\nOutput stats:")
print(f"  Min: {output_seg.min():.6f}")
print(f"  Max: {output_seg.max():.6f}")
print(f"  Mean: {output_seg.mean():.6f}")
print(f"  RMS: {(output_seg**2).mean().sqrt():.6f}")

# Calculate gain ratio
input_rms = (input_seg**2).mean().sqrt()
output_rms = (output_seg**2).mean().sqrt()
gain_ratio = output_rms / (input_rms + 1e-8)

print(f"\nGain ratio (output/input): {gain_ratio:.2f}x")
print("\n" + "="*60)
print("EXPECTED RESULTS:")
print("="*60)
print("✓ Output RMS should be ~3-10x higher than input")
print("✓ Both signals should be in range [-1, 1]")
print("✓ If gain ratio is ~1.0, normalization is BROKEN")
print("="*60)
