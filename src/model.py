"""
WaveNet-style neural network for amp modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Single residual block with dilated convolution"""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding="same", dilation=dilation
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        # Store input for residual connection
        residual = x

        # Apply convolution and activation
        out = self.conv(x)
        out = self.activation(out)

        # Add residual connection
        return out + residual


class WaveNetAmp(nn.Module):
    """WaveNet-style architecture for amp modeling"""

    def __init__(self, channels=16, num_layers=10, kernel_size=3, dilation_base=2):
        super().__init__()

        # Input convolution: 1 channel (mono) -> channels
        self.input_conv = nn.Conv1d(1, channels, kernel_size=1)

        # Stack of residual blocks with increasing dilation
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**i,
                )
                for i in range(num_layers)
            ]
        )

        # Output convolution: channels -> 1 channel (mono)
        self.output_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, 1, time_samples)
        x = self.input_conv(x)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output layer
        x = self.output_conv(x)

        return x

    def receptive_field(self, kernel_size=3, num_layers=10, dilation_base=2):
        """Calculate the receptive field of the model"""
        rf = 1
        for i in range(num_layers):
            dilation = dilation_base**i
            rf += (kernel_size - 1) * dilation
        return rf


if __name__ == "__main__":
    # Test the model
    model = WaveNetAmp(channels=16, num_layers=10)

    # Print model info
    print(f"Model: {model.__class__.__name__}")
    print(f"Receptive field: {model.receptive_field()} samples")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    test_input = torch.randn(1, 1, 8192)  # (batch, channels, samples)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
