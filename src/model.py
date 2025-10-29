"""
WaveNet-style neural network for amp modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Single residual block with dilated convolution

    Supports both causal (real-time) and non-causal (offline) modes.
    Causal mode uses left-only padding to avoid looking into the future.
    """

    def __init__(self, channels, kernel_size, dilation, causal=False):
        super().__init__()
        self.causal = causal

        if causal:
            # Causal: pad only on the left (past samples only)
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                channels, channels, kernel_size, padding=0, dilation=dilation
            )
        else:
            # Non-causal: centered padding (uses future samples)
            self.conv = nn.Conv1d(
                channels, channels, kernel_size, padding="same", dilation=dilation
            )

        self.activation = nn.Tanh()

    def forward(self, x):
        # Store input for residual connection (before padding)
        residual = x

        if self.causal:
            # Manually pad on the left only
            x = F.pad(x, (self.padding, 0))

        # Apply convolution and activation
        out = self.conv(x)
        out = self.activation(out)

        # Add residual connection
        # For causal: output size matches original input size (padding is consumed by conv)
        # For non-causal: output size already matches input due to padding="same"
        return out + residual


class WaveNetAmp(nn.Module):
    """WaveNet-style architecture for amp modeling

    Args:
        channels: Number of channels in residual blocks
        num_layers: Number of residual blocks
        kernel_size: Convolution kernel size
        dilation_base: Base for exponential dilation (2^i pattern)
        causal: If True, use causal convolutions for real-time processing
    """

    def __init__(
        self, channels=16, num_layers=10, kernel_size=3, dilation_base=2, causal=False
    ):
        super().__init__()

        self.causal = causal
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base

        # Input convolution: 1 channel (mono) -> channels
        self.input_conv = nn.Conv1d(1, channels, kernel_size=1)

        # Stack of residual blocks with increasing dilation
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation_base**i,
                    causal=causal,
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

    def receptive_field(self):
        """Calculate the receptive field of the model

        The receptive field determines the minimum latency for real-time processing.
        For causal models, this is the amount of past context needed.
        """
        rf = 1
        for i in range(self.num_layers):
            dilation = self.dilation_base**i
            rf += (self.kernel_size - 1) * dilation
        return rf


class StreamingWaveNetAmp:
    """Wrapper for real-time streaming audio processing

    Manages internal buffer to maintain context between chunks for causal models.
    This allows processing audio in small chunks (e.g., 128-512 samples) while
    maintaining the full receptive field context from previous chunks.

    Args:
        model: WaveNetAmp model (must be causal=True)
        device: torch device ('cuda' or 'cpu')

    Example:
        >>> model = WaveNetAmp(channels=16, num_layers=6, causal=True)
        >>> streamer = StreamingWaveNetAmp(model, device='cuda')
        >>> chunk = torch.randn(1, 1, 512)  # Process 512 samples
        >>> output = streamer.process_chunk(chunk)
    """

    def __init__(self, model, device="cpu"):
        if not model.causal:
            raise ValueError(
                "StreamingWaveNetAmp requires a causal model. "
                "Initialize WaveNetAmp with causal=True"
            )

        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.receptive_field = model.receptive_field()

        # Initialize buffer with zeros (context from past)
        self.buffer = torch.zeros(1, 1, self.receptive_field, device=device)

    def process_chunk(self, chunk):
        """Process a single chunk of audio with context from buffer

        Args:
            chunk: Audio tensor of shape (batch, 1, samples)

        Returns:
            Processed audio tensor of same shape as input chunk
        """
        # Move chunk to correct device
        chunk = chunk.to(self.device)

        # Concatenate buffer (past context) with new chunk
        input_with_context = torch.cat([self.buffer, chunk], dim=-1)

        # Process through model
        with torch.no_grad():
            output = self.model(input_with_context)

        # Update buffer: keep last receptive_field samples as context for next chunk
        # This includes both the new chunk and enough past context
        self.buffer = input_with_context[:, :, -self.receptive_field :]

        # Return only the output corresponding to the new chunk
        # (discard output from buffered context)
        return output[:, :, -chunk.shape[-1] :]

    def reset(self):
        """Clear the internal buffer

        Call this between separate audio streams or when starting a new session.
        """
        self.buffer.zero_()

    def get_latency_samples(self):
        """Get the latency in samples (equal to receptive field)"""
        return self.receptive_field

    def get_latency_ms(self, sample_rate=44100):
        """Get the latency in milliseconds"""
        return (self.receptive_field / sample_rate) * 1000


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Non-Causal Model (Offline Processing)")
    print("=" * 60)

    # Test non-causal model
    model_offline = WaveNetAmp(channels=16, num_layers=10, causal=False)

    # Print model info
    print(f"Model: {model_offline.__class__.__name__}")
    print(f"Mode: Non-Causal (uses future samples)")
    print(f"Receptive field: {model_offline.receptive_field()} samples")
    print(f"Parameters: {sum(p.numel() for p in model_offline.parameters()):,}")

    # Test forward pass
    test_input = torch.randn(1, 1, 8192)  # (batch, channels, samples)
    output = model_offline(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("Testing Causal Model (Real-Time Processing)")
    print("=" * 60)

    # Test causal model
    model_realtime = WaveNetAmp(channels=16, num_layers=10, causal=True)

    print(f"Model: {model_realtime.__class__.__name__}")
    print(f"Mode: Causal (no future samples)")
    print(f"Receptive field: {model_realtime.receptive_field()} samples")
    latency_ms = model_realtime.receptive_field() / 44100 * 1000
    print(f"Minimum latency: {latency_ms:.2f}ms @ 44.1kHz")
    print(f"Parameters: {sum(p.numel() for p in model_realtime.parameters()):,}")

    # Test forward pass
    output_rt = model_realtime(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_rt.shape}")

    print("\n" + "=" * 60)
    print("Low-Latency Configuration Example")
    print("=" * 60)

    # Test low-latency causal model
    model_low_latency = WaveNetAmp(channels=16, num_layers=6, causal=True)

    print(f"Layers reduced: 10 -> 6")
    print(f"Receptive field: {model_low_latency.receptive_field()} samples")
    latency_ms = model_low_latency.receptive_field() / 44100 * 1000
    print(f"Minimum latency: {latency_ms:.2f}ms @ 44.1kHz")
    print(f"Parameters: {sum(p.numel() for p in model_low_latency.parameters()):,}")

    # Check CUDA availability
    print("\n" + "=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
