"""
Dataset class for loading and preprocessing audio data
"""

import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np


class AmpDataset(Dataset):
    """Dataset for amp modeling - pairs of input/output audio"""

    def __init__(self, input_file, output_file, segment_length=8192, sample_rate=48000):
        """
        Args:
            input_file: Path to clean guitar DI recording
            output_file: Path to amp output recording
            segment_length: Length of each training segment (samples)
            sample_rate: Audio sample rate
        """
        self.segment_length = segment_length
        self.sample_rate = sample_rate

        # Load audio files
        print(f"Loading {input_file}...")
        self.input_audio, sr_in = sf.read(input_file)
        print(f"Loading {output_file}...")
        self.output_audio, sr_out = sf.read(output_file)

        # Convert stereo to mono if needed
        if self.input_audio.ndim == 2:
            print(f"  Input is stereo, converting to mono (averaging channels)")
            self.input_audio = np.mean(self.input_audio, axis=1)
        if self.output_audio.ndim == 2:
            print(f"  Output is stereo, converting to mono (averaging channels)")
            self.output_audio = np.mean(self.output_audio, axis=1)

        # Verify sample rates match
        assert (
            sr_in == sr_out == sample_rate
        ), f"Sample rate mismatch: {sr_in}, {sr_out}, expected {sample_rate}"

        # Verify lengths match
        assert len(self.input_audio) == len(
            self.output_audio
        ), "Input and output audio must be same length"

        # Normalize using the SAME scale factor to preserve gain relationship
        # Find max of both signals and use that for scaling
        max_val = max(np.abs(self.input_audio).max(), np.abs(self.output_audio).max())
        if max_val > 0:
            self.input_audio = self.input_audio / max_val
            self.output_audio = self.output_audio / max_val
            print(f"Normalized both signals by max value: {max_val:.6f}")

        # Calculate number of segments
        self.num_segments = len(self.input_audio) // segment_length

        print(f"Loaded {len(self.input_audio)} samples")
        print(f"Duration: {len(self.input_audio) / sample_rate:.2f} seconds")
        print(f"Created {self.num_segments} segments")

    def normalize(self, audio):
        """Normalize audio to [-1, 1]"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        """Get a segment of input/output pair"""
        start = idx * self.segment_length
        end = start + self.segment_length

        input_segment = self.input_audio[start:end]
        output_segment = self.output_audio[start:end]

        # Convert to torch tensors and add channel dimension
        input_tensor = torch.from_numpy(input_segment).float().unsqueeze(0)
        output_tensor = torch.from_numpy(output_segment).float().unsqueeze(0)

        return input_tensor, output_tensor


if __name__ == "__main__":
    # Test dataset (will fail until you have audio files)
    print("Dataset class loaded successfully!")
    print("Add audio files to data\\raw\\ to test loading")
