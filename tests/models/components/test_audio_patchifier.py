import pytest
import torch

from ami.models.components.audio_patchifier import AudioPatchifier


class TestAudioPatchifier:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_samples", [16320])
    @pytest.mark.parametrize("in_channels", [1, 2]) # monoral and stereo audio respectively.
    @pytest.mark.parametrize("out_channels", [512])
    def test_forward(self, batch_size: int, n_samples: int, in_channels: int, out_channels: int):
        audio_patchifier = AudioPatchifier(in_channels=in_channels, out_channels=out_channels)
        input_audios = torch.randn(batch_size, in_channels, n_samples)
        output_patches = audio_patchifier(input_audios)
        assert isinstance(output_patches, torch.Tensor)
        # Assuming window_size=400[samples] and hop_size=320[samples] as total.
        window_size, hop_size = 400, 320
        expected_n_patches = int((n_samples - 400) / 320) + 1
        assert output_patches.shape == (batch_size, expected_n_patches, out_channels)
