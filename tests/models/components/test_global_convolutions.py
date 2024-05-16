import pytest
import torch

from ami.models.components.global_convolutions import GlobalConvolutions1d


class TestGlobalConvolutions1d:
    @pytest.mark.parametrize(
        "batch_size, in_channels, out_channels, seq_length, num_layers, kernel_size, padding, do_batchnorm, global_pool_method",
        [
            (8, 3, 5, 10, 2, 3, 1, False, "mean"),
            (8, 3, 5, 10, 2, 3, 1, False, "max"),
            (8, 3, 5, 10, 2, 3, 1, True, "mean"),
            (8, 3, 5, 10, 2, 3, 1, True, "max"),
            (8, 3, 5, 20, 3, 5, "same", True, "mean"),
        ],
    )
    def test_global_convolutions_1d(
        self,
        batch_size,
        in_channels,
        out_channels,
        seq_length,
        num_layers,
        kernel_size,
        padding,
        do_batchnorm,
        global_pool_method,
    ):
        model = GlobalConvolutions1d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            padding=padding,
            do_batchnorm=do_batchnorm,
            global_pool_method=global_pool_method,
        )

        input_tensor = torch.randn(batch_size, in_channels, seq_length)
        output_tensor = model(input_tensor)

        assert output_tensor.shape == (
            batch_size,
            out_channels,
        ), f"Expected shape {(batch_size, out_channels)}, but got {output_tensor.shape}"
