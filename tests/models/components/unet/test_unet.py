import pytest
import torch

from ami.models.components.unet.unet import UNet


class TestUNet:
    # model params
    @pytest.mark.parametrize("in_channels", [3])
    @pytest.mark.parametrize("timestep_hidden_dim", [32])
    @pytest.mark.parametrize("timestep_embed_dim", [128])
    @pytest.mark.parametrize(
        "encoder_blocks_in_and_out_channels",
        [
            # [(32, 32), (32, 32), (32, 64), (64, 96), (96, 128)],
            [(32, 32), (32, 64), (64, 96), (96, 128)],
        ],
    )
    @pytest.mark.parametrize("out_channels", [3])
    @pytest.mark.parametrize("ssl_latent_dim", [384])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("image_height", [32])
    @pytest.mark.parametrize("image_width", [32])
    @pytest.mark.parametrize("max_timestep", [1000])
    @pytest.mark.parametrize("n_patches_of_latents_from_ssl", [14 * 14])
    def test_unet(
        self,
        in_channels: int,
        timestep_hidden_dim: int,
        timestep_embed_dim: int,
        encoder_blocks_in_and_out_channels: list[tuple[int, int]],
        out_channels: int,
        ssl_latent_dim: int,
        batch_size: int,
        image_height: int,
        image_width: int,
        max_timestep: int,
        n_patches_of_latents_from_ssl: int,
    ):
        # define model
        unet = UNet(
            in_channels=in_channels,
            timestep_hidden_dim=timestep_hidden_dim,
            timestep_embed_dim=timestep_embed_dim,
            encoder_blocks_in_and_out_channels=encoder_blocks_in_and_out_channels,
            out_channels=out_channels,
            n_res_blocks=2,
            attention_start_depth=3,
            num_heads=4,
            ssl_latent_dim=ssl_latent_dim,
        )
        # prepare inputs
        input_images = torch.randn([batch_size, in_channels, image_height, image_width])
        timesteps = torch.randint(0, max_timestep, [batch_size])
        latents_from_ssl = torch.randn([batch_size, n_patches_of_latents_from_ssl, ssl_latent_dim])
        # get output
        output_images = unet(
            input_images=input_images,
            timesteps=timesteps,
            latents_from_ssl=latents_from_ssl,
        )
        # check dims
        assert output_images.size(0) == batch_size, "batch size mismatch."
        assert output_images.size(1) == out_channels, "channels mismatch."
        assert output_images.size(2) == image_height, "image size (height) mismatch."
        assert output_images.size(3) == image_width, "image size (width) mismatch."
