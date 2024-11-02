import pytest
import torch

from ami.models.bool_mask_audio_jepa import (
    BoolMaskAudioJEPAEncoder,
    BoolTargetAudioJEPAPredictor,
    ModelWrapper,
    encoder_infer_mean_along_patch,
    audio_jepa_encoder_infer,
)


def make_bool_masks_randomly(
    batch_size: int,
    n_patches: int,
    mask_ratio: float = 0.75,
) -> torch.Tensor:
    """Boolean mask maker for the tests.

    Args:
        batch_size (int): Batch size.
        n_patches (int): Total number of patches.
        mask_ratio (float): Ratio of patches to be masked. Defaults to 0.75.

    Returns:
        torch.Tensor: Boolean mask tensor.
            Shape: [batch_size, n_patches].
            True values indicate masked patches.
    """
    mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
    n_masked = int(n_patches * mask_ratio)
    for i in range(batch_size):
        mask[i, torch.randperm(n_patches)[:n_masked]] = True
    return mask


class TestBoolMaskAudioJEPAEncoder:
    # model params
    @pytest.mark.parametrize("input_sample_size", [16080])
    @pytest.mark.parametrize("patch_sample_size", [400])
    @pytest.mark.parametrize("stride", [320])
    @pytest.mark.parametrize("in_channels", [1, 2])
    @pytest.mark.parametrize(
        ["embed_dim", "out_dim", "depth", "num_heads", "mlp_ratio"],
        [
            [8, 4, 2, 2, 4],  # tiny
            [32, 16, 2, 4, 4],  # small
        ],
    )
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_forward(
        self,
        input_sample_size: int,
        patch_sample_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        batch_size: int,
        use_mask: bool,
    ):
        assert patch_sample_size <= input_sample_size
        assert stride <= input_sample_size
        assert (input_sample_size - (patch_sample_size - stride)) % stride == 0
        # define encoder made of ViT
        encoder = BoolMaskAudioJEPAEncoder(
            input_sample_size=input_sample_size,
            patch_sample_size=patch_sample_size,
            stride=stride,
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # define sample inputs
        audios = torch.randn([batch_size, in_channels, input_sample_size])
        # calc num of patches
        n_patches = (input_sample_size - (patch_sample_size - stride)) // stride
        # make boolean mask for encoder
        masks_for_context_encoder = None
        if use_mask:
            masks_for_context_encoder = make_bool_masks_randomly(batch_size, n_patches)
        # get latents
        latent = encoder(audios=audios, masks_for_context_encoder=masks_for_context_encoder)
        # check size of output latent
        assert latent.size(0) == batch_size, "batch_size mismatch"
        assert latent.size(1) == n_patches, "num of patch mismatch"
        assert latent.size(2) == out_dim, "out_dim mismatch"


class TestBoolTargetAudioJEPAPredictor:
    # model params
    @pytest.mark.parametrize("input_sample_size", [16080])
    @pytest.mark.parametrize("patch_sample_size", [400])
    @pytest.mark.parametrize("stride", [320])
    @pytest.mark.parametrize("in_channels", [1, 2])
    @pytest.mark.parametrize("context_encoder_out_dim", [8])
    @pytest.mark.parametrize("hidden_dim", [32])
    @pytest.mark.parametrize("depth", [2])
    @pytest.mark.parametrize("num_heads", [2, 4])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_bool_target_vision_transformer_predictor(
        self,
        input_sample_size: int,
        patch_sample_size: int,
        stride: int,
        in_channels: int,
        context_encoder_out_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        batch_size: int,
    ):
        assert patch_sample_size <= input_sample_size
        assert stride <= input_sample_size
        assert (input_sample_size - (patch_sample_size - stride)) % stride == 0
        # calc num of patches
        n_patches = (input_sample_size - (patch_sample_size - stride)) // stride
        # define predictor made of ViT
        predictor = BoolTargetAudioJEPAPredictor(
            n_patches=n_patches,
            context_encoder_out_dim=context_encoder_out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
        )
        # define sample inputs
        latents = torch.randn([batch_size, n_patches, context_encoder_out_dim])
        predictor_targets = ~make_bool_masks_randomly(batch_size, n_patches)
        # get predictions
        predictions = predictor(
            latents=latents,
            predictor_targets=predictor_targets,
        )
        # check size of output predictions
        assert predictions.size(0) == batch_size, "batch_size mismatch"
        assert predictions.size(1) == n_patches, "num of patch mismatch"
        assert predictions.size(2) == context_encoder_out_dim, "out_dim mismatch"

    
def test_audio_jepa_encoder_infer(device):
    wrapper = ModelWrapper(
        BoolMaskAudioJEPAEncoder(
            input_sample_size=16080,
            patch_sample_size=400,
            stride=320,
            in_channels=2,
            embed_dim=64, 
            out_dim=64, 
            depth=4, 
            num_heads=2,
        ),
        device,
        has_inference=True,
        inference_forward=audio_jepa_encoder_infer,
    )
    wrapper.to_default_device()
    
    n_patches = 50

    out: torch.Tensor = wrapper.infer(torch.randn(2, 16080))
    assert out.shape == (n_patches, 64)
    assert out.device == device

    out: torch.Tensor = wrapper.infer(torch.randn(8, 2, 16080))
    assert out.shape == (8, n_patches, 64)


def test_encoder_infer_mean_patch(device):
    wrapper = ModelWrapper(
        BoolMaskAudioJEPAEncoder(
            input_sample_size=16080,
            patch_sample_size=400,
            stride=320,
            in_channels=2,
            embed_dim=64, 
            out_dim=64, 
            depth=4, 
            num_heads=2,
        ),
        device,
        has_inference=True,
        inference_forward=encoder_infer_mean_along_patch,
    )
    wrapper.to_default_device()

    out: torch.Tensor = wrapper.infer(torch.randn(2, 16080))
    assert out.shape == (64,)
    assert out.device == device

    out: torch.Tensor = wrapper.infer(torch.randn(8, 2, 16080))
    assert out.shape == (8, 64)
