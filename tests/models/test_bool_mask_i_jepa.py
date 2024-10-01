import pytest
import torch

from ami.models.bool_mask_i_jepa import (
    BoolMaskIJEPAEncoder,
    BoolTargetIJEPAPredictor,
    ModelWrapper,
    encoder_infer_mean_patch,
    i_jepa_encoder_infer,
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


class TestBoolMaskIEPAEncoder:
    # model params
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
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
        image_size: int,
        patch_size: int,
        embed_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        batch_size: int,
        use_mask: bool,
    ):
        assert image_size % patch_size == 0
        # define encoder made of ViT
        encoder = BoolMaskIJEPAEncoder(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # define sample inputs
        images = torch.randn([batch_size, 3, image_size, image_size])
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patches = n_patch_vertical * n_patch_horizontal
        # make boolean mask for encoder
        masks_for_context_encoder = None
        if use_mask:
            masks_for_context_encoder = make_bool_masks_randomly(batch_size, n_patches)
        # get latents
        latent = encoder(images=images, masks_for_context_encoder=masks_for_context_encoder)
        # check size of output latent
        assert latent.size(0) == batch_size, "batch_size mismatch"
        assert latent.size(1) == n_patches, "num of patch mismatch"
        assert latent.size(2) == out_dim, "out_dim mismatch"


class TestBoolTargetIJEPAPredictor:
    # model params
    @pytest.mark.parametrize("image_size", [224])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("context_encoder_out_dim", [8])
    @pytest.mark.parametrize("hidden_dim", [32])
    @pytest.mark.parametrize("depth", [2])
    @pytest.mark.parametrize("num_heads", [2, 4])
    # test input params
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_bool_target_vision_transformer_predictor(
        self,
        image_size: int,
        patch_size: int,
        context_encoder_out_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        batch_size: int,
    ):
        assert image_size % patch_size == 0
        n_patch_vertical = image_size // patch_size
        n_patch_horizontal = image_size // patch_size
        n_patches = n_patch_vertical * n_patch_horizontal
        # define predictor made of ViT
        predictor = BoolTargetIJEPAPredictor(
            n_patches=(n_patch_vertical, n_patch_horizontal),
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


def test_i_jepa_encoder_infer(device):
    wrapper = ModelWrapper(
        BoolMaskIJEPAEncoder(
            img_size=128, patch_size=16, in_channels=3, embed_dim=64, out_dim=64, depth=4, num_heads=2
        ),
        device,
        has_inference=True,
        inference_forward=i_jepa_encoder_infer,
    )
    wrapper.to_default_device()

    out: torch.Tensor = wrapper.infer(torch.randn(3, 128, 128))
    assert out.shape == ((128 // 16) ** 2, 64)
    assert out.device == device

    out: torch.Tensor = wrapper.infer(torch.randn(8, 3, 128, 128))
    assert out.shape == (8, (128 // 16) ** 2, 64)


def test_encoder_infer_mean_patch(device):
    wrapper = ModelWrapper(
        BoolMaskIJEPAEncoder(
            img_size=128, patch_size=16, in_channels=3, embed_dim=64, out_dim=64, depth=4, num_heads=2
        ),
        device,
        has_inference=True,
        inference_forward=encoder_infer_mean_patch,
    )
    wrapper.to_default_device()

    out: torch.Tensor = wrapper.infer(torch.randn(3, 128, 128))
    assert out.shape == (64,)
    assert out.device == device

    out: torch.Tensor = wrapper.infer(torch.randn(8, 3, 128, 128))
    assert out.shape == (8, 64)
