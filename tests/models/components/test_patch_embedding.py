import pytest
import torch

from ami.models.components.patch_embedding import PatchEmbedding


class TestPatchEmbedding:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("img_size", [224, 512])
    @pytest.mark.parametrize("patch_size", [16])
    @pytest.mark.parametrize("embed_dim", [768])
    def test_forward(self, batch_size, img_size, patch_size, embed_dim):

        layer = PatchEmbedding(patch_size, 3, embed_dim)

        image = torch.randn(batch_size, 3, img_size, img_size)
        out = layer(image)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size, (img_size // patch_size) ** 2, embed_dim)
