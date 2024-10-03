import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .components.patch_embedding import PatchEmbedding
from .components.positional_embeddings import get_2d_positional_embeddings
from .components.vision_transformer_layer import VisionTransformerLayer
from .model_wrapper import ModelWrapper
from .utils import size_2d, size_2d_to_int_tuple


class BoolMaskIJEPAEncoder(nn.Module):
    """Used as I-JEPA context_encoder and target_encoder with boolean mask
    support."""

    def __init__(
        self,
        img_size: size_2d = 224,
        patch_size: size_2d = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        out_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the BoolMaskIJEPAEncoder.

        Args:
            img_size (size_2d): Input image size. Defaults to 224.
            patch_size (size_2d): Pixel size per patch. Defaults to 16.
            in_channels (int): Input image channels. Defaults to 3.
            embed_dim (int): Embedding dimension per patch. Defaults to 768.
            out_dim (int): Output dimension per patch. Defaults to 384.
            depth (int): Number of transformer layers. Defaults to 12.
            num_heads (int): Number of attention heads for transformer layers. Defaults to 12.
            mlp_ratio (float): Ratio for MLP hidden dimension in transformer layers. Defaults to 4.0.
            qkv_bias (bool): Whether to use bias in query, key, value projections. Defaults to True.
            qk_scale (float | None): Scale factor for query-key dot product. Defaults to None.
            drop_rate (float): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float): Attention dropout rate. Defaults to 0.0.
            drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
            init_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads

        # define input layer to convert input image into patches.
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # define mask token_vector
        self.mask_token_vector = nn.Parameter(torch.empty(embed_dim))

        # define positional encodings
        img_size = size_2d_to_int_tuple(img_size)
        patch_size = size_2d_to_int_tuple(patch_size)
        img_height, img_width = img_size
        patch_height, patch_width = patch_size

        assert img_height % patch_height == 0
        assert img_width % patch_width == 0

        n_patches_hw = (img_height // patch_height), (img_width // patch_width)
        n_patches = n_patches_hw[0] * n_patches_hw[1]

        self.positional_encodings: Tensor
        positional_encodings = get_2d_positional_embeddings(
            embed_dim,
            n_patches_hw,
        ).reshape(1, n_patches, embed_dim)
        self.register_buffer("positional_encodings", torch.from_numpy(positional_encodings).float())

        # define transformers
        dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule
        self.vit_layers = nn.ModuleList(
            [
                VisionTransformerLayer(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.out_proj = nn.Linear(embed_dim, out_dim)

        # initialize
        self.init_std = init_std
        nn.init.trunc_normal_(self.mask_token_vector, std=self.init_std)
        self.apply(partial(_init_weights, init_std=init_std))
        fix_init_weight(self.vit_layers)

    def forward(self, images: Tensor, masks_for_context_encoder: Tensor | None = None) -> Tensor:
        """Encode input images into latents, applying boolean masks if
        provided.

        Args:
            images (Tensor): Input images.
                Shape: [batch_size, 3, height, width]
            masks_for_context_encoder (Tensor | None): Boolean masks for images embedded as patches.
                Shape: [batch_size, n_patches]. True values indicate masked patches. Defaults to None.

        Returns:
            Tensor: Encoded latents. Shape: [batch_size, n_patches, embed_dim]
        """
        # Patchify input images
        x: Tensor = self.patch_embed(images)
        # x: [batch_size, n_patches, embed_dim]

        # Apply mask if provided
        if masks_for_context_encoder is not None:
            assert x.shape[:-1] == masks_for_context_encoder.shape
            x = x.clone()  # Avoid breaking gradient graph
            x[masks_for_context_encoder] = self.mask_token_vector

        # Add positional embedding to x
        x = x + self.positional_encodings

        # Apply Vision Transformers
        for vit_layer in self.vit_layers:
            x = vit_layer(x)

        x = self.norm(x)
        x = self.out_proj(x)
        return x


class BoolTargetIJEPAPredictor(nn.Module):
    """Used as I-JEPA predictor with boolean target support."""

    def __init__(
        self,
        n_patches: size_2d,
        context_encoder_out_dim: int = 384,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        """Initialize the BoolTargetIJEPAPredictor.

        Args:
            n_patches (size_2d): Number of patches along vertical and horizontal axes.
            context_encoder_out_dim (int): Output dimension of the context encoder. Defaults to 384.
            hidden_dim (int): Hidden dimension for prediction. Defaults to 384.
            depth (int): Number of transformer layers. Defaults to 6.
            num_heads (int): Number of attention heads for transformer layers. Defaults to 12.
            mlp_ratio (float): Ratio for MLP hidden dimension in transformer layers. Defaults to 4.0.
            qkv_bias (bool): Whether to use bias in query, key, value projections. Defaults to True.
            qk_scale (float | None): Scale factor for query-key dot product. Defaults to None.
            drop_rate (float): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float): Attention dropout rate. Defaults to 0.0.
            drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
            init_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        """

        super().__init__()

        self.input_proj = nn.Linear(context_encoder_out_dim, hidden_dim, bias=True)

        # prepare tokens representing patches to be predicted
        self.prediction_token_vector = nn.Parameter(torch.empty(hidden_dim))

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, depth).tolist()

        # define positional encodings
        (n_patches_vertical, n_patches_horizontal) = size_2d_to_int_tuple(n_patches)
        positional_encodings = get_2d_positional_embeddings(
            hidden_dim, grid_size=(n_patches_vertical, n_patches_horizontal)
        ).reshape(1, n_patches_vertical * n_patches_horizontal, hidden_dim)

        self.positional_encodings: torch.Tensor
        self.register_buffer("positional_encodings", torch.from_numpy(positional_encodings).float())

        # define transformers
        self.vit_layers = nn.ModuleList(
            [
                VisionTransformerLayer(
                    embedding_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(hidden_dim, context_encoder_out_dim, bias=True)

        # initialize
        self.init_std = init_std
        nn.init.trunc_normal_(self.prediction_token_vector, std=self.init_std)
        self.apply(partial(_init_weights, init_std=init_std))
        fix_init_weight(self.vit_layers)

    def forward(
        self,
        latents: Tensor,
        predictor_targets: Tensor,
    ) -> Tensor:
        """Predict latents of target patches based on input latents and boolean
        targets.

        Args:
            latents (Tensor): Input latents from context_encoder.
                Shape: [batch, n_patches, context_encoder_out_dim]
            predictor_targets (Tensor): Boolean targets for patches.
                Shape: [batch, n_patches]. True values indicate target patches to be predicted.

        Returns:
            Tensor: Prediction results for target patches.
                    Shape: [batch, n_patches, context_encoder_out_dim]
        """
        # Map from encoder-dim to predictor-dim
        x: Tensor = self.input_proj(latents)

        # Shape: [batch, n_patches, hidden_dim]
        # Apply targets: adding prediction tokens,
        x = x.clone()  # Avoid breaking gradient graph.
        x[predictor_targets] += self.prediction_token_vector

        x = x + self.positional_encodings

        # Apply Vision Transformers
        for vit_layer in self.vit_layers:
            x = vit_layer(x)
        x = self.predictor_norm(x)

        x = self.predictor_proj(x)

        return x


def fix_init_weight(vit_layers: nn.ModuleList) -> None:
    def rescale(param: Tensor, layer_id: int) -> None:
        param.div_(math.sqrt(2.0 * layer_id))

    layer: VisionTransformerLayer
    for layer_id, layer in enumerate(vit_layers, start=1):
        rescale(layer.attn.proj.weight.data, layer_id)
        rescale(layer.mlp.fc2.weight.data, layer_id)


def _init_weights(m: nn.Module, init_std: float) -> None:
    match m:
        case nn.Linear():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        case nn.LayerNorm():
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        case nn.Conv2d():
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def i_jepa_encoder_infer(wrapper: ModelWrapper[BoolMaskIJEPAEncoder], image: torch.Tensor) -> torch.Tensor:
    """Customizes the inference flow in the agent.

    Please specify to `ModelWrapper(inference_forward=<this>)`.
    Adding batch axis if image does not have it.

    Args:
        wrapper: ModelWrapper instance that wraps BoolMaskIJEPAEncoder.
        image: Input for BoolMaskIJEPAEncoder.
            shape (channel, height, width) or (batch, channel, height, width)

    Returns:
        torch.Tensor: Output of BoolMaskIJEPAEncoder.
            shape (patch, dim) or (batch, patch, dim)
    """
    device = wrapper.device
    no_batch = image.ndim == 3

    if no_batch:
        image = image.unsqueeze(0)  # batched

    image = image.to(device)

    out: torch.Tensor = wrapper(image)
    out = torch.nn.functional.layer_norm(out, (out.size(-1),))
    if no_batch:
        out = out.squeeze(0)

    return out


def encoder_infer_mean_along_patch(wrapper: ModelWrapper[BoolMaskIJEPAEncoder], image: Tensor) -> Tensor:
    """Customizes the inference flow in the encoder.

    Please specify to `ModelWrapper(inference_forward=<this>)`.
    Adds batch axis if image does not have it, applies layer normalization,
    and means latent along the patch axis.

    Args:
        wrapper: ModelWrapper instance that wraps IJEPAEncoder.
        image: Input for IJEPAEncoder.
            shape (channel, height, width) or (batch, channel, height, width)

    Returns:
        torch.Tensor: Output of IJEPAEncoder.
            shape (dim,) or (batch, dim)
    """

    return i_jepa_encoder_infer(wrapper, image).mean(-2)
