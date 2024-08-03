# Ref: https://github.com/facebookresearch/ijepa

import math

import numpy as np
import torch
import torch.nn as nn

from .components.positional_embeddings import get_2d_positional_embeddings
from .components.vision_transformer_layer import VisionTransformerLayer


def repeat_interleave_batch(x: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    """

    Args:
        x (torch.Tensor):
        batch_size (int):
            batch size
        repeat (int): 

    Returns:
        torch.Tensor:
            (shape: [])
    """
    N = len(x) // batch_size
    x = torch.cat(
        [torch.cat([x[i * batch_size : (i + 1) * batch_size] for _ in range(repeat)], dim=0) for i in range(N)],
        dim=0,
    )
    return x


def select_mask_patches(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Extract patches from x using masks.

    Args:
        x (torch.Tensor):
            Input patches.
            (shape: [batch_size, n_patches, dims])
        masks (list[torch.Tensor]):
            Masks to select indices of patches.
            (len(masks) is num of masks, shape of all Tensors: [batch_size, n_patches_to_be_selected])

    Returns:
        torch.Tensor:
            patches Selected from input x.
            (shape: [batch_size, n_patches_to_be_selected, dims])
    """
    selected_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        selected_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(selected_x, dim=0)


class PatchEmbed(nn.Module):
    """Convert input images into patch embeddings."""

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """

        Args:
            patch_size (int):
                Pixel size per a patch.
                Defaults to 16.
            in_channels (int):
                Num of input images channels.
                Defaults to 3.
            embed_dim (int):
                Num of embed dimensions per a patch
                Defaults to 768.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
        )

    # (batch, channels, height, width) -> (batch, n_patches, embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(-2).transpose(-2, -1)
        return x


class VisionTransformerEncoder(nn.Module):
    """Used as I-JEPA context_encoder and target_encoder."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
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
        """Used as I-JEPA context_encoder and target_encoder.

        Args:
            img_size (int):
                Input image size.
                Defaults to 224.
            patch_size (int):
                Pixel size per a patch.
                Defaults to 16.
            in_channels (int):
                Input images channels.
                Defaults to 3.
            embed_dim (int):
                Output dims per a patch.
                Defaults to 768.
            depth (int):
                Num of transformer layers.
                Defaults to 12.
            num_heads (int):
                Num of heads for transformer layers.
                Defaults to 12.
            mlp_ratio (float):
                Specify hidden dims for transformer layers.
                Defaults to 4.0.
            qkv_bias (bool):
                Whether to use bias in MLPs used to get qkv in attention module.
                Defaults to True.
            qk_scale (float | None):
                The multiplier to be applied to the matrix product of q and v in attention module.
                Defaults to None.
            drop_rate (float):
                Ratio of Dropout to be performed within last MLP in each transformer layers.
                Defaults to 0.0.
            attn_drop_rate (float):
                Ratio of Dropout to be performed within MLP for the matrix product of q and k in attention module.
                Defaults to 0.0.
            drop_path_rate (float):
                Maximum value of stochastic depth decay rule.
                Defaults to 0.0.
            init_std (float):
                Std for initializing layers.
                Defaults to 0.02.
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # define input layer to convert input image into patches.
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        # define positional encodings
        assert img_size % patch_size == 0
        n_patches = (img_size//patch_size)**2
        self.positional_encodings: torch.Tensor
        positional_encodings = get_2d_positional_embeddings(
            embed_dim,
            img_size//patch_size,
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
        # initialize
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.vit_layers, start=1):
            rescale(layer.attn.proj.weight.data, layer_id)
            rescale(layer.mlp.fc2.weight.data, layer_id)

    def _init_weights(self, m: nn.Module) -> None:
        match m:
            case nn.Linear():
                torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            case nn.LayerNorm():
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            case nn.Conv2d():
                torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        images: torch.Tensor,
        masks_for_context_encoder: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode input images into latents.

        Args:
            images (torch.Tensor):
                Input images.
                (shape: [batch_size, 3, img_size, img_size])
            masks_for_context_encoder (list[torch.Tensor] | None):
                Masks to select indices of images embedded as patches.
                (shape of each Tensor: [batch_size, n_patches])
                If None, all patches are selected and used to create latents.
                Default to None.

        Returns:
            torch.Tensor:
                if masks_for_context_encoder is None:
                    shape: [batch_size, n_patches, embed_dim]
                else:
                    shape: [batch_size*len(masks_for_context_encoder), n_patches, embed_dim]
        """
        # patchify input images
        x = self.patch_embed(images)
        # x: [batch_size, n_patches, embed_dim]

        # add positional embedding to x
        positional_encodings = self.interpolate_pos_encoding(x, self.positional_encodings)
        x = x + positional_encodings

        # Extract patches from x based on indices contained in masks_for_context_encoder
        if masks_for_context_encoder is not None:
            x = select_mask_patches(x, masks_for_context_encoder)

        # Apply Vision Transformers
        for vit_layer in self.vit_layers:
            x = vit_layer(x)

        x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x: torch.Tensor, positional_embeddings: torch.Tensor) -> torch.Tensor:
        npatch = x.shape[1] - 1
        N = positional_embeddings.shape[1] - 1
        if npatch == N:
            return positional_embeddings
        class_emb = positional_embeddings[:, 0]
        positional_embeddings = positional_embeddings[:, 1:]
        dim = x.shape[-1]
        positional_embeddings = nn.functional.interpolate(
            positional_embeddings.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode="bicubic",
        )
        positional_embeddings = positional_embeddings.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), positional_embeddings), dim=1)


class VisionTransformerPredictor(nn.Module):
    """Used as I-JEPA predictor."""

    def __init__(
        self,
        n_patches: int,
        context_encoder_embed_dim: int = 768,
        predictor_embed_dim: int = 384,
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
        """Used as I-JEPA predictor.

        Args:
            n_patches (int):
                Total num of patches.
            context_encoder_embed_dim (int):
                Embedding dims of input latents.
                Defaults to 768.
            predictor_embed_dim (int):
                Convert the input dimension to this dimension for prediction.
                Defaults to 384.
            depth (int):
                Num of transformer layers.
                Defaults to 6.
            num_heads (int):
                Num of heads for transformer layers.
                Defaults to 12.
            mlp_ratio (float):
                Specify hidden dims for transformer layers.
                Defaults to 4.0.
            qkv_bias (bool):
                Whether to use bias in MLPs used to get qkv in attention module.
                Defaults to True.
            qk_scale (float | None):
                The multiplier to be applied to the matrix product of q and v in attention module.
                Defaults to None.
            drop_rate (float):
                Ratio of Dropout to be performed within last MLP in each transformer layers.
                Defaults to 0.0.
            attn_drop_rate (float):
                Ratio of Dropout to be performed within MLP for the matrix product of q and k in attention module.
                Defaults to 0.0.
            drop_path_rate (float):
                Maximum value of stochastic depth decay rule.
                Defaults to 0.0.
            init_std (float):
                Std for initializing layers.
                Defaults to 0.02.
        """

        super().__init__()
        self.predictor_embed = nn.Linear(context_encoder_embed_dim, predictor_embed_dim, bias=True)
        # prepare mask tokens representing patches to be predicted
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule
        # define positional encodings
        self.positional_encodings: torch.Tensor
        positional_encodings = get_2d_positional_embeddings(
            predictor_embed_dim, int(n_patches**0.5)
        ).reshape(1, n_patches, predictor_embed_dim)
        self.register_buffer("positional_encodings", torch.from_numpy(positional_encodings).float())
        # define transformers
        self.vit_layers = nn.ModuleList(
            [
                VisionTransformerLayer(
                    embedding_dim=predictor_embed_dim,
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
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_embed_dim, context_encoder_embed_dim, bias=True)
        # initialize
        self.init_std = init_std
        torch.nn.init.trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.vit_layers, start=1):
            rescale(layer.attn.proj.weight.data, layer_id)
            rescale(layer.mlp.fc2.weight.data, layer_id)

    def _init_weights(self, m: nn.Module) -> None:
        match m:
            case nn.Linear():
                torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            case nn.LayerNorm():
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            case nn.Conv2d():
                torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        latents: torch.Tensor,
        masks_for_context_encoder: list[torch.Tensor],
        masks_for_predictor: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict latents of patches at the position specified by
        masks_for_predictor with a hint of input latents and positional
        information (masks_for_context_encoder).

        Args:
            latents (torch.Tensor):
                Input latents from context_encoder.
                (shape: [batch_size, n_patches_selected_in_context_encoder, context_encoder_embed_dim])
            masks_for_context_encoder (list[torch.Tensor]):
                Masks used to create input latents above using context_encoder.
                (shape of each Tensor: [batch_size, n_patches_selected_in_context_encoder])
            masks_for_predictor (list[torch.Tensor]):
                Masks corresponding to the position of the patches to be predict.
                (shape of each Tensor: [batch_size, n_patches_to_predict])

        Returns:
            torch.Tensor:
                prediction results.
                (shape: [batch_size*len(masks_for_predictor), n_patches_to_predict, context_encoder_embed_dim])
        """

        batch_size = len(latents) // len(masks_for_context_encoder)

        # map from encoder-dim to pedictor-dim
        x = self.predictor_embed(latents)

        # add positional embedding correspond to context_encoder
        x_positional_encodings = self.positional_encodings.repeat(batch_size, 1, 1)
        x += select_mask_patches(x_positional_encodings, masks_for_context_encoder)

        _, boundary, _ = x.shape

        # prepare positional embedding for patches to be predicted
        positional_encodings_for_prediction = self.positional_encodings.repeat(batch_size, 1, 1)
        positional_encodings_for_prediction = select_mask_patches(positional_encodings_for_prediction, masks_for_predictor)
        positional_encodings_for_prediction = repeat_interleave_batch(positional_encodings_for_prediction, batch_size, repeat=len(masks_for_context_encoder))
        # prepare mask tokens for patch to be predicted
        tokens_for_prediction = self.mask_token.repeat(positional_encodings_for_prediction.size(0), positional_encodings_for_prediction.size(1), 1)
        # add positional embedding
        tokens_for_prediction += positional_encodings_for_prediction
        # concat x with patches to be predicted
        x = x.repeat(len(masks_for_predictor), 1, 1)
        x = torch.cat([x, tokens_for_prediction], dim=1)

        # Apply Vision Transformers
        for vit_layer in self.vit_layers:
            x = vit_layer(x)
        x = self.predictor_norm(x)

        # return predictions for mask tokens
        x = x[:, boundary:]
        x = self.predictor_proj(x)

        return x
