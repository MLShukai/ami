# Ref: https://github.com/facebookresearch/RCDM

import math

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """A residual block for Unet Components."""

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        out_channels: int,
        num_norm_groups_in_input_layer: int = 32,
        num_norm_groups_in_output_layer: int = 32,
    ) -> None:
        """A residual block for Unet Components.

        Args:
            in_channels (int):
                Input num of channels.
            emb_channels (int):
                Input embedding num of channels.
            out_channels (int):
                Output num of channels.
            num_norm_groups_in_input_layer (int):
                num of groups in nn.GroupNorm in self.in_layers.
                Default to 32.
            num_norm_groups_in_output_layer (int):
                num of groups in nn.GroupNorm in self.out_layers.
                Default to 32.
        """
        super().__init__()
        assert in_channels%num_norm_groups_in_input_layer == 0
        assert out_channels%num_norm_groups_in_output_layer == 0
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=num_norm_groups_in_input_layer, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=num_norm_groups_in_output_layer, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        # initialize conv layer as zeros
        for p in self.out_layers[-1].parameters():
            p.detach().zero_()

        self.skip_connection: nn.Module = (
            nn.Identity() if out_channels == in_channels else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the block to input Tensor, conditioned on emb.

        Args:
            x (torch.Tensor):
                Input features.
                (shape: [batch_size, in_channels, height, width])
            emb (torch.Tensor):
                Input Embeddings for conditioning.
                (shape: [batch_size, emb_channels])
        Returns:
            torch.Tensor:
                Output features.
                (shape: [batch_size, out_channels, height, width])
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # FiLM
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
        return self.skip_connection(x) + h


# To Do:
# Temporarily use this implementation of the original paper,
# but it will be changed to use nn.MultiheadAttention in Unet components.
class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each
    other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = nn.Conv1d(channels, channels, 1)
        # initialize as zeros
        self.proj_out.weight.detach().zero_()
        self.proj_out.bias.detach().zero_()  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# To Do:
# Temporarily use this implementation of the original paper,
# but it will be changed to use nn.MultiheadAttention in Unet components.
class QKVAttentionLegacy(nn.Module):
    """A module which performs QKV attention.

    Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
