import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import CenterCrop


class SmallDeconvNet(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        dim_in: int,
        positional_bias: bool = True,
        nl: Callable[[Tensor], Tensor] = nn.LeakyReLU(negative_slope=0.2),
        do_batchnorm: bool = False,
    ):
        """Reconstruct images from latent variables. `strides` differs from
        original implementation. For the original implementation, see
        https://github.com/openai/large-scale-
        curiosity/blob/master/utils.py#L147.

        Args:
            height (int): height of the reconstructed image.
            width (int): width of the reconstructed image.
            channels (int): Channels of the reconstructed image.
            dim_in (int): The size of latent variable.
            positional_bias (bool): Whether to add positional bias or not in last layer.
            nl (Callable): NonLinear function for activation.
            do_batchnorm(bool, optional): Whether to do batchnorm. Defaults to False.
        """
        super().__init__()
        batch_norm_cls = nn.BatchNorm2d if do_batchnorm else nn.Identity

        self.height = height
        self.width = width
        self.channels = channels
        self.kernel_sizes = ((4, 4), (8, 8), (8, 8))
        self.strides = ((2, 2), (2, 2), (4, 4))
        self.paddings = ((1, 1), (3, 3), (2, 2))
        self.do_batchnorm = do_batchnorm

        init_output_size = self.init_output_size
        dim_output_init_fc = init_output_size[0] * init_output_size[1]
        self.fc_init = nn.Linear(dim_in, dim_output_init_fc)

        self.deconv1 = nn.ConvTranspose2d(
            1, 128, kernel_size=self.kernel_sizes[0], stride=self.strides[0], padding=self.paddings[0]
        )
        self.bn1 = batch_norm_cls(128)
        self.deconv2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=self.kernel_sizes[1],
            stride=self.strides[1],
            padding=self.paddings[1],
        )
        self.bn2 = batch_norm_cls(64)
        self.deconv3 = nn.ConvTranspose2d(
            64, 3, kernel_size=self.kernel_sizes[2], stride=self.strides[2], padding=self.paddings[2]
        )

        self.center_crop = CenterCrop((height, width))

        self.bias = nn.Parameter(torch.zeros(channels, height, width), requires_grad=True) if positional_bias else None
        self.nl = nl

    @property
    def init_output_size(self) -> tuple[int, ...]:
        """Execute `_compute_input_shape` for the same number of times as
        convolution layers.

        Returns:
            tuple[int, int]: Required size for self.conv1.
        """
        output_size: tuple[int, ...] = (self.height, self.width)
        for kernel_size, stride, padding in zip(self.kernel_sizes[::-1], self.strides[::-1], self.paddings[::-1]):
            output_size = tuple(map(self._compute_input_shape, output_size, kernel_size, stride, padding))
        return output_size

    def _compute_input_shape(
        self,
        edge_output_dim: int,
        kernel_size: int,
        edge_stride: int,
        edge_padding: int,
        dilation: int = 1,
        out_pad: int = 0,
    ) -> int:
        """compute required input size by computing inverse function of the
        following equation.

        H_out = (H_in - 1) * stride - 2 * padding + dilation*(kernel_size - 1) + output_padding + 1
        See https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        `math.ceil()` is used to make the output size larger than the required size.

        Args:
            edge_output_dim (int): Correspond to H_out in above equation.
            kernel_size (int): Size of the convolving kernel.
            edge_stride (int): Stride width of convolution.
            edge_padding (int): Padding size of convolution.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            out_pad (int, optional): Additional size added to one side of each dimension in the output shape. Default: 0 Defaults to 0.

        Returns:
            int : Required input size. Correspond to H_in in above equation.
        """
        return math.ceil(
            (edge_output_dim - 1 - out_pad - dilation * (kernel_size - 1) + 2 * edge_padding) / edge_stride + 1
        )

    def forward(self, x: Tensor) -> Tensor:
        no_batch = x.ndim == 1
        if no_batch:
            x = x.unsqueeze(0)
        x = self.fc_init(x)
        x = x.view(-1, 1, self.init_output_size[0], self.init_output_size[1])
        x = self.nl(self.bn1(self.deconv1(x)))
        x = self.nl(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        x = self.center_crop(x)
        if self.bias is not None:
            x = x + self.bias
        if no_batch:
            x = x.squeeze(0)
        return x
