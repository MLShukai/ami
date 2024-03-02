from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def _identity(tensor: Tensor) -> Tensor:
    return tensor


class SmallConvNet(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        channels: int,
        dim_out: int,
        do_batchnorm: bool = False,
        do_layernorm: bool = False,
        nl: Callable[[Tensor], Tensor] = nn.LeakyReLU(negative_slope=0.2),
        last_nl: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Construct small conv net.

        Args:
            height (int): height of pictured frame.
            width (int): width of pictured frame.
            channels (int): channels of pictured frame.
            dim_out (int): The number of dimensions of the output tensor.
            do_batchnorm(bool, optional): Whether to do batchnorm. Defaults to False. https://github.com/openai/large-scale-curiosity/blob/master/utils.py#L133
            do_layernorm (bool, optional): Whether to do layernorm. Defaults to False. https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L7
            nl (Optional[nn.Module], optional): NonLinear function for activation. Defaults to nn.LeakyReLU(). https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L39
                                                                                    The value of "negative slope" follows to the default value of tf.nn.leaky_relu(https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu)
            last_nl (Optional[nn.Module], optional): NonLinearFunction for activation for the last layer. Defaults to None. https://github.com/openai/large-scale-curiosity/blob/master/auxiliary_tasks.py#L46
        """
        super().__init__()
        batch_norm_cls = nn.BatchNorm2d if do_batchnorm else nn.Identity

        self.conv2d1 = nn.Conv2d(channels, 32, 8, stride=4)
        self.bn1 = batch_norm_cls(32)
        self.conv2d2 = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = batch_norm_cls(64)
        self.conv2d3 = nn.Conv2d(64, 64, 3, stride=1)
        self.bn3 = batch_norm_cls(64)
        self.fc = nn.Linear(
            ((((height - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * ((((width - (8 - 4)) // 4 - (4 - 2)) // 2 - (3 - 1)) // 1)
            * 64,
            dim_out,
        )
        self.nl = nl
        if last_nl is None:
            last_nl = _identity
        self.last_nl = last_nl
        self.do_batchnorm = do_batchnorm
        self.layernorm = nn.LayerNorm(dim_out) if do_layernorm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if no_batch := (not has_batch_axis(x)):
            x = x.unsqueeze(0)
        x = self.bn1(self.conv2d1(x))
        x = self.nl(x)
        x = self.bn2(self.conv2d2(x))
        x = self.nl(x)
        x = self.bn3(self.conv2d3(x))
        x = self.nl(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.last_nl(x)
        x = self.layernorm(x)
        if no_batch:
            x = x.squeeze(0)
        return x


def has_batch_axis(input: Tensor) -> bool:
    """Checks whether the input image tensor has batch axis or not."""
    match input.dim():
        case 4:  # (Batch, Channels, Height, Width)
            return True
        case 3:  # (Channels, Height, Width)
            return False
        case _:
            raise ValueError(f"The dimension of image tensor must be 3 or 4! Input: {input.dim()}")
