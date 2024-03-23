import torch
import torch.nn as nn
from torch import Tensor


class StackedHiddenState(nn.Module):
    def __init__(self, module_list: nn.ModuleList):
        super().__init__()
        self.module_list = module_list

    # (batch, len, dim), (batch, depth, dim) -> (batch, len, dim), (batch, depth, len, dim)
    def forward(self, x: Tensor, hidden_stack: Tensor) -> tuple[Tensor, Tensor]:
        hidden_out_list = []
        for i, module in enumerate(self.module_list):
            x, hidden_out = module(x, hidden_stack[:, i, :])
            hidden_out_list.append(hidden_out)
        return x, torch.stack(hidden_out_list).transpose(1, 0)
