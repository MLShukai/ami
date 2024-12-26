import torch
import torch.nn as nn
from torch import Tensor


class StackedHiddenState(nn.Module):
    def __init__(self, module_list: nn.ModuleList):
        super().__init__()
        self.module_list = module_list

    # (*batch, len, dim), (*batch, depth, dim) -> (*batch, len, dim), (*batch, depth, len, dim) or
    # (len, dim), (depth, dim) -> (len, dim), (depth, len, dim) or
    # (*batch, dim), (*batch, depth, dim) -> (*batch, dim), (*batch, depth, dim) or
    # (dim), (depth, dim) -> (dim), (depth, dim)
    def forward(self, x: Tensor, hidden_stack: Tensor) -> tuple[Tensor, Tensor]:
        no_batch = len(hidden_stack.shape) < 3
        if no_batch:
            x = x.unsqueeze(0)
            hidden_stack = hidden_stack.unsqueeze(0)

        no_len = len(x.shape) < 3
        if no_len:
            x = x.unsqueeze(1)

        batch_shape = x.shape[:-2]
        x = x.reshape(-1, *x.shape[len(batch_shape) :])
        hidden_stack = hidden_stack.reshape(-1, *hidden_stack.shape[len(batch_shape) :])

        hidden_out_list = []
        for i, module in enumerate(self.module_list):
            x, hidden_out = module(x, hidden_stack[:, i, :])
            hidden_out_list.append(hidden_out)

        hidden_out_stack = torch.stack(hidden_out_list).transpose(1, 0)

        x = x.view(*batch_shape, *x.shape[1:])
        hidden_out_stack = hidden_out_stack.view(*batch_shape, *hidden_out_stack.shape[1:])

        if no_len:
            x = x.squeeze(1)
            hidden_out_stack = hidden_out_stack.squeeze(2)

        if no_batch:
            x = x.squeeze(0)
            hidden_out_stack = hidden_out_stack.squeeze(0)

        return x, hidden_out_stack
