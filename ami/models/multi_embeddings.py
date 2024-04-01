import torch
import torch.nn as nn
from torch import Tensor


class MultiEmbeddings(nn.Module):
    """Convert discrete actions to embedding vectors."""

    def __init__(self, choices_per_category: list[int], embedding_dim: int, do_flatten: bool = False) -> None:
        """Constructs Multi-Embedding class.

        Args:
            choices_per_category: A list of choice size per category.
            embedding_dim: The length of embedding vector.
        """
        super().__init__()

        self.do_flatten = do_flatten
        self.embeds = nn.ModuleList()
        for choice in choices_per_category:
            self.embeds.append(nn.Embedding(choice, embedding_dim))

    @property
    def choices_per_category(self) -> list[int]:
        return [e.num_embeddings for e in self.embeds]

    def forward(self, input: Tensor) -> Tensor:
        """
        Shape:
            input: (*, num_category). `num_category` equals to `len(choices_per_category)`.
            return: (*, num_category * embedding_dim) if do_flatten else (*, num_category, embedding_dim)
        """
        batch_list = []
        for (layer, tensor) in zip(self.embeds, input.movedim(-1, 0)):
            batch_list.append(layer(tensor))

        output = torch.stack(batch_list, dim=-2)
        if self.do_flatten:
            output = output.reshape(*output.shape[:-2], output.shape[-2] * output.shape[-1])
        return output
