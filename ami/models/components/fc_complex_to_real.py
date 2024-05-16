import torch
import torch.nn as nn


class FCComplexToReal(nn.Module):
    """Fully connected layer converting complex input to real output."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Args:
            in_features: Number of input features (complex).
            out_features: Number of output features (real).
        """
        super().__init__()
        self.fc = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass converting complex input to real output.

        Args:
            x: Complex input tensor of shape (*, in_features).

        Returns:
            Real output tensor of shape (*, out_features).
        """
        return self.fc(torch.view_as_real(x).flatten(-2))
