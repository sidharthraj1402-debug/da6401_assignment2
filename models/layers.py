"""Reusable custom layers
"""

import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        # probability of dropping a neuron during training
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        # pass through unchanged during eval or if p=0
        if not self.training or self.p == 0.0:
            return x
        binary_mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p)) # 1 = keep, 0 = drop
        return x * binary_mask / (1 - self.p) # scale by 1/(1-p) to keep expected output magnitude
