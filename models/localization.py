"""Localization modules
"""

import torch
import torch.nn as nn

from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.image_size = 224
        self.encoder = VGG11Encoder(in_channels=in_channels) # outputs [B, 512, H/32, W/32]

        self.regression_head = nn.Sequential(
            nn.Flatten(), # [B, 512, 7, 7] -> [B, 25088]
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4), # raw output: 4 values
            nn.Sigmoid(), # squash to (0, 1)
        )   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        features = self.encoder(x)
        output = self.regression_head(features) # sigmoid output in (0, 1)
        return output * self.image_size # scale to pixel space (0, 224)


# Regression head design:
# - Flatten converts the encoder's [B, 512, 7, 7] feature map into [B, 25088] for FC layers.
# - Two FC layers with ReLU and dropout learn non-linear mappings from features to bbox coords while reducing overfitting.
# - Final layer outputs 4 values (cx, cy, w, h); Sigmoid squashes them to (0,1), then scaled by image_size to pixel space.

