"""Classification components
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    # Flatten -> FC(25088->4096) -> BN -> ReLU -> Dropout -> FC(4096->4096) -> BN -> ReLU -> Dropout -> FC(4096->num_classes)

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels) # outputs [B, 512, H/32, W/32]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)), # pool to fixed [B, 512, 7, 7] regardless of input size
            nn.Flatten(), # [B, 512, 7, 7] -> [B, 25088]
            nn.Linear(512 * 7 * 7, 4096), # maps 25088 conv features -> 4096 dense units
            nn.BatchNorm1d(4096), # normalize before ReLU to reduce covariate shift
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes) # output raw logits for each class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x) # [B, 3, H, W] -> [B, 512, H/32, W/32], returns bottleneck only
        logits = self.classifier(features) # pass features through FC head -> [B, num_classes] raw logits
        return logits