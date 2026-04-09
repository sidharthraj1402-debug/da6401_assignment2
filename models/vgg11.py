"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # block 1 - 1 conv layer, 64 channels, batch norm, relu, max pool
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), # 3x3 conv with padding 1 to keep the spatial dimensions the same and output 64 channels
            nn.BatchNorm2d(64), # batch normalization to stabilize training and improve convergence
            nn.ReLU(inplace=True), # relu activation function to introduce non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2) # max pooling with kernel size 2 and stride 2 to downsample the feature maps by a factor of 2
        )

        # block 2 - 1 conv layer, 128 channels, batch norm, relu, max pool
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 3 - 2 conv layers, 256 channels, batch norm, relu, max pool
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 4 - 2 conv layers, 512 channels, batch norm, relu, max pool
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 5 - 2 conv layers, 512 channels, batch norm, relu, max pool
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # x — input image batch [B, 3, H, W]
        s1 = self.block1(x)  # [B, 64, H/2, W/2] 3 to 64 channels, downsampled by 2
        s2 = self.block2(s1)  # [B, 128, H/4, W/4] 64 to 128 channels, downsampled by 2
        s3 = self.block3(s2)  # [B, 256, H/8, W/8] 128 to 256 channels, downsampled by 2
        s4 = self.block4(s3)  # [B, 512, H/16, W/16] 256 to 512 channels, downsampled by 2
        s5 = self.block5(s4)  # [B, 512, H/32, W/32] 512 to 512 channels, downsampled by 2

        if return_features:
            features = {
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "s4": s4,
                "s5": s5
            }
            return s5, features # (bottleneck, skip dict) — used by U-Net decoder

        return s5 # bottleneck only — used by classifier and localizer
    
VGG11 = VGG11Encoder
# Feature intuition: s1=edges/textures, s2=corners/shapes, s3=object parts, s4=whole objects, s5=abstract semantics

# Batch Normalization Placement
# BatchNorm2d after every Conv2d (before ReLU), BatchNorm1d after every Linear (before ReLU).
# Justification: BN normalizes activations before ReLU, reducing internal covariate shift and letting
# ReLU see the full pre-activation range. Conv -> BN -> ReLU converges faster than post-ReLU normalization.

# Dropout Placement
# CustomDropout is applied only in the FC classification head, not in the conv backbone.
# Justification: FC layers hold ~120M parameters (25088×4096 + 4096×4096) and are the main source of
# overfitting. Dropout forces redundant representations by training a different subnetwork each step.
# Conv layers are omitted because weight sharing limits overfitting, and spatial correlations make
# dropped values easy to infer from neighbors. p=0.5 balances regularization strength and signal retention.