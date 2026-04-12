"""Segmentation model
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels) # outputs [B, 512, H/32, W/32]

        # decoder: ConvTranspose2d upsamples at each stage; skip connections from encoder restore spatial detail.
        # final 1x1 conv outputs raw logits — CrossEntropyLoss applies softmax internally during training.

        self.upsample5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2) # upsample s5 from [B, 512, H/32, W/32] to [B, 512, H/16, W/16]
        self.dec5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1), # concatenate with s4 from the encoder to get [B, 1024, H/16, W/16], then apply conv to get [B, 512, H/16, W/16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # upsample from [B, 512, H/16, W/16] to [B, 256, H/8, W/8]].
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), # concatenate with s3 from the encoder to get [B, 512, H/8, W/8], then apply conv to get [B, 256, H/8, W/8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # upsample from [B, 256, H/8, W/8] to [B, 128, H/4, W/4]
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), # concatenate with s2 from the encoder to get [B, 256, H/4, W/4], then apply conv to get [B, 128, H/4, W/4]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # upsample from [B, 128, H/4, W/4] to [B, 64, H/2, W/2]
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), # concatenate with s1 from the encoder to get [B, 128, H/2, W/2], then apply conv to get [B, 64, H/2, W/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) # upsample from [B, 64, H/2, W/2] to [B, 64, H, W]
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # no skip connection for the last layer, just apply conv to get [B, 64, H, W]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(64, num_classes, kernel_size=1) # final output layer to get [B, num_classes, H, W]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        s5, features = self.encoder(x, return_features=True) # get the bottleneck features and the skip connection features from the encoder
        s4 = features["s4"]
        s3 = features["s3"]
        s2 = features["s2"]
        s1 = features["s1"]

        d5 = self.upsample5(s5) # upsample s5 to get d5
        d5 = torch.cat([d5, s4], dim=1) # concatenate with s4 to get [B, 1024, H/16, W/16]
        d5 = self.dec5(d5) # apply conv to get [B, 512, H/16, W/16]

        d4 = self.upsample4(d5) # upsample d5 to get d4
        d4 = torch.cat([d4, s3], dim=1) # concatenate with s3 to get [B, 512, H/8, W/8]
        d4 = self.dec4(d4) # apply conv to get [B, 256, H/8, W/8]

        d3 = self.upsample3(d4) # upsample d4 to get d3
        d3 = torch.cat([d3, s2], dim=1) # concatenate with s2 to get [B, 256, H/4, W/4]
        d3 = self.dec3(d3) # apply conv to get [B, 128, H/4, W/4]

        d2 = self.upsample2(d3) # upsample d3 to get d2
        d2 = torch.cat([d2, s1], dim=1) # concatenate with s1 to get [B, 128, H/2, W/2]
        d2 = self.dec2(d2) # apply conv to get [B, 64, H/2, W/2]

        d1 = self.upsample1(d2) # upsample d2 to get d1
        d1 = self.dec1(d1) # apply conv to get [B, 64, H, W]

        output = self.output(d1) # apply final output layer to get [B, num_classes, H, W]
        return output
        
#Encoder 

# Input [B, 3, 224, 224]
#   -> block1 -> [B,  64, 112, 112]  <- saved as s1
#   -> block2 -> [B, 128,  56,  56]  <- saved as s2
#   -> block3 -> [B, 256,  28,  28]  <- saved as s3
#   -> block4 -> [B, 512,  14,  14]  <- saved as s4
#   -> block5 -> [B, 512,   7,   7]  <- bottleneck (s5)

# Decoder 

# upX — ConvTranspose2d doubles the spatial size (learnable upsampling)
# torch.cat — concatenates with the matching encoder skip along the channel dimension, restoring spatial detail that was lost during pooling
# decX — conv block refines the combined features

# s5  [B, 512,  7,  7]
#   -> up5  -> [B, 512, 14, 14]
#   -> cat(s4) -> [B, 1024, 14, 14]   (512 upsampled + 512 from encoder)
#   -> dec5 -> [B, 512, 14, 14]

#   -> up4  -> [B, 256, 28, 28]
#   -> cat(s3) -> [B, 512, 28, 28]    (256 + 256)
#   -> dec4 -> [B, 256, 28, 28]

#   -> up3  -> [B, 128, 56, 56]
#   -> cat(s2) -> [B, 256, 56, 56]    (128 + 128)
#   -> dec3 -> [B, 128, 56, 56]

#   -> up2  -> [B, 64, 112, 112]
#   -> cat(s1) -> [B, 128, 112, 112]  (64 + 64)
#   -> dec2 -> [B, 64, 112, 112]

#   -> up1  -> [B, 64, 224, 224]      (no skip — back to original resolution)
#   -> dec1 -> [B, 64, 224, 224]
#   -> output (1×1 conv) -> [B, 3, 224, 224]   ← per-pixel class logits


# Why skip connections matter
# Without them, the decoder only has the bottleneck [B, 512, 7, 7] to work from — extremely compressed. The skip connections inject fine spatial detail (edges, textures) from the encoder at each resolution, letting the decoder precisely localise boundaries rather than producing blurry masks.