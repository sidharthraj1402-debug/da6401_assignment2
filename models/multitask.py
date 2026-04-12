"""Unified multi-task model
"""

import torch
import torch.nn as nn
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
import gdown

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()   

        gdown.download(id="1TuE3_lKjVkw2KfuKcz33r84qZzP1wLw2",output=classifier_path,quiet=False) 
        gdown.download(id="1wslZpQL3IKWeEyaMO6h_66nKbeyAXbZ8",output=localizer_path,quiet=False)
        gdown.download(id="1X5-ss2SHHRjyJS3WTdrejaoB-U8eUXM4",output=unet_path,quiet=False)
        
        classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels) 
        localizer = VGG11Localizer(in_channels=in_channels)
        unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load the trained weights for each head
        def load_weights(model, path):
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
            model.load_state_dict(state)  
        
        load_weights(classifier, classifier_path)
        load_weights(localizer, localizer_path)
        load_weights(unet, unet_path)

        # the encoder from the classifier as the shared backbone for all three tasks
        self.encoder = classifier.encoder

        self.classifier_head = classifier.classifier # classifier 
        self.localizer_head = localizer.regression_head # localizer 
        self.image_size = localizer.image_size # needed to scale sigmoid output (0,1) -> pixel coords (0,224)
       
        # unet decoder for segmnentation
        self.upsample5 = unet.upsample5 
        self.dec5 = unet.dec5
        self.upsample4 = unet.upsample4
        self.dec4 = unet.dec4
        self.upsample3 = unet.upsample3
        self.dec3 = unet.dec3
        self.upsample2 = unet.upsample2
        self.dec2 = unet.dec2
        self.upsample1 = unet.upsample1
        self.dec1 = unet.dec1
        self.segmentation_output = unet.output

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # shared encoder runs once; bottleneck feeds all three heads, skips feed the U-Net decoder
        bottleneck, skips = self.encoder(x, return_features=True)
        classification_logits = self.classifier_head(bottleneck) # bottleneck -> [B, num_breeds] class logits
        localization_output = self.localizer_head(bottleneck) * self.image_size # sigmoid (0,1) -> pixel coords (0,224)
         
        # U-Net decoder: upsample bottleneck and fuse skip connections at each scale
        x = self.upsample5(bottleneck)
        x = self.dec5(torch.cat([x, skips['s4']], dim=1))
        x = self.upsample4(x)
        x = self.dec4(torch.cat([x, skips['s3']], dim=1))
        x = self.upsample3(x)
        x = self.dec3(torch.cat([x, skips['s2']], dim=1))
        x = self.upsample2(x)
        x = self.dec2(torch.cat([x, skips['s1']], dim=1))
        x = self.upsample1(x)
        x = self.dec1(x)
        segmentation_logits = self.segmentation_output(x)   

        return {
            "classification": classification_logits,
            "localization": localization_output,
            "segmentation": segmentation_logits
        }

