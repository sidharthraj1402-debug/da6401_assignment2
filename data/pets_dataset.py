"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
import xml.etree.ElementTree as ET # built-in python xml parser to read the annotation files(bounding boxes and segmentation masks) provided in the dataset

import torch
import torchvision.io as io # reads image files directly into PyTorch as tensors
import torchvision.transforms.functional as F # used for data augmentation and preprocessing
import torchvision.transforms as T
from torch.utils.data import Dataset

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    
    def __init__(self, root: str, split: str = "trainval", augment: bool = False):
        self.size = 224
        self.root = root
        self.augment = augment

        self.mean = [0.485, 0.456, 0.406] # ImageNet mean/std for normalizing inputs
        self.std = [0.229, 0.224, 0.225]

        split_file = os.path.join(root, "annotations", f"{split}.txt") # lists image ids for the split
        with open(split_file) as f:
            self.image_ids = [line.strip().split()[0] for line in f] # e.g. "Abyssinian_1", "Bengal_10"
        self.image_ids = [
            image_id for image_id in self.image_ids
            if os.path.exists(os.path.join(root, "images", f"{image_id}.jpg")) and os.path.exists(os.path.join(root, "annotations", "trimaps", f"{image_id}.png"))
        ]

        list_files = os.path.join(root, "annotations", "list.txt") # maps image ids to class labels
        self.class_labels = {}
        with open(list_files) as f:
            for line in f:
                if line.startswith("#"): # skip the comment lines in the list.txt file
                    continue
                parts = line.strip().split()
                if parts[0] in self.image_ids: # only store labels for images in this split
                    image_id = parts[0]
                    class_label = int(parts[1]) - 1 # list.txt is 1-indexed; subtract 1 for 0-indexed labels
                    self.class_labels[image_id] = class_label # map image_id -> class index

        image_dir = os.path.join(root, "images")
        trimap_dir = os.path.join(root, "annotations", "trimaps")
        bbox_xml_dir = os.path.join(root, "annotations", "xmls")

        if os.path.exists(image_dir) and os.path.exists(trimap_dir) and os.path.exists(bbox_xml_dir):
            self.image_dir = image_dir
            self.trimap_dir = trimap_dir
            self.bbox_xml_dir = bbox_xml_dir
        else:
            raise FileNotFoundError("One or more required directories (images, trimaps, xmls) are missing in the specified root directory.")
        
    def __len__(self) -> int:
        return len(self.image_ids)
    
    # convert xml bbox to cx,cy,w,h
    def parse_xml(self, xml_path: str) -> torch.Tensor:
        """Parse the XML file to extract bounding box coordinates and convert them to (x_center, y_center, width, height) format."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # XML has an 'object/bndbox' element with xmin, ymin, xmax, ymax fields
        bndbox = root.find('.//object/bndbox')
        if bndbox is None:
            raise ValueError(f"No bounding box found in XML file: {xml_path}")
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        return torch.tensor([x_center, y_center, width, height], dtype=torch.float32)
        
    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx] # example idx = 0, image_id = "Abyssinian_1"
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        trimap_path = os.path.join(self.trimap_dir, f"{image_id}.png")
        bbox_xml_path = os.path.join(self.bbox_xml_dir, f"{image_id}.xml")

        image = io.read_image(image_path) # reads image as [C, H, W] tensor with pixel values in [0, 255]
        if image.shape[0] == 4:
            image = image[:3] # discard alpha channel, keep only RGB channels
        elif image.shape[0] == 1: # grayscale -> repeat channel 3 times to make RGB
            image = image.repeat(3, 1, 1) # [1, H, W] -> [3, H, W]
    
        trimap = io.read_image(trimap_path) # reads trimap as [1, H, W] tensor with pixel values in [0, 255]
        if os.path.exists(bbox_xml_path):
            bbox = self.parse_xml(bbox_xml_path) # returns [cx, cy, w, h] in original image pixel coords
        else:
            bbox = torch.zeros(4, dtype=torch.float32) # no annotation available, fill with zeros

        class_label = self.class_labels[image_id]

        # resize image and trimap to 224x224; scale bbox coords to match new dimensions
        original_h, original_w = image.shape[1], image.shape[2] # original dims before resizing
        image = F.resize(image, (self.size, self.size)) # [C, H, W] -> [C, 224, 224]

        scale_x = self.size / original_w # scale factor for x: 224 / original_width
        scale_y = self.size / original_h # scale factor for y: 224 / original_height

        bbox[0] = bbox[0] * scale_x # adjust x_center for new image width
        bbox[1] = bbox[1] * scale_y # adjust y_center for new image height
        bbox[2] = bbox[2] * scale_x # scaling the width
        bbox[3] = bbox[3] * scale_y # scaling the height

        image = image.float() / 255.0 # scale pixel values from [0, 255] to [0, 1]
        image = F.normalize(image, mean=self.mean, std=self.std) # apply ImageNet mean/std normalization

        trimap = F.resize(trimap, (self.size, self.size), interpolation=F.InterpolationMode.NEAREST)  # nearest neighbor to preserve integer class labels
        trimap = trimap.squeeze(0).long() - 1 # [1,224,224] -> [224,224], convert pixel values to class indices 0/1/2
        
        if self.augment and torch.rand(1).item() > 0.5: # apply random horizontal flip with a probability of 0.5
            image = F.hflip(image)
            trimap = F.hflip(trimap)
            bbox[0] = self.size - bbox[0] # mirror x_center after horizontal flip

        # random crop disabled — bbox adjustment unreliable with augmentation
        #if self.augment and torch.rand(1).item() > 0.5: # apply random crop with a probability of 0.5
            #i,j,h,w = T.RandomCrop.get_params(image, output_size=(196,196)) # get random crop parameters for cropping the image and trimap to the same random location, we will use these parameters to crop both the image and the trimap to ensure that they are still aligned after cropping.
            #image = F.crop(image, i, j, h, w) # crop the image
            #image = F.resize(image, (self.size, self.size)) # resize the cropped image back to 224x224
            #trimap = F.crop(trimap, i, j, h, w) # crop the trimap using the same parameters to ensure alignment
            #trimap = F.resize(trimap.unsqueeze(0), (self.size, self.size), interpolation=F.InterpolationMode.NEAREST).squeeze(0) # resize the cropped trimap back to 224x224 using nearest neighbor interpolation, we need to unsqueeze and squeeze the channel dimension to maintain the correct shape of the trimap after resizing.
            #scale = self.size / 196 # scale factor to adjust bbox coordinates after resizing cropped region back to 224x224
            #bbox[0] = ((bbox[0] - j) * scale).clamp(0, self.size) # adjust x_center by subtracting crop left offset j and scaling, clamp to image bounds
            #bbox[1] = ((bbox[1] - i) * scale).clamp(0, self.size) # adjust y_center by subtracting crop top offset i and scaling, clamp to image bounds
            #bbox[2] = (bbox[2] * scale).clamp(0, self.size) # scale width, clamp to image bounds
            #bbox[3] = (bbox[3] * scale).clamp(0, self.size) # scale height, clamp to image bounds

        return {
            "image": image,
            "mask": trimap,
            "bbox": bbox,
            "label": class_label
        }   
    