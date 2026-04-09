"""Inference and evaluation
"""

import os
import argparse
import torch
import torchvision.io as io
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


def get_args():
    parser = argparse.ArgumentParser(description="Run inference with the multi-task model.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--split", type=str, default="test", choices=["trainval", "test"], help="Dataset split to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--classifier_path", type=str, default="classifier.pth", help="Path to save/load classifier weights.")
    parser.add_argument("--localizer_path", type=str, default="localizer.pth", help="Path to save/load localizer weights.")
    parser.add_argument("--unet_path", type=str, default="unet.pth", help="Path to save/load unet weights.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single image for inference. If provided, runs single image inference instead of dataset evaluation.")
    return parser.parse_args()


def evaluate(args):
    """Run evaluation on the full dataset split and print metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path
    )
    model.to(device)
    model.eval()

    dataset = OxfordIIITPetDataset(root=args.data_root, split=args.split, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    iou_criterion = IoULoss(reduction="none") # for per-sample IoU calculation

    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    iou_sum = 0.0
    iou_at_50 = 0
    iou_at_75 = 0
    dice_sum = 0.0
    dice_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            masks = batch["mask"].to(device)
            bboxes = batch["bbox"].to(device)

            outputs = model(images)

            # classification metrics
            preds = outputs["classification"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # localization metrics - only for samples with valid bbox (non-zero)
            valid = bboxes.sum(dim=1) > 0
            if valid.sum() > 0:
                pred_boxes = outputs["localization"][valid]
                target_boxes = bboxes[valid]
                iou = 1 - iou_criterion(pred_boxes, target_boxes) # convert loss to IoU
                iou_sum += iou.sum().item()
                iou_at_50 += (iou > 0.5).sum().item()
                iou_at_75 += (iou > 0.75).sum().item()

            # segmentation metrics - dice per class
            seg_preds = outputs["segmentation"].argmax(dim=1)
            for c in range(3):
                pred_c = (seg_preds == c).float()
                target_c = (masks == c).float()
                intersection = (pred_c * target_c).sum().item()
                union = pred_c.sum().item() + target_c.sum().item()
                dice = (2 * intersection) / (union + 1e-8)
                dice_sum += dice
            dice_batches += 1

    # classification
    val_acc = correct / total
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # localization
    n_valid = sum(1 for b in dataset for _ in [b] if b["bbox"].sum() > 0) if hasattr(dataset, '__iter__') else total
    mean_iou = iou_sum / total
    acc_iou_50 = iou_at_50 / total * 100
    acc_iou_75 = iou_at_75 / total * 100

    # segmentation
    mean_dice = dice_sum / (dice_batches * 3)

    print("=" * 50)
    print(f"Classification  — Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f}")
    print(f"Localization    — Mean IoU: {mean_iou:.4f} | Acc@IoU=0.5: {acc_iou_50:.1f}% | Acc@IoU=0.75: {acc_iou_75:.1f}%")
    print(f"Segmentation    — Mean Dice: {mean_dice:.4f}")
    print("=" * 50)


def predict_single(args):
    """Run inference on a single image and print predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path
    )
    model.to(device)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = io.read_image(args.image_path) # read image as [C, H, W]
    if image.shape[0] == 4:
        image = image[:3]
    elif image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image = F.resize(image, (224, 224))
    image = image.float() / 255.0
    image = F.normalize(image, mean=mean, std=std)
    image = image.unsqueeze(0).to(device) # add batch dimension [1, C, H, W]

    with torch.no_grad():
        outputs = model(image)

    pred_class = outputs["classification"].argmax(dim=1).item()
    pred_bbox = outputs["localization"][0].cpu().tolist()
    pred_mask = outputs["segmentation"].argmax(dim=1)[0].cpu() # [H, W]

    print(f"Predicted class index : {pred_class}")
    print(f"Predicted bbox (cx,cy,w,h): {[round(v, 2) for v in pred_bbox]}")
    print(f"Segmentation mask shape  : {list(pred_mask.shape)}")
    print(f"Unique mask values       : {pred_mask.unique().tolist()} (0=pet, 1=background, 2=border)")


if __name__ == "__main__":
    args = get_args()
    if args.image_path is not None:
        predict_single(args)
    else:
        evaluate(args)
