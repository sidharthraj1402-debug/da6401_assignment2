"""Custom IoU loss
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Reduction must be one of 'none', 'mean', or 'sum', but got {reduction}")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        
        # convert (cx, cy, w, h) to (x1, y1, x2, y2) for intersection calc
        
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # intersection is max of top-lefts, min of bottom-rights
        
        intersection_x1 = torch.max(pred_x1, target_x1)
        intersection_y1 = torch.max(pred_y1, target_y1)
        intersection_x2 = torch.min(pred_x2, target_x2)
        intersection_y2 = torch.min(pred_y2, target_y2) 

        # clamp to 0 — non-overlapping boxes give negative width/height
        intersection = (intersection_x2 - intersection_x1).clamp(min=0) * (intersection_y2 - intersection_y1).clamp(min=0)

        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1) # area of predicted box
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1) # area of target box

        union = pred_area + target_area - intersection + self.eps # add eps to avoid division by zero
    
        iou = intersection / union # IoU is the area of the intersection divided by the area of the union

        iou_loss = 1 - iou # loss = 0 when perfect overlap, 1 when no overlap
        
        if self.reduction == "mean":
            return iou_loss.mean()
        elif self.reduction == "sum":
            return iou_loss.sum()
        else:
            return iou_loss
