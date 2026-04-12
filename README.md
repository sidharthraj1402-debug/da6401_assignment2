# DA6401 Assignment 2 — Visual Perception Pipeline

Multi-task visual perception on the Oxford-IIIT Pet dataset using a shared VGG11 backbone.
Covers classification, bounding box localization, semantic segmentation, and a unified multi-task model.

**GitHub Repository:** https://github.com/sidharthraj1402-debug/da6401_assignment2.git

**W&B Report:** https://wandb.ai/ns26z087-indian-institute-of-technology-madras/da6401-assignment-2/reports/DA6401-Assignment2--VmlldzoxNjQ4MzI2NA?accessToken=wlf2u5u9eehoh5m1freoj6oxl2hszfh9wkp9z3d21g7hauaukdmbif1spzp7ago2

---

## Tasks

| Task | Description | Model |
|------|-------------|-------|
| 1 | Breed classification (37 classes) | VGG11 + FC head |
| 2 | Bounding box localization | VGG11 + regression head |
| 3 | Semantic segmentation (3 classes) | VGG11 U-Net |
| 4 | Unified multi-task inference | Shared backbone + 3 heads |

---

## Project Structure

```
da6401_assignment_2/
├── data/
│   └── pets_dataset.py       # Oxford-IIIT Pet dataset loader
├── models/
│   ├── layers.py             # CustomDropout
│   ├── vgg11.py              # VGG11 encoder (shared backbone)
│   ├── classification.py     # Task 1 — VGG11Classifier
│   ├── localization.py       # Task 2 — VGG11Localizer
│   ├── segmentation.py       # Task 3 — VGG11UNet
│   └── multitask.py          # Task 4 — MultiTaskPerceptionModel
├── losses/
│   └── iou_loss.py           # Custom IoU loss
├── train.py                  # Training script (all tasks)
├── inference.py              # Evaluation + single image inference
└── README.md
```

---

## Dataset Setup

Download the Oxford-IIIT Pet dataset and place it as:

```
/path/to/dataset/
├── images/
├── annotations/
│   ├── list.txt
│   ├── trainval.txt
│   ├── test.txt
│   ├── trimaps/
│   └── xmls/
```

---

## Setup

```bash
pip install torch torchvision wandb scikit-learn gdown
```

---

## Training

All three models must be trained sequentially — localization and segmentation both load the classifier encoder weights.

### Task 1 — Classification

```bash
python train.py \
  --task classification \
  --data_root /path/to/dataset \
  --batch_size 64 \
  --num_epochs 30 \
  --learning_rate 1e-4 \
  --weight_decay 5e-4 \
  --save_dir ./checkpoints
```

### Task 2 — Localization

Run after classification. Loads classifier encoder weights and freezes them, trains only the regression head.

```bash
python train.py \
  --task localization \
  --data_root /path/to/dataset \
  --batch_size 64 \
  --num_epochs 30 \
  --learning_rate 1e-4 \
  --weight_decay 5e-4 \
  --save_dir ./checkpoints
```

### Task 3 — Segmentation

Run after classification. Loads classifier encoder weights, trains the U-Net decoder on top.

```bash
python train.py \
  --task segmentation \
  --data_root /path/to/dataset \
  --batch_size 32 \
  --num_epochs 30 \
  --learning_rate 1e-4 \
  --weight_decay 5e-4 \
  --freeze_strategy frozen \
  --save_dir ./checkpoints
```

`--freeze_strategy` options: `frozen` (freeze all encoder layers), `partial` (freeze first 3 blocks), `full` (fine-tune entire encoder).

---

## Inference

### Evaluate on full dataset split

```bash
python inference.py \
  --data_root /path/to/dataset \
  --split test \
  --classifier_path classifier.pth \
  --localizer_path localizer.pth \
  --unet_path unet.pth
```

### Single image prediction

```bash
python inference.py \
  --image_path /path/to/image.jpg \
  --classifier_path classifier.pth \
  --localizer_path localizer.pth \
  --unet_path unet.pth
```

---

## Architecture

### VGG11 Encoder (shared backbone)

5 blocks of Conv → BN → ReLU → MaxPool. Output: `[B, 512, 7, 7]` for 224×224 input.

```
Input [B, 3, 224, 224]
  block1 → [B,  64, 112, 112]   (s1)
  block2 → [B, 128,  56,  56]   (s2)
  block3 → [B, 256,  28,  28]   (s3)
  block4 → [B, 512,  14,  14]   (s4)
  block5 → [B, 512,   7,   7]   (s5 / bottleneck)
```

### U-Net Decoder (Task 3)

ConvTranspose2d upsampling at each stage, skip connections concatenated from matching encoder blocks.

```
s5 → upsample5 → cat(s4) → dec5 → [B, 512, 14, 14]
   → upsample4 → cat(s3) → dec4 → [B, 256, 28, 28]
   → upsample3 → cat(s2) → dec3 → [B, 128, 56, 56]
   → upsample2 → cat(s1) → dec2 → [B,  64, 112, 112]
   → upsample1 →           dec1 → [B,  64, 224, 224]
   → output (1×1 conv)          → [B,   3, 224, 224]
```

### Multi-task Model (Task 4)

Single shared encoder (classifier backbone) feeds all three heads simultaneously. Checkpoints downloaded from Google Drive at init.

---

## Loss Functions

| Task | Loss |
|------|------|
| Classification | CrossEntropyLoss |
| Localization | MSELoss + IoULoss (combined) |
| Segmentation | CrossEntropyLoss (per-pixel) |
