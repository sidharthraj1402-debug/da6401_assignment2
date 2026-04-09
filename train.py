"""Training entrypoint
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score
import shutil

from data.pets_dataset import OxfordIIITPetDataset
from torch.utils.data import Subset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

def get_args():
    parser = argparse.ArgumentParser(description="Train multi-task model on Oxford-IIIT Pet dataset.")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "localization", "segmentation"], help="Task to train: 'classification', 'localization', or 'segmentation'.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability for the models.")
    parser.add_argument("--use_bn",type=str, default="True", choices=["True", "False"], help="Whether to use batch normalization in the models.")
    parser.add_argument("--freeze_strategy", type=str, default="full", choices=["frozen", "partial", "full"],help="Strategy for freezing encoder weights during training: 'frozen' (freeze all encoder layers), 'partial' (freeze some encoder layers), 'full' (fine-tune all encoder layers).")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints.")
    return parser.parse_args()

def train_classifier(args):
    use_bn = args.use_bn == "True" # convert string argument to boolean

    wandb.init(project="da6401-assignment-2", name=f"classifier_bn={use_bn}_dropout={args.dropout_p}", config=vars(args))
    train_dataset = OxfordIIITPetDataset(root=args.data_root, split="trainval", augment=True) # trainval split for training
    val_dataset = OxfordIIITPetDataset(root=args.data_root, split="test", augment=False) # test split for validation

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,pin_memory=True) # shuffle for better generalization
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,pin_memory=True) # no shuffle for val

    model = VGG11Classifier(num_classes=37, in_channels=3, dropout_p=args.dropout_p) # 37 different breeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available, otherwise use CPU
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay) # Adam with weight decay
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6) # cosine annealing LR scheduler

    best_f1 = 0.0
    for epoch in range(args.num_epochs):

        #training
        model.train() # sets dropout/BN to train mode
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad() # clear gradients
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item() * images.size(0) # accumulate total loss
        train_loss = train_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss = val_loss + loss.item() * images.size(0)

                predicted = logits.argmax(dim=1) # predicted class index per sample
                correct = correct + (predicted == labels).sum().item() # count correct predictions
                total = total + labels.size(0) # track total samples

                all_preds.extend(predicted.cpu().numpy()) # collect preds for F1
                all_labels.extend(labels.cpu().numpy()) # collect labels for F1

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_f1 = f1_score(all_labels, all_preds, average="macro",zero_division=0) # macro F1 — averages F1 equally across all 37 classes
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "epoch": epoch+1})
        
        scheduler.step() # decay LR following cosine curve each epoch
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch+1,
                "best_metric": best_f1
             }, os.path.join(args.save_dir, "classifier_best.pth"))
            shutil.copy(os.path.join(args.save_dir, "classifier_best.pth"), "/content/drive/MyDrive/da6401_checkpoints/classifier_best.pth")
            print(f"Checkpoint saved to Drive at epoch {epoch+1}")

    wandb.finish()

# localization
def train_localizer(args):
    wandb.init(project="da6401-assignment-2", name=f"localizer_lr={args.learning_rate}", config=vars(args))

    full_train = OxfordIIITPetDataset(root=args.data_root, split="trainval", augment=True)
    full_val = OxfordIIITPetDataset(root=args.data_root, split="trainval", augment=False)

    # only keep images that have XML bounding box annotations
    xml_dir = os.path.join(args.data_root, "annotations", "xmls")
    valid_indices = [i for i, img_id in enumerate(full_train.image_ids)
                     if os.path.exists(os.path.join(xml_dir, f"{img_id}.xml"))]

    # use all XML images for training, keep 10% for validation to track metrics
    train_size = int(0.9 * len(valid_indices))
    train_indices = valid_indices[:train_size]
    val_indices = valid_indices[train_size:]

    train_dataset = Subset(full_train, train_indices)
    val_dataset = Subset(full_val, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=2,pin_memory=True)
    
    model = VGG11Localizer(in_channels=3, dropout_p=args.dropout_p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # load pretrained encoder weights from classifier checkpoint
    classifier_ckpt = torch.load(os.path.join(args.save_dir, "classifier_best.pth"), map_location=device)
    classifier_state = classifier_ckpt["state_dict"]
    encoder_state = {k.replace("encoder.", ""): v for k, v in classifier_state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state)
    for param in model.encoder.parameters():
        param.requires_grad = False # freeze encoder, only train regression head

    mse_criterion = nn.MSELoss() # penalizes coordinate-wise distance between predicted and target bbox
    iou_criterion = IoULoss(reduction="mean") # overlap-based loss, combined with MSE for better bbox quality
    iou_criterion_no_reduction = IoULoss(reduction="none") # per-sample IoU for tracking val metrics
    optimizer = torch.optim.Adam(model.regression_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) # only train the regression head, encoder is frozen
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_iou = 0.0

    for epoch in range(args.num_epochs):
        #training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            preds = model(images)
            
            loss = mse_criterion(preds, bboxes) + iou_criterion(preds, bboxes) # combine MSE loss and IoU loss for better localization performance
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        #evaluation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)

                preds = model(images)
                loss = mse_criterion(preds, bboxes) + iou_criterion(preds, bboxes)
                val_loss = val_loss + loss.item() * images.size(0)

                iou = 1 - iou_criterion_no_reduction(preds, bboxes)
                val_iou = val_iou + iou.sum().item()
                total = total + images.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_iou = val_iou / total

        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou, "epoch": epoch+1})

        scheduler.step(val_loss)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch+1,
                "best_metric": best_iou
                }, os.path.join(args.save_dir, "localizer_best.pth"))
            shutil.copy(os.path.join(args.save_dir, "localizer_best.pth"), "/content/drive/MyDrive/da6401_checkpoints/localizer_best.pth")
            print(f"Checkpoint saved to Drive at epoch {epoch+1}")
        
    wandb.finish()

def train_segmentation(args):

    wandb.init(project="da6401-assignment-2", name=f"segmentation_freeze={args.freeze_strategy}", config=vars(args))

    train_dataset = OxfordIIITPetDataset(root=args.data_root, split="trainval", augment=True)
    val_dataset = OxfordIIITPetDataset(root=args.data_root, split="test", augment=False)    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,pin_memory=True) 

    model = VGG11UNet(num_classes=3, in_channels=3, dropout_p=args.dropout_p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # load pretrained encoder weights from classifier checkpoint
    classifier_ckpt = torch.load(os.path.join(args.save_dir, "classifier_best.pth"), map_location=device)
    classifier_state = classifier_ckpt["state_dict"]
    encoder_state = {k.replace("encoder.", ""): v for k, v in classifier_state.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state)

    if args.freeze_strategy == "frozen": # freeze all encoder layers
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.freeze_strategy == "partial": # freeze first 3 blocks, fine-tune rest
        for param in model.encoder.block1.parameters():
            param.requires_grad = False
        for param in model.encoder.block2.parameters():
            param.requires_grad = False
        for param in model.encoder.block3.parameters():
            param.requires_grad = False
    elif args.freeze_strategy == "full": # fine-tune entire encoder
        for param in model.encoder.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Invalid freeze strategy: {args.freeze_strategy}. Must be one of 'frozen', 'partial', or 'full'.")    
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay) # only update trainable params based on freeze strategy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_dice = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss = val_loss + loss.item() * images.size(0)

                # compute per-class Dice score: 2*intersection / (pred + target)
                preds = logits.argmax(dim=1) # argmax over class dim to get predicted mask
                for c in range(3): # pet=0, bg=1, border=2
                    pred_c = (preds == c).float() # binary mask for class c predictions
                    target_c = (masks == c).float() # binary mask for class c ground truth
                    intersection = (pred_c * target_c).sum().item() # overlapping pixels
                    union = pred_c.sum().item() + target_c.sum().item() # total pixels
                    dice = (2 * intersection) / (union + 1e-8) # eps avoids division by zero
                    val_dice = val_dice + dice # accumulate across classes and batches
                total = total + images.size(0) # track total samples

        val_loss = val_loss / len(val_loader.dataset)
        val_dice = val_dice / (len(val_loader) * 3) # average the Dice coefficient across all batches and 3 classes

        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
                
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice, "epoch": epoch+1})

        scheduler.step(val_loss)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch+1,
                "best_metric": best_dice
                }, os.path.join(args.save_dir, "segmentation_best.pth"))    
            shutil.copy(os.path.join(args.save_dir, "segmentation_best.pth"), "/content/drive/MyDrive/da6401_checkpoints/segmentation_best.pth")
            print(f"Checkpoint saved to Drive at epoch {epoch+1}")
    
    wandb.finish()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.task == "classification":
        train_classifier(args)
    elif args.task == "localization":
        train_localizer(args)
    elif args.task == "segmentation":
        train_segmentation(args)
    else:
        raise ValueError(f"Invalid task: {args.task}. Must be one of 'classification', 'localization', or 'segmentation'.") 