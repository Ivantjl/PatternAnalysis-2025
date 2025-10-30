import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import HipMRIDataset, load_data_2D
from modules import UNetImproved

# Config
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
WEIGHT_DECAY = 1e-6
NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", DEVICE)

DATA_DIR = "/Users/ivan/Assignments/keras_slices_data"

def get_paths(subdir):
    folder = os.path.join(DATA_DIR, subdir)
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

# Load data
train_img_paths = get_paths("keras_slices_train")
train_mask_paths = get_paths("keras_slices_seg_train")
val_img_paths   = get_paths("keras_slices_validate")
val_mask_paths  = get_paths("keras_slices_seg_validate")

train_images = load_data_2D(train_img_paths, normImage=True, categorical=False)
train_masks  = load_data_2D(train_mask_paths, categorical=True)
val_images   = load_data_2D(val_img_paths, normImage=True, categorical=False)
val_masks    = load_data_2D(val_mask_paths, categorical=True)

print("Loaded all data:")
print("Train:", train_images.shape, train_masks.shape)
print("Val:  ", val_images.shape, val_masks.shape)

train_loader = DataLoader(HipMRIDataset(train_images, train_masks), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(HipMRIDataset(val_images, val_masks), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model setup
model = UNetImproved(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Dice metric
@torch.no_grad()
def dice_mean(logits, target, num_classes=5, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    target = torch.argmax(target, dim=1)
    total = 0.0
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        dice = (2 * inter + eps) / (p.sum() + t.sum() + eps)
        total += dice
    return total / num_classes

# Training
train_losses, val_losses, val_dice_scores = [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)
        targets = torch.argmax(masks, dim=1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
            targets = torch.argmax(masks, dim=1)

            logits = model(images)
            vloss = criterion(logits, targets)
            val_loss += vloss.item()
            val_dice += dice_mean(logits, masks, NUM_CLASSES).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_val_dice)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} "
          f"| Val Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")

# Save model and plots
torch.save(model.state_dict(), "model_final.pth")
print("Training complete!")

plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend(); plt.title("Loss")
plt.savefig("loss_curve.png"); plt.close()

plt.figure()
plt.plot(val_dice_scores, label="Val Dice")
plt.legend(); plt.title("Dice")
plt.savefig("val_dice_curve.png"); plt.close()
