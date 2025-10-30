import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import load_data_2D
from modules import UNetImproved

# Config
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()
    else "cpu"
)
NUM_CLASSES = 5
MODEL_PATH = "model_final.pth"
DATA_DIR = "/Users/ivan/Assignments/keras_slices_data"
OUT_DIR = "predictions"
os.makedirs(OUT_DIR, exist_ok=True)

def get_paths(subdir):
    folder = os.path.join(DATA_DIR, subdir)
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

test_img_paths = get_paths("keras_slices_test")
test_mask_paths = get_paths("keras_slices_seg_test")

# Load model
model = UNetImproved(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# Load test data
images = load_data_2D(test_img_paths, normImage=True, categorical=False)
masks = load_data_2D(test_mask_paths, categorical=True)
print("Loaded test data:", images.shape, masks.shape)

@torch.no_grad()
def dice_per_class(pred_logits, target_onehot, num_classes=5, eps=1e-6):
    pred = torch.argmax(pred_logits, dim=1)
    target = torch.argmax(target_onehot, dim=3)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        dice = (2 * inter + eps) / (p.sum() + t.sum() + eps)
        dices.append(dice.item())
    return dices

indices = range(len(images))  # evaluate all slices
sample_indices = [10, 300, 480, 521]
all_dice = np.zeros(NUM_CLASSES)

for idx in indices:
    img = images[idx]
    gt_mask = masks[idx]
    gt_mask_argmax = np.argmax(gt_mask, axis=-1)

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    gt_tensor = torch.tensor(gt_mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        dices = dice_per_class(logits, gt_tensor, NUM_CLASSES)
        all_dice += np.array(dices)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    # save visualizations for selected slices
    if idx in sample_indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input Image"); axes[0].axis("off")
        axes[1].imshow(gt_mask_argmax, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title(f"Prediction\nDice: {np.mean(dices):.3f}")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/pred_{idx}.png", dpi=200)
        plt.close()


mean_dice_per_class = all_dice / len(indices)
mean_dice_all = mean_dice_per_class.mean()

print("\nPer-class Dice on full test set:")
for c, d in enumerate(mean_dice_per_class):
    print(f"C{c}: {d:.4f}")
print(f"\nMean Dice (all classes): {mean_dice_all:.4f}")

