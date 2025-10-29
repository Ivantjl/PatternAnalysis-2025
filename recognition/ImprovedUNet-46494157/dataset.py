import os
import numpy as np
import nibabel as nib
import skimage.transform as skTrans
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

def to_channels(arr: np.ndarray, num_classes=5, dtype=np.uint8) -> np.ndarray:
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in range(num_classes):
        res[..., c][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, early_stop=False):
    affines = []
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]

    if categorical:
        first_case = to_channels(first_case)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames, desc="Loading NIfTI files")):
        nifti = nib.load(inName)
        img = nifti.get_fdata(caching="unchanged")

        if len(img.shape) == 3:
            img = img[:, :, 0]

        img = skTrans.resize(img, (256, 128), order=0, preserve_range=True)

        if normImage:
            mean, std = img.mean(), img.std()
            img = (img - mean) / std if std > 1e-6 else np.zeros_like(img)

        if categorical:
            img = to_channels(img, dtype=dtype)
            images[i, :, :, :] = img
        else:
            images[i, :, :] = img

        affines.append(nifti.affine)

        if early_stop and i > 20:
            break

    return (images, affines) if getAffines else images

class HipMRIDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        mask = torch.tensor(self.masks[idx], dtype=torch.float32).permute(2, 0, 1)  # [5,H,W]
        return image, mask
