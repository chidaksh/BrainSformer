import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MRIDataset(Dataset):
    def __init__(self, dataset_path=None, augmentations=None, mode="train", include_augmented=True, aug_prob=0.8):
        """
        Initialize the dataset class.

        :param dataset_path: Path to the dataset directory.
        :param augmentations: Augmentation function to apply, e.g., combine_aug or binary_combine_aug.
        :param mode: Dataset mode, either "train" or "val".
        :param include_augmented: Whether to include augmented samples as extra data.
        :param aug_prob: Probability of applying augmentations in `__getitem__`.
        """
        self.include_augmented = include_augmented
        self.augmentations = augmentations
        self.aug_prob = aug_prob

        # Gather data paths
        all_image_paths = glob(os.path.join(dataset_path, mode, "image", "*.npy"))
        all_mask_paths = glob(os.path.join(dataset_path, mode, "mask", "*.npy"))

        # Pair images and masks
        self.data_pairs = list(zip(all_image_paths, all_mask_paths))

        # Duplicate data if including augmented samples
        if self.include_augmented:
            augmented_pairs = [(img, msk, True) for img, msk in self.data_pairs]
            self.data_pairs = [(img, msk, False) for img, msk in self.data_pairs] + augmented_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path, is_augmented = self.data_pairs[idx]

        # Load image and mask
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        # Apply augmentations probabilistically
        if random.random() < self.aug_prob and is_augmented:
            image, mask = self.augmentations(image, mask)

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).permute(3, 0, 1, 2)  #(C, F, H, W)
        mask = torch.tensor(mask, dtype=torch.long).permute(3, 0, 1, 2)

        return image, mask