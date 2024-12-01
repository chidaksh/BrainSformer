import glob
import gzip
import os.path
import shutil
from glob import glob

import nibabel as nib
import numpy as np
from numpy import ndarray
from tqdm import tqdm


def calc_z_score(img: ndarray) -> ndarray:
    """
    Standardize the image data using the zscore (z = (x-μ)/σ).

    :param img: Image data with shape components of (width, height, depth).
    :return: standardized image.
    """
    avg_pixel_value = np.sum(img) / np.count_nonzero(img)
    sd_pixel_value = np.std(img[np.nonzero(img)])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i, j, k] != 0:
                    img[i, j, k] = (img[i, j, k] - avg_pixel_value) / sd_pixel_value

    return img


def change_mask_shape(mask: ndarray):
    """
    Reshape mask to dimensions of 128 x 128 x 128 x 4.

    :param mask: Mask data to reshape.
    :return: reshaped mask.
    """
    if mask.shape == (128, 128, 128, 4):
        raise ValueError(
            f"Mask shape is already (128, 128, 128, 4)")

    new_mask = np.zeros((128, 128, 128, 4))
    for i in range(128):
        for j in range(128):
            for k in range(128):
                new_mask[i, j, k, mask[i, j, k]] = 1

    return new_mask


def normalize_mri_data(t1: ndarray, t1ce: ndarray, t2: ndarray, flair: ndarray, mask: ndarray) \
        -> tuple[ndarray, ndarray]:
    """
    Normalize the MRI data from the dataset using the z-score of the MRI data.

    :param t1: T1-Weighted MRI data.
    :param t1ce: T1-Weighted Contrast Enhanced MRI data.
    :param t2: T2-Weighted MRI data.
    :param flair: Flair MRI data.
    :param mask: Segmented mask data.
    :return: Stacked MRI data and segmented mask data.
    """
    t2 = t2[56:184, 56:184, 13:141]
    t2 = t2.reshape(-1, t2.shape[-1]).reshape(t2.shape)
    t2 = calc_z_score(t2)
    t1ce = t1ce[56:184, 56:184, 13:141]
    t1ce = t1ce.reshape(-1, t1ce.shape[-1]).reshape(t1ce.shape)
    t1ce = calc_z_score(t1ce)

    flair = flair[56:184, 56:184, 13:141]
    flair = flair.reshape(-1, flair.shape[-1]).reshape(flair.shape)
    flair = calc_z_score(flair)

    t1 = t1[56:184, 56:184, 13:141]
    t1 = t1.reshape(-1, t1.shape[-1]).reshape(t1.shape)
    t1 = calc_z_score(t1)

    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3
    mask = mask[56:184, 56:184, 13:141]

    data = np.stack([flair, t1ce, t1, t2], axis=3)

    mask = change_mask_shape(mask)

    return data, mask


def get_mri_data_from_directory(patient_directory: str, t1: str, t1ce: str, t2: str, flair: str, mask: str) \
        -> type[ndarray, ndarray]:
    """
    Load MRI data from .nii files in a patient directory.

    :param patient_directory: parent patient directory.
    :param t1: t1 .nii file.
    :param t1ce: t1ce .nii file.
    :param t2: t2 .nii file.
    :param flair: flair .nii file.
    :param mask: mask .nii file.
    :return: normalized MRI data (stacked MRI data, mask data).
    """
    t1_data = nib.load(os.path.join(patient_directory, t1)).get_fdata()
    t1ce_data = nib.load(os.path.join(patient_directory, t1ce)).get_fdata()
    t2_data = nib.load(os.path.join(patient_directory, t2)).get_fdata()
    flair_data = nib.load(os.path.join(patient_directory, flair)).get_fdata()
    mask_data = nib.load(os.path.join(patient_directory, mask)).get_fdata()

    return normalize_mri_data(t1_data, t1ce_data, t2_data, flair_data, mask_data)