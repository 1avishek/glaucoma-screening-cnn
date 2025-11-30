import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold


class FundusSegmentationDataset(Dataset):
    """
    Dataset for segmentation of optic disc & cup.

    Expects REFUGE to be laid out in ONE of these ways:

      1) Pooled square images:
         REFUGE/Images_Square
         REFUGE/Masks_Square

      2) Split by train:
         REFUGE/train/Images
         REFUGE/train/Masks

    File names must match between images and masks.
    """

    def __init__(self, img_paths: List[Path], mask_paths: List[Path], transform=None):
        assert len(img_paths) == len(mask_paths), "Image/mask count mismatch"
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # single-channel mask

        image = np.array(image)
        mask = np.array(mask, dtype=np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
            else:
                mask = mask.long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        image_id = img_path.stem
        return image, mask, image_id


IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find_matching_mask(mask_dir: Path, stem: str) -> Path:
    for ext in MASK_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _collect_images_masks(img_dir: Path, mask_dir: Path):
    """Helper: collect matching image/mask pairs if dir exists & not empty."""
    if not img_dir.is_dir() or not mask_dir.is_dir():
        return [], []

    # Allow png/jpg/jpeg
    img_paths = []
    for ext in IMAGE_EXTS:
        img_paths.extend(img_dir.glob(ext))
    img_paths = sorted(img_paths)

    if not img_paths:
        return [], []

    missing = []
    mask_paths = []
    for img_path in img_paths:
        mask_path = _find_matching_mask(mask_dir, img_path.stem)
        if mask_path is None:
            missing.append(img_path.name)
        else:
            mask_paths.append(mask_path)

    if missing:
        sample = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f", ... +{len(missing)-5} more"
        raise FileNotFoundError(
            "Masks missing for images: "
            f"{sample}{more}. Looked under {mask_dir} with extensions {MASK_EXTS}"
        )

    return img_paths, mask_paths


def get_refuge_file_lists(project_root: str) -> Tuple[List[Path], List[Path]]:
    """
    Try multiple plausible REFUGE layouts and pick the first that exists
    and contains images.
    """
    root = Path(project_root)
    refuge_root = root / "REFUGE"

    candidates = [
        # Pooled square images (1200)
        (refuge_root / "Images_Square", refuge_root / "Masks_Square", "REFUGE/Images_Square + Masks_Square"),
        # Train split (400)
        (refuge_root / "train" / "Images", refuge_root / "train" / "Masks", "REFUGE/train/Images + Masks"),
        # Train square variant if present
        (refuge_root / "train" / "Images_Square", refuge_root / "train" / "Masks_Square", "REFUGE/train/Images_Square + Masks_Square"),
    ]

    for img_dir, mask_dir, label in candidates:
        img_paths, mask_paths = _collect_images_masks(img_dir, mask_dir)
        if img_paths:
            print(f"[REFUGE] Using {label}: found {len(img_paths)} images")
            return img_paths, mask_paths

    raise FileNotFoundError(
        "Could not find REFUGE image/mask folders.\n"
        "Tried:\n"
        f"  - {refuge_root / 'Images_Square'} & {refuge_root / 'Masks_Square'}\n"
        f"  - {refuge_root / 'train' / 'Images'} & {refuge_root / 'train' / 'Masks'}\n"
        f"  - {refuge_root / 'train' / 'Images_Square'} & {refuge_root / 'train' / 'Masks_Square'}\n"
        "Check that your REFUGE folder is correctly extracted under project_root and filenames are images."
    )


def get_k_fold_splits(
    img_paths: List[Path],
    mask_paths: List[Path],
    num_folds: int,
    fold_index: int,
    seed: int = 42,
):
    assert 0 <= fold_index < num_folds, "fold_index out of range"

    n = len(img_paths)
    if n == 0:
        raise RuntimeError("No samples available for K-fold split.")

    from numpy.random import seed as np_seed
    import numpy as np

    np_seed(seed)
    indices = np.arange(n)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    train_idx, val_idx = None, None
    for i, (tr, va) in enumerate(kf.split(indices)):
        if i == fold_index:
            train_idx, val_idx = tr, va
            break

    train_imgs = [img_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]

    val_imgs = [img_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    print(
        f"[KFold] Fold {fold_index}/{num_folds} -> "
        f"train={len(train_imgs)}, val={len(val_imgs)}"
    )

    return train_imgs, train_masks, val_imgs, val_masks
