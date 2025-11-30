from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    FundusSegmentationDataset,
    get_k_fold_splits,
    get_refuge_file_lists,
)
from model import build_model
from utils import dice_score


def evaluate_fold(project_root: Path, fold_idx: int, device: torch.device):
    print(f"\n=== Evaluating fold {fold_idx} ===")
    outputs_dir = project_root / "outputs" / f"fold_{fold_idx}"
    ckpt = outputs_dir / "best_model.pth"
    if not ckpt.exists():
        print(f"[WARN] Missing checkpoint: {ckpt}")
        return None

    img_paths, mask_paths = get_refuge_file_lists(str(project_root))
    _, _, val_imgs, val_masks = get_k_fold_splits(
        img_paths, mask_paths, num_folds=5, fold_index=fold_idx, seed=42
    )

    tf = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    val_ds = FundusSegmentationDataset(val_imgs, val_masks, tf)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    model = build_model("unet", num_classes=3).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dice_sums = {1: 0.0, 2: 0.0}
    n_batches = 0

    with torch.no_grad():
        for img, mask, _ in val_loader:
            img = img.to(device)
            mask = mask.to(device)

            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            dices = dice_score(pred, mask, num_classes=3)
            for cls in dice_sums:
                dice_sums[cls] += dices[cls].item()
            n_batches += 1

    if n_batches == 0:
        print("[WARN] No validation batches?")
        return None

    disc_dice = dice_sums[1] / n_batches
    cup_dice = dice_sums[2] / n_batches
    mean_dice = (disc_dice + cup_dice) / 2.0

    print(
        "Fold {idx}: Disc Dice={disc:.4f}, Cup Dice={cup:.4f}, Mean={mean:.4f}".format(
            idx=fold_idx, disc=disc_dice, cup=cup_dice, mean=mean_dice
        )
    )
    return disc_dice, cup_dice, mean_dice


def main():
    project_root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fold_results = []
    for i in range(5):
        result = evaluate_fold(project_root, i, device)
        if result is not None:
            fold_results.append(result)

    if not fold_results:
        print("No folds evaluated.")
        return

    fold_results = np.array(fold_results)
    disc_mean, cup_mean, overall_mean = fold_results.mean(axis=0)
    disc_std, cup_std, overall_std = fold_results.std(axis=0)

    print("\n=== 5-fold Segmentation Summary ===")
    print(f"Disc Dice: mean={disc_mean:.4f}, std={disc_std:.4f}")
    print(f"Cup  Dice: mean={cup_mean:.4f}, std={cup_std:.4f}")
    print(f"Mean Dice: mean={overall_mean:.4f}, std={overall_std:.4f}")


if __name__ == "__main__":
    main()
