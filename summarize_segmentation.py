import re
from pathlib import Path
import numpy as np

def extract_best_dice(log_file):
    pattern = r"Val dice=([0-9.]+)"
    dice_scores = []
    for line in open(log_file):
        m = re.search(pattern, line)
        if m:
            dice_scores.append(float(m.group(1)))
    return max(dice_scores) if dice_scores else None

def main():
    root = Path("outputs")
    fold_scores = []

    for i in range(5):
        log_path = root / f"fold_{i}" / "train_log.txt"
        if not log_path.exists():
            print(f"[WARN] Missing log for fold {i}")
            continue

        best = extract_best_dice(str(log_path))
        if best is not None:
            fold_scores.append(best)
            print(f"Fold {i}: Best Dice = {best:.4f}")
        else:
            print(f"[WARN] No Dice scores for fold {i}")

    if fold_scores:
        mean = np.mean(fold_scores)
        std  = np.std(fold_scores)
        print("\n=== Final Segmentation Accuracy (5-fold) ===")
        print(f"Mean Dice: {mean:.4f}")
        print(f"Std Dice:  {std:.4f}")
    else:
        print("No fold scores collected.")

if __name__ == "__main__":
    main()
