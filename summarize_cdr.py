# summarize_cdr.py
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    outputs = root / "outputs"

    all_cdrs = []
    for i in range(5):
        fold_dir = outputs / f"fold_{i}"
        cdr_file = fold_dir / "val_cdrs.json"
        if not cdr_file.exists():
            print(f"[WARN] {cdr_file} missing, skipping.")
            continue

        data = json.load(open(cdr_file))
        cdrs = [d["cdr"] for d in data]
        all_cdrs.extend(cdrs)

        print(f"Fold {i}: {len(cdrs)} samples, mean CDR={np.mean(cdrs):.3f}, std={np.std(cdrs):.3f}")

    if all_cdrs:
        print("\n=== Overall (all folds combined) ===")
        all_cdrs = np.array(all_cdrs)
        print(f"Total: {len(all_cdrs)} samples")
        print(f"Mean CDR: {all_cdrs.mean():.3f}")
        print(f"Std CDR:  {all_cdrs.std():.3f}")


if __name__ == "__main__":
    main()
