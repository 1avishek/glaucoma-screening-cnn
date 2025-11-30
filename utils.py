import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(pred, target, num_classes):
    dices = {}
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(1, num_classes):  # skip bg
        pred_c = (pred == cls).float()
        tgt_c = (target == cls).float()

        inter = (pred_c * tgt_c).sum()
        denom = pred_c.sum() + tgt_c.sum()
        dices[cls] = (2 * inter + 1e-6) / (denom + 1e-6)

    return dices


def compute_vertical_diameter(mask):
    rows = np.any(mask > 0, axis=1)
    if not rows.any():
        return 0
    idx = np.where(rows)[0]
    return int(idx[-1] - idx[0] + 1)


def compute_cdr(label_mask):
    disc = (label_mask == 1).astype(np.uint8)
    cup = (label_mask == 2).astype(np.uint8)

    disc_h = compute_vertical_diameter(disc)
    cup_h = compute_vertical_diameter(cup)

    return 0.0 if disc_h == 0 else cup_h / disc_h
# --- Glaucoma classification from CDR --- #

def classify_glaucoma(cdr: float,
                      suspect_threshold: float = 0.6,
                      glaucoma_threshold: float = 0.7):
    """
    Simple clinical-style risk classification from vertical CDR.
    You can tune thresholds based on literature / your data.
    """
    if cdr >= glaucoma_threshold:
        label = "Glaucoma likely"
        level = "high_risk"
    elif cdr >= suspect_threshold:
        label = "Glaucoma suspect"
        level = "moderate_risk"
    else:
        label = "Normal"
        level = "low_risk"
    return label, level
