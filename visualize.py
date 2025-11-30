# visualize.py
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2


def decode_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H, W) with values 0=bg, 1=disc, 2=cup
    returns: (H, W, 3) RGB uint8 color mask
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Colors (BGR in cv2, will convert to RGB at the end)
    bg = (0, 0, 0)
    disc = (0, 255, 0)   # green
    cup = (0, 0, 255)    # red

    color_mask[mask == 0] = bg
    color_mask[mask == 1] = disc
    color_mask[mask == 2] = cup

    # convert BGR->RGB
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
    return color_mask


def create_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    cdr: float,
    label: str,
) -> np.ndarray:
    """
    image_rgb: (H, W, 3) uint8
    mask: (H, W) int {0,1,2}
    returns overlay RGB uint8
    """
    # Resize image to mask size if needed
    h, w = mask.shape
    if image_rgb.shape[:2] != (h, w):
        image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)

    color_mask = decode_segmentation_mask(mask)
    overlay = cv2.addWeighted(image_rgb, 0.7, color_mask, 0.3, 0)

    # Draw disc & cup contours for nicer clinical visualization
    for cls_id, color_bgr in [(1, (0, 255, 0)), (2, (0, 0, 255))]:
        binary = (mask == cls_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_bgr, 1)

    # Put text: CDR + label
    text = f"VCDR: {cdr:.2f} | {label}"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.putText(
        overlay_bgr,
        text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay
