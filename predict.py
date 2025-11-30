# predict.py
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model
from utils import compute_cdr, classify_glaucoma
from visualize import create_overlay, decode_segmentation_mask


def load_ensemble_models(project_root: Path, device: torch.device, num_classes: int = 3):
    outputs_dir = project_root / "outputs"
    model_paths = sorted(outputs_dir.glob("fold_*/best_model.pth"))
    if not model_paths:
        raise FileNotFoundError(f"No best_model.pth found under {outputs_dir}")

    models = []
    for p in model_paths:
        print(f"Loading model from {p}")
        m = build_model("unet", num_classes=num_classes).to(device)
        state = torch.load(p, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} models for ensemble.")
    return models


def build_inference_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def infer_single_image(
    img_path: Path,
    models,
    device: torch.device,
    transform,
    num_classes: int = 3,
):
    # Load image as RGB
    image_pil = Image.open(img_path).convert("RGB")
    image_np = np.array(image_pil)

    augmented = transform(image=image_np)
    image_tensor = augmented["image"].unsqueeze(0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        probs_accum = None
        for m in models:
            logits = m(image_tensor)
            probs = torch.softmax(logits, dim=1)  # (1,C,H,W)
            probs_accum = probs if probs_accum is None else probs_accum + probs

        probs_mean = probs_accum / len(models)
        pred = torch.argmax(probs_mean, dim=1)[0].cpu().numpy().astype(np.int64)

    # Compute CDR & classification
    cdr = compute_cdr(pred)
    label, risk_level = classify_glaucoma(cdr)

    # Resize original image to prediction size for overlay
    h, w = pred.shape
    image_resized = np.array(image_pil.resize((w, h), Image.BILINEAR))

    overlay = create_overlay(image_resized, pred, cdr, label)
    color_mask = decode_segmentation_mask(pred)

    return {
        "image_resized": image_resized,
        "mask": pred,
        "overlay": overlay,
        "color_mask": color_mask,
        "cdr": float(cdr),
        "label": label,
        "risk_level": risk_level,
    }


def save_results(
    img_path: Path,
    result: dict,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    # Save overlay, color mask, raw mask
    Image.fromarray(result["overlay"]).save(output_dir / f"{stem}_overlay.png")
    Image.fromarray(result["color_mask"]).save(output_dir / f"{stem}_mask_color.png")

    # Raw mask as single-channel
    mask_uint8 = result["mask"].astype(np.uint8)
    Image.fromarray(mask_uint8).save(output_dir / f"{stem}_mask.png")

    # Save JSON summary
    meta = {
        "image_id": stem,
        "cdr": result["cdr"],
        "label": result["label"],
        "risk_level": result["risk_level"],
    }
    with open(output_dir / f"{stem}_result.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Glaucoma screening inference pipeline")
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Project root (contains outputs/, REFUGE/, etc.).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a fundus image or a directory of images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Where to save overlays, masks, and JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu).",
    )

    args = parser.parse_args()
    project_root = Path(args.project_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    models = load_ensemble_models(project_root, device, num_classes=3)
    transform = build_inference_transform()

    input_path = Path(args.input)
    output_dir = project_root / args.output_dir

    if input_path.is_dir():
        img_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            img_paths.extend(sorted(input_path.glob(ext)))
        if not img_paths:
            raise FileNotFoundError(f"No images found in directory {input_path}")
    else:
        img_paths = [input_path]

    print(f"Running inference on {len(img_paths)} image(s).")

    for p in img_paths:
        print(f"Processing {p}")
        result = infer_single_image(p, models, device, transform)
        save_results(p, result, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
