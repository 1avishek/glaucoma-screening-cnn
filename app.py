# app.py
from pathlib import Path
import numpy as np
from PIL import Image

import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model
from utils import compute_cdr, classify_glaucoma
from visualize import create_overlay, decode_segmentation_mask


@st.cache_resource
def load_ensemble(project_root: Path, device: torch.device, num_classes: int = 3):
    outputs_dir = project_root / "outputs"
    model_paths = sorted(outputs_dir.glob("fold_*/best_model.pth"))
    if not model_paths:
        raise FileNotFoundError(f"No best_model.pth found under {outputs_dir}")

    models = []
    for p in model_paths:
        m = build_model("unet", num_classes=num_classes).to(device)
        state = torch.load(p, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models


def get_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def run_inference(image_pil, models, device):
    image_np = np.array(image_pil)

    transform = get_transform()
    augmented = transform(image=image_np)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs_accum = None
        for m in models:
            logits = m(image_tensor)
            probs = torch.softmax(logits, dim=1)
            probs_accum = probs if probs_accum is None else probs_accum + probs

        probs_mean = probs_accum / len(models)
        pred = torch.argmax(probs_mean, dim=1)[0].cpu().numpy().astype(np.int64)

    cdr = compute_cdr(pred)
    label, risk_level = classify_glaucoma(cdr)

    h, w = pred.shape
    resized_img = image_pil.resize((w, h), Image.BILINEAR)
    resized_np = np.array(resized_img)

    overlay = create_overlay(resized_np, pred, cdr, label)
    color_mask = decode_segmentation_mask(pred)

    return resized_np, color_mask, overlay, cdr, label, risk_level


def main():
    st.title("Automated Glaucoma Screening from Fundus Images")

    project_root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st.sidebar.write(f"Device: `{device}`")
    st.sidebar.write("Loading ensemble models...")
    models = load_ensemble(project_root, device, num_classes=3)
    st.sidebar.success(f"Loaded {len(models)} models.")

    # --- Input mode selection --- #
    st.sidebar.subheader("Choose input mode")
    mode = st.sidebar.radio(
        "How do you want to provide an image?",
        ["Upload your own", "Use sample image"],
    )

    image_pil = None

    if mode == "Upload your own":
        uploaded = st.file_uploader(
            "Upload a fundus image (JPG/PNG/TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )
        if uploaded is not None:
            image_pil = Image.open(uploaded).convert("RGB")
    else:
        # Sample images from sample_images folder
        sample_dir = project_root / "sample_images"
        sample_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            sample_paths.extend(sorted(sample_dir.glob(ext)))

        if not sample_paths:
            st.warning("No sample images found in sample_images/ folder.")
        else:
            names = [p.name for p in sample_paths]
            choice = st.selectbox("Choose a sample image:", names)
            chosen_path = sample_dir / choice
            image_pil = Image.open(chosen_path).convert("RGB")
            st.caption(f"Using sample image: {choice}")

    if image_pil is not None:
        st.subheader("Input image")
        st.image(image_pil, use_column_width=True)

        if st.button("Run glaucoma analysis"):
            with st.spinner("Running segmentation & computing CDR..."):
                img_vis, mask_vis, overlay, cdr, label, risk = run_inference(
                    image_pil, models, device
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("Resized Image")
                st.image(img_vis, use_column_width=True)
            with col2:
                st.caption("Segmentation (Disc=Green, Cup=Red)")
                st.image(mask_vis, use_column_width=True)
            with col3:
                st.caption("Overlay + CDR")
                st.image(overlay, use_column_width=True)

            st.markdown(f"**Vertical CDR:** `{cdr:.2f}`")
            st.markdown(f"**Risk classification:** `{label}`  (`{risk}`)")

    else:
        st.info("Upload an image or choose a sample image to get started.")


if __name__ == "__main__":
    main()
