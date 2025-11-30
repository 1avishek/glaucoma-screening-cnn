from pathlib import Path
import numpy as np
from PIL import Image

import streamlit as st
import albumentations as A
import onnxruntime as ort

from utils import compute_cdr, classify_glaucoma
from visualize import create_overlay, decode_segmentation_mask


def get_transform():
    # No ToTensorV2, we stay in NumPy for ONNXRuntime
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


@st.cache_resource
def load_session(project_root: Path):
    """
    Load ONNX model with ONNX Runtime.
    Expects glaucoma_unet.onnx at the repo root.
    """
    onnx_path = project_root.parent / "glaucoma_unet.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}.\n"
            "Make sure glaucoma_unet.onnx is committed to the repo."
        )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def run_inference(image_pil, sess, input_name: str, output_name: str):
    """
    Run segmentation using ONNX model and compute CDR + glaucoma risk.
    """
    image_np = np.array(image_pil)

    # Albumentations on NumPy
    transform = get_transform()
    augmented = transform(image=image_np)
    img = augmented["image"].astype(np.float32)  # HWC

    # ONNX expects NCHW
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)        # CHW -> NCHW

    # Inference
    logits = sess.run([output_name], {input_name: img})[0]  # (N, C, H, W)
    pred = np.argmax(logits, axis=1)[0].astype(np.int64)    # (H, W)

    # CDR + risk
    cdr = compute_cdr(pred)
    label, risk_level = classify_glaucoma(cdr)

    # For visualization, resize original image to mask size
    h, w = pred.shape
    resized_img = image_pil.resize((w, h), Image.BILINEAR)
    resized_np = np.array(resized_img)

    overlay = create_overlay(resized_np, pred, cdr, label)
    color_mask = decode_segmentation_mask(pred)

    return resized_np, color_mask, overlay, cdr, label, risk_level


def main():
    st.title("Automated Glaucoma Screening from Fundus Images")

    project_root = Path(__file__).resolve().parent
    device_label = "cpu (ONNXRuntime)"

    st.sidebar.write(f"Device: `{device_label}`")
    st.sidebar.write("Loading ONNX model...")
    sess, input_name, output_name = load_session(project_root)
    st.sidebar.success("Model loaded.")

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
        # Sample images from web_app/sample_images folder
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
                    image_pil, sess, input_name, output_name
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