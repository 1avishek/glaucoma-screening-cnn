from pathlib import Path
import shutil
import numpy as np
from PIL import Image

import streamlit as st
import albumentations as A
import onnxruntime as ort
import requests
import torch

from model import build_model
from utils import compute_cdr, classify_glaucoma
from visualize import create_overlay, decode_segmentation_mask


MODEL_FILE_ID = "1XNH_wZFU3E0X0F3wM9zUA9qLAkiDt_6t"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download"
CHECKPOINT_MIN_BYTES = 5 * 1024 * 1024  # guard against HTML placeholder
ONNX_MIN_BYTES = 100 * 1024


def get_transform():
    # No ToTensorV2, we stay in NumPy for ONNXRuntime
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def _get_confirm_token(response: requests.Response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response: requests.Response, destination: Path):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def download_checkpoint(destination: Path):
    session = requests.Session()
    response = session.get(GOOGLE_DRIVE_URL, params={"id": MODEL_FILE_ID}, stream=True)
    token = _get_confirm_token(response)

    if token:
        response = session.get(
            GOOGLE_DRIVE_URL,
            params={"id": MODEL_FILE_ID, "confirm": token},
            stream=True,
        )

    response.raise_for_status()
    _save_response_content(response, destination)


def ensure_checkpoint(project_root: Path) -> Path:
    weights_dir = project_root / "web_model"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "best_model.pth"

    def valid() -> bool:
        return weights_path.exists() and weights_path.stat().st_size > CHECKPOINT_MIN_BYTES

    if valid():
        return weights_path

    if weights_path.exists():
        weights_path.unlink()

    repo_root = project_root.parent
    local_checkpoint = repo_root / "outputs" / "fold_0" / "best_model.pth"
    if local_checkpoint.exists() and local_checkpoint.stat().st_size > CHECKPOINT_MIN_BYTES:
        shutil.copy2(local_checkpoint, weights_path)
        return weights_path

    download_checkpoint(weights_path)
    if not valid():
        raise RuntimeError(
            "Downloaded checkpoint looks invalid. Upload outputs/fold_0/best_model.pth manually."
        )
    return weights_path


def export_checkpoint_to_onnx(checkpoint_path: Path, onnx_path: Path):
    device = torch.device("cpu")
    model = build_model("unet", num_classes=3).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 256, 256, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def ensure_onnx_model(project_root: Path) -> Path:
    onnx_path = project_root.parent / "glaucoma_unet.onnx"
    if onnx_path.exists() and onnx_path.stat().st_size > ONNX_MIN_BYTES:
        return onnx_path

    checkpoint_path = ensure_checkpoint(project_root)
    export_checkpoint_to_onnx(checkpoint_path, onnx_path)
    if not onnx_path.exists() or onnx_path.stat().st_size <= ONNX_MIN_BYTES:
        raise RuntimeError("Failed to export ONNX model. Check checkpoint integrity.")
    return onnx_path


@st.cache_resource
def load_session(project_root: Path):
    """
    Load ONNX model with ONNX Runtime, exporting it on the fly if missing.
    """
    onnx_path = ensure_onnx_model(project_root)

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
