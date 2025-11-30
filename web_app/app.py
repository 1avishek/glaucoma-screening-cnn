from pathlib import Path
import shutil
import numpy as np
from PIL import Image

import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import requests  # for downloading model weights

from model import build_model
from utils import compute_cdr, classify_glaucoma
from visualize import create_overlay, decode_segmentation_mask


# Google Drive ID for best_model.pth
MODEL_FILE_ID = "1XNH_wZFU3E0X0F3wM9zUA9qLAkiDt_6t"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download"
MIN_EXPECTED_SIZE = 5 * 1024 * 1024  # ~5 MB guard to detect HTML warning pages


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


def download_model_weight(weights_path: Path):
    """Download the model weight from Google Drive, handling the virus warning page."""
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
    _save_response_content(response, weights_path)


def ensure_model_weight(weights_path: Path):
    """
    Ensure that the model weight file exists at weights_path.
    Prefer copying from local outputs/, otherwise fall back to Google Drive download.
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    def _has_valid_file() -> bool:
        return weights_path.exists() and weights_path.stat().st_size > MIN_EXPECTED_SIZE

    if _has_valid_file():
        return

    if weights_path.exists():
        weights_path.unlink()  # remove corrupted HTML placeholder

    repo_root = Path(__file__).resolve().parent.parent
    local_checkpoint = repo_root / "outputs" / "fold_0" / "best_model.pth"
    if local_checkpoint.exists():
        shutil.copy2(local_checkpoint, weights_path)
        print(f"[MODEL] Copied weights from {local_checkpoint}")
        return

    try:
        print("[MODEL] Downloading weights from Google Drive...")
        download_model_weight(weights_path)
        if not _has_valid_file():
            raise RuntimeError("Downloaded file is suspiciously small.")
        print(f"[MODEL] Saved weights to {weights_path}")
    except Exception as e:
        raise RuntimeError(
            "Could not obtain model weights. "
            "Upload outputs/fold_0/best_model.pth manually or check network access."
        ) from e


@st.cache_resource
def load_model(project_root: Path, device: torch.device, num_classes: int = 3):
    """
    Load a single U-Net model from web_app/web_model/best_model.pth.
    If the file is missing, it will be downloaded first.
    """
    weights_dir = project_root / "web_model"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "best_model.pth"

    ensure_model_weight(weights_path)

    model = build_model("unet", num_classes=num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


def run_inference(image_pil, model, device):
    image_np = np.array(image_pil)

    transform = get_transform()
    augmented = transform(image=image_np)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.int64)

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
    st.sidebar.write("Loading model...")
    model = load_model(project_root, device, num_classes=3)
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
                    image_pil, model, device
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
