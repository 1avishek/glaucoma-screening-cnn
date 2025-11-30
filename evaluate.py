# export_onnx.py
from pathlib import Path
import torch

from model import build_model


def main():
    project_root = Path(__file__).resolve().parent
    best_model = project_root / "outputs" / "fold_0" / "best_model.pth"  # or whichever fold you like
    onnx_path = project_root / "glaucoma_unet.onnx"

    device = torch.device("cpu")
    model = build_model("unet", num_classes=3).to(device)
    state = torch.load(best_model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 256, 256, device=device)

    # Use a recent opset and keep shapes static
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,  # modern opset, matches current exporter expectations
    )

    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
