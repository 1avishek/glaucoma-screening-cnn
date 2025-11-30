import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(
        description="Glaucoma segmentation + CDR pipeline (U-Net + 5-fold CV)"
    )

    # Paths
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Root directory of the project (where REFUGE/, G1020/, etc. live).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="REFUGE",
        choices=["REFUGE"],
        help="Dataset to use (for now we wire REFUGE first).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where logs, checkpoints, and results are saved.",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="unet",
        choices=["unet"],
        help="Segmentation backbone (can extend later).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of classes in mask: 0=background,1=disc,2=cup.",
    )

    # Cross-validation
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument(
        "--fold_index",
        type=int,
        default=0,
        help="Which fold to train (0..num_folds-1).",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Misc
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")

    args = parser.parse_args()
    return args
