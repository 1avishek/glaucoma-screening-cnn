#!/usr/bin/env python3
"""
Helper script to download glaucoma_unet.onnx (+ optional .data) from Google Drive.

Usage:
    python download_onnx_assets.py --onnx-id <FILE_ID> [--data-id <FILE_ID>] [--output-dir .]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gdown


def download(file_id: str, destination: Path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination), quiet=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-id",
        required=True,
        help="Google Drive file ID for glaucoma_unet.onnx",
    )
    parser.add_argument(
        "--data-id",
        default="",
        help="Optional Google Drive file ID for glaucoma_unet.onnx.data",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the files will be saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "glaucoma_unet.onnx"
    download(args.onnx_id, onnx_path)

    if args.data_id.strip():
        data_path = out_dir / "glaucoma_unet.onnx.data"
        download(args.data_id, data_path)

    print("Download complete:")
    print(f"  {onnx_path.resolve()}")
    if args.data_id.strip():
        print(f"  {(out_dir / 'glaucoma_unet.onnx.data').resolve()}")


if __name__ == "__main__":
    main()
