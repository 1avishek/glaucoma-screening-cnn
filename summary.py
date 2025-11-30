import argparse
import math
from pathlib import Path

image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"}

def human_size(nbytes: int) -> str:
    if nbytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = min(int(math.log(nbytes, 1024)), len(units) - 1)
    return f"{nbytes / (1024 ** idx):.1f} {units[idx]}"

def summarize_folder(base: Path):
    summary = []
    for path in base.rglob("*"):
        if path.is_dir():
            files = list(p for p in path.iterdir() if p.is_file())
            subdirs = list(p for p in path.iterdir() if p.is_dir())
            img_count = sum(1 for f in files if f.suffix.lower() in image_exts)
            total_size = sum(f.stat().st_size for f in files)
            summary.append({
                "folder": str(path.relative_to(base)),
                "files": len(files),
                "images": img_count,
                "subfolders": len(subdirs),
                "size": human_size(total_size),
            })
    # Add root itself
    files = list(p for p in base.iterdir() if p.is_file())
    subdirs = list(p for p in base.iterdir() if p.is_dir())
    img_count = sum(1 for f in files if f.suffix.lower() in image_exts)
    total_size = sum(f.stat().st_size for f in files)
    summary.append({
        "folder": ".",
        "files": len(files),
        "images": img_count,
        "subfolders": len(subdirs),
        "size": human_size(total_size),
    })
    return summary

def print_tree(base: Path):
    base = base.resolve()
    print(base)
    for path in sorted(base.rglob("*")):
        depth = len(path.relative_to(base).parts)
        prefix = "    " * (depth - 1) + ("└─ " if depth else "")
        marker = "/" if path.is_dir() else ""
        print(f"{prefix}{path.name}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Summarize image folders")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to summarize (defaults to this script's folder).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    summary = summarize_folder(root)
    print_tree(root)

    print("\nFolder summary:")
    col_order = ["folder", "files", "images", "subfolders", "size"]
    max_len = {c: max(len(c), *(len(str(r[c])) for r in summary)) for c in col_order}
    header = " | ".join(c.ljust(max_len[c]) for c in col_order)
    print(header)
    print("-" * len(header))
    for row in sorted(summary, key=lambda r: (r["folder"].count("/"), r["folder"])):
        line = " | ".join(str(row[c]).ljust(max_len[c]) for c in col_order)
        print(line)


if __name__ == "__main__":
    main()
