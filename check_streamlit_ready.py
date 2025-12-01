#!/usr/bin/env python3
"""
Quick sanity checker before deploying the Streamlit app.

Usage:
    python check_streamlit_ready.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "web_app" / "app.py"
WEB_REQS = ROOT / "web_app" / "requirements.txt"
ONNX_PATH = ROOT / "glaucoma_unet_fp16.onnx"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def check_git_clean() -> tuple[bool, str]:
    proc = run(["git", "status", "--short"])
    clean = proc.stdout.strip() == ""
    return clean, proc.stdout.strip()


def check_file_tracked(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    proc = run(["git", "ls-files", "--error-unmatch", str(rel)])
    return proc.returncode == 0


def check_python_compile() -> tuple[bool, str]:
    proc = run([sys.executable, "-m", "py_compile", str(APP_PATH)])
    return proc.returncode == 0, proc.stderr.strip()


def main() -> int:
    issues: list[str] = []
    warnings: list[str] = []

    if not ONNX_PATH.exists():
        issues.append(f"Missing ONNX model: {ONNX_PATH}")
    elif ONNX_PATH.stat().st_size < 1_000_000:
        warnings.append(f"ONNX model seems too small ({ONNX_PATH.stat().st_size} bytes)")
    elif not check_file_tracked(ONNX_PATH):
        warnings.append("ONNX model exists but is not tracked by git.")

    if not WEB_REQS.exists():
        issues.append(f"Missing web app requirements: {WEB_REQS}")

    if not APP_PATH.exists():
        issues.append(f"Missing Streamlit app: {APP_PATH}")
    else:
        text = APP_PATH.read_text()
        if "glaucoma_unet_fp16.onnx" not in text:
            warnings.append("app.py does not reference glaucoma_unet_fp16.onnx path.")

    ok_compile, compile_err = check_python_compile()
    if not ok_compile:
        issues.append(f"Failed to byte-compile app.py:\n{compile_err}")

    clean, dirty_output = check_git_clean()
    if not clean:
        warnings.append(f"Git working tree not clean:\n{dirty_output}")

    print("=== Streamlit readiness check ===")
    if issues:
        print("\nProblems:")
        for msg in issues:
            print(f"  - {msg}")
    else:
        print("\nNo blocking issues detected.")

    if warnings:
        print("\nWarnings:")
        for msg in warnings:
            print(f"  - {msg}")

    if not issues:
        print("\n✅ Ready for deployment.")
        if warnings:
            print("   (Resolve warnings if possible.)")
        return 0

    print("\n❌ Please fix the problems above before deploying.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
