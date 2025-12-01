#!/usr/bin/env bash
set -e

echo "=== Pre-push sanity check ==="

# 1) Ensure we are in a git repo
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "ERROR: Not inside a git repository."
  exit 1
fi

# 2) Show current branch and status
echo
echo "=== Git status ==="
git status

# 3) Check for unstaged changes
echo
echo "=== Checking for unstaged changes ==="
if ! git diff --quiet; then
  echo "ERROR: There are unstaged changes. Stage them or revert before pushing."
  exit 1
else
  echo "OK: No unstaged changes."
fi

# 4) Check for uncommitted staged changes
echo
echo "=== Checking for staged-but-uncommitted changes ==="
if ! git diff --cached --quiet; then
  echo "NOTE: There are staged changes that will be included in the next commit."
else
  echo "NOTE: No staged changes. You are about to push only existing commits."
fi

# 5) Ensure FP16 ONNX model exists and is not huge
echo
echo "=== Checking ONNX model ==="
MODEL="glaucoma_unet_fp16.onnx"
if [ ! -f "$MODEL" ]; then
  echo "ERROR: $MODEL is missing in the repo root."
  exit 1
fi

SIZE_BYTES=$(stat -c%s "$MODEL" 2>/dev/null || stat -f%z "$MODEL" 2>/dev/null || echo 0)
SIZE_MB=$(( SIZE_BYTES / 1024 / 1024 ))
echo "Found $MODEL (${SIZE_MB} MB)"

if [ "$SIZE_BYTES" -gt 100000000 ]; then
  echo "ERROR: $MODEL is larger than 100 MB. GitHub will reject this."
  exit 1
fi

echo "OK: $MODEL is under 100 MB."

# 6) Warn about any tracked files over 90 MB
echo
echo "=== Scanning tracked files for > 90 MB ==="
LARGE_FILES=false
while IFS= read -r f; do
  if [ -f "$f" ]; then
    sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt 90000000 ]; then
      mb=$(( sz / 1024 / 1024 ))
      echo "WARNING: Tracked file '$f' is ${mb} MB (> 90 MB)"
      LARGE_FILES=true
    fi
  fi
done < <(git ls-files)

if [ "$LARGE_FILES" = false ]; then
  echo "OK: No tracked files over 90 MB."
fi

echo
echo "=== All checks passed ==="
echo "You are safe to commit and push."
