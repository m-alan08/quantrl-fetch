#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/install.sh [auto|cpu|cu126|cu130|rocm]"
  echo
  echo "  auto   cpu on macOS; on Linux cu126 if nvidia-smi works, rocm if rocm-smi works, else cpu"
  echo "  cpu    CPU-only torch (also what macOS uses, with MPS available)"
  echo "  cu126  NVIDIA GPUs, CUDA 12.6 build (driver 525.60 or newer)"
  echo "  cu130  NVIDIA GPUs, CUDA 13.0 build (driver 580 or newer)"
  echo "  rocm   AMD GPUs, ROCm 7.2 build (x86_64 only)"
}

variant="${1:-auto}"
case "$variant" in
  -h|--help) usage; exit 0 ;;
  auto|cpu|cu126|cu130|rocm) ;;
  *) echo "Unknown variant: $variant" >&2; usage >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it without root with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "then open a new shell (or add ~/.local/bin to PATH) and rerun this script." >&2
  exit 1
fi

if [ "$variant" = "auto" ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    variant="cpu"
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    variant="cu126"
  elif command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
    variant="rocm"
  else
    variant="cpu"
  fi
  echo "auto selected torch variant: $variant"
fi

if [ "$variant" = "rocm" ] && [ "$(uname -m)" != "x86_64" ]; then
  echo "The ROCm build of torch is only published for x86_64, this machine is $(uname -m)." >&2
  exit 1
fi

echo "Installing Python 3.12 and the locked dependencies with torch variant '$variant'"
uv sync --locked --no-default-groups --group dev --group "$variant"
echo "$variant" > .venv/quantrl-torch-variant

echo
echo "Installed. Check:"
.venv/bin/python - <<'PYTHON'
import platform

import torch

import quantrl
from quantrl.utils.device import available_devices, resolve_device

print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, ROCm build {getattr(torch.version, 'hip', None)}")
print(f"quantrl {quantrl.__version__}")
print(f"available devices: {available_devices()}")
print(f"device chosen by auto: {resolve_device('auto')}")
PYTHON

echo
if [ "$variant" = "cpu" ]; then
  echo "Run commands with: uv run <command>"
else
  echo "This environment uses the '$variant' torch build, which is not uv's default."
  echo "Run commands with 'uv run --no-sync <command>' or '.venv/bin/python <script>',"
  echo "because a plain 'uv run' would switch torch back to the CPU build."
fi
