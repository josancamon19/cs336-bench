#!/bin/bash
# Runs inside runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (torch 2.8 + CUDA 12.8
# already installed). Installs the remaining cs336 assignment-1 deps into the
# system Python so `pytest` / `python` just work.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

# Remove PEP 668 marker so uv can install into system Python safely.
rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Minimal apt additions (rsync for verifier overlay, git + build-essential for
# pip source builds that may appear).
apt-get update
apt-get install -y --no-install-recommends rsync git build-essential cmake ripgrep jq
rm -rf /var/lib/apt/lists/*

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Install the assignment's dependencies (torch is already in the base image;
# uv will skip/upgrade only what's needed). System-wide so `pytest` / `python`
# work from anywhere. /app is populated by harbor AFTER this script runs, so
# we can't reference it here — only ensure the directory exists.
uv pip install --system --no-cache \
    einops einx jaxtyping numpy psutil pytest regex tiktoken tqdm \
    tokenizers transformers pyarrow line-profiler setuptools \
    pytest-json-report

mkdir -p /app /logs/verifier

echo "[setup] cs336 assignment-1 environment ready"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
