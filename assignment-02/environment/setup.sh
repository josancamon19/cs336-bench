#!/bin/bash
# Runs inside runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 before /app is populated.
# Installs cs336-systems deps into the system Python.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

rm -f /usr/lib/python*/EXTERNALLY-MANAGED

apt-get update
apt-get install -y --no-install-recommends rsync git build-essential cmake ripgrep jq
rm -rf /var/lib/apt/lists/*

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Torch is pre-installed in the base image (2.8 + cu128). triton is too.
# Install the remaining cs336-systems deps.
uv pip install --system --no-cache \
    einops einx jaxtyping numpy psutil pytest regex tiktoken tqdm \
    tokenizers transformers pyarrow setuptools wandb humanfriendly \
    matplotlib pandas pytest-json-report

mkdir -p /app /logs/verifier

echo "[setup] cs336 assignment-2 environment ready"
python3 -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda', torch.cuda.is_available())"
