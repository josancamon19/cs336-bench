#!/bin/bash
# Runs inside runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 before /app is populated.
# Installs cs336-alignment deps (vLLM, flash-attn, accelerate, transformers...).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PATH="/root/.local/bin:$PATH"

rm -f /usr/lib/python*/EXTERNALLY-MANAGED

apt-get update
apt-get install -y --no-install-recommends rsync git build-essential cmake ripgrep jq
rm -rf /var/lib/apt/lists/*

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Core cs336-alignment stack. Keep vllm/flash-attn pins loose so they resolve
# against whatever torch ships in the runpod image (2.8 + cu128). The upstream
# pyproject pins vllm==0.7.2 / flash-attn==2.7.4.post1 but those were built
# against torch 2.5 — newer wheels are fine for the unit tests.
uv pip install --system --no-cache \
    accelerate transformers datasets evaluate \
    einops einx jaxtyping numpy psutil pytest regex tqdm \
    tokenizers pyarrow setuptools wandb typer pylatexenc xopen \
    pytest-json-report

# Flash-attn and vLLM — pin to versions that support torch 2.8. Build of
# flash-attn is heavy; skip build isolation so it uses the already-installed
# torch.
uv pip install --system --no-cache --no-build-isolation flash-attn
uv pip install --system --no-cache vllm

# math-verify, alpaca-eval (test deps)
uv pip install --system --no-cache "math-verify[antlr4-13-2]" alpaca-eval || true

mkdir -p /app /logs/verifier

echo "[setup] cs336 assignment-5 environment ready"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python3 -c "import vllm; print('vllm ok')" 2>&1 || echo "[setup] WARN: vllm import failed"
python3 -c "import flash_attn; print('flash_attn ok')" 2>&1 || echo "[setup] WARN: flash_attn import failed"
