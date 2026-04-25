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
    pytest-json-report \
    pytest-timeout

# vLLM and flash-attn are imported by the oracle's training scripts but never
# CALLED by the adapter tests. Installing real vLLM (5-10 min cold + ~10GB
# disk) only to satisfy import statements would blow the verifier budget.
# Instead, drop a minimal stub package on the path so every `from vllm
# import ...` succeeds; the stubbed classes raise if anyone actually tries
# to use them.
STUB_DIR="$(python3 -c 'import site; print(site.getsitepackages()[0])')/vllm"
mkdir -p "$STUB_DIR"
cat > "$STUB_DIR/__init__.py" <<'PY'
class LLM:
    def __init__(self, *a, **k): pass
    def generate(self, *a, **k):
        raise RuntimeError("vllm stub: LLM.generate is not available in the test environment")
class SamplingParams:
    def __init__(self, *a, **k): pass
PY
cat > "$STUB_DIR/sampling_params.py" <<'PY'
class SamplingParams:
    def __init__(self, *a, **k): pass
class GuidedDecodingParams:
    def __init__(self, *a, **k): pass
PY
mkdir -p "$STUB_DIR/model_executor"
cat > "$STUB_DIR/model_executor/__init__.py" <<'PY'
def set_random_seed(seed): pass
PY

# Note: we deliberately do NOT stub flash_attn. transformers probes for it at
# import time and breaks if a partial stub is found; better to leave it absent
# so transformers' own absence detection takes over. The oracle source doesn't
# import flash_attn directly, only its pyproject lists it as a dep.

# math-verify, alpaca-eval (test deps)
uv pip install --system --no-cache "math-verify[antlr4-13-2]" alpaca-eval || true

mkdir -p /app /logs/verifier

echo "[setup] cs336 assignment-5 environment ready"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python3 -c "import vllm; print('vllm ok')" 2>&1 || echo "[setup] WARN: vllm import failed"
python3 -c "import flash_attn; print('flash_attn ok')" 2>&1 || echo "[setup] WARN: flash_attn import failed"
