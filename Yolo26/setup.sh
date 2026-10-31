#!/usr/bin/env bash
# CPU-only environment: keeps the GPU free for other workloads.
set -euo pipefail
cd "$(dirname "$0")"
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
