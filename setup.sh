#!/usr/bin/env bash
# setup.sh — one-shot VM setup for gpu_and_inference_hw
# Tested on Ubuntu 22.04 / 24.04 with CUDA pre-installed (Nebius H100/L40S images)
set -e

REPO_URL="https://github.com/NnamdiOdozi/gpu_and_inference_hw.git"
REPO_DIR="gpu_and_inference_hw"

# ── 1. System packages ────────────────────────────────────────────────────────
sudo apt-get update -y
sudo apt-get install -y python3-dev python3.12-venv git

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL"
fi
cd "$REPO_DIR"

# ── 3. Virtual environment ────────────────────────────────────────────────────
python3 -m venv .venv
source .venv/bin/activate

# ── 4. Python dependencies ────────────────────────────────────────────────────
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ── 5. Smoke test ─────────────────────────────────────────────────────────────
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "Setup complete. Activate next time with:"
echo "  source ${REPO_DIR}/.venv/bin/activate"
