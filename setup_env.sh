#!/usr/bin/env bash
# setup_env.sh — conda environment setup for WSL/Linux (RTX 4070, CUDA 12.8 wheels, driver 13.1)
# Usage: bash setup_env.sh

set -e

ENV_NAME="kartikeya-khana-venv"
PYTHON_VERSION="3.11"

# Initialise conda for this shell session
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo "Activating environment..."
conda activate "${ENV_NAME}"

echo "Installing packages (PyTorch cu126 + dependencies)..."
pip install -r requirements.txt

echo ""
echo "Verifying CUDA..."
python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("WARNING: CUDA not available — check NVIDIA driver in WSL2")
EOF

echo ""
echo "Done! To activate in future sessions:"
echo "  conda activate ${ENV_NAME}"
