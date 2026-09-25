#!/usr/bin/env bash
# ==============================================================================
# CrowdVision Automated Environment Setup Script
# Supports:
#   - Linux / WSL (NVIDIA RTX / CUDA)
#   - macOS (Apple Silicon MPS / Intel)
#   - CPU Fallback
# ==============================================================================

set -e

# Change directory to project root
cd "$(dirname "$0")"

echo "=========================================================="
echo "    CROWDVISION - CROSS-PLATFORM ENVIRONMENT SETUP       "
echo "=========================================================="

# 1. Detect Operating System & Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "[INFO] Detected Operating System : $OS ($ARCH)"

# 2. Detect Hardware Acceleration (CUDA vs MPS vs CPU)
HW_TARGET="cpu"
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    HW_TARGET="cuda"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    CUDA_VER=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | head -n 1)
    echo "[INFO] NVIDIA GPU Detected       : $GPU_NAME ($CUDA_VER)"
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    HW_TARGET="mps"
    echo "[INFO] Apple Silicon GPU Detected: Apple M-Series (MPS)"
else
    echo "[WARN] No dedicated GPU detected. Falling back to CPU mode."
fi

# 3. Locate Compatible Python Binary (Python 3.10 - 3.12)
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" &> /dev/null; then
        PY_VER=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$candidate"
            echo "[INFO] Using Python Binary       : $candidate (v$PY_VER)"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 3.10, 3.11, or 3.12 is required but was not found."
    echo "        Please install Python 3.10+ before running setup.sh."
    exit 1
fi

# 4. Create or Verify Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating new virtual environment in .venv..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "[INFO] Found existing .venv virtual environment."
fi

echo "[INFO] Activating virtual environment..."
source .venv/bin/activate

echo "[INFO] Upgrading pip, setuptools, and wheel..."
pip install --quiet --upgrade pip setuptools wheel

# 5. Install PyTorch tailored for target accelerator
echo "=========================================================="
echo "    INSTALLING PYTORCH ACCELERATION PACKAGES             "
echo "=========================================================="

if [ "$HW_TARGET" = "cuda" ]; then
    echo "[INFO] Target: NVIDIA RTX (CUDA 12.1)"
    echo "[INFO] Installing PyTorch + CUDA wheels from official PyTorch repo..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [ "$HW_TARGET" = "mps" ]; then
    echo "[INFO] Target: Apple Silicon MPS"
    echo "[INFO] Installing native PyTorch..."
    pip install torch torchvision
else
    echo "[INFO] Target: CPU"
    echo "[INFO] Installing CPU-optimized PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 6. Install Project Dependencies
echo "=========================================================="
echo "    INSTALLING PROJECT DEPENDENCIES                      "
echo "=========================================================="
pip install -r requirements.txt

# 7. Run Verification Self-Test
echo "=========================================================="
echo "    RUNNING SYSTEM AUDIT & MODEL VERIFICATION            "
echo "=========================================================="
python test_env.py

echo "=========================================================="
echo "  SETUP COMPLETE! CROWDVISION IS READY.                   "
echo "  Run './run.sh' to launch the Command Center.           "
echo "=========================================================="
