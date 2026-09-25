#!/usr/bin/env bash
set -e

# Pindah ke direktori project
cd "$(dirname "$0")"

echo "========================================="
echo "   CROWDVISION - DRONE COMMAND CENTER   "
echo "========================================="

# Cek dan aktifkan virtual environment .venv
if [ -d ".venv" ]; then
    echo "[INFO] Mengaktifkan virtual environment (.venv)..."
    . .venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
elif [ -f "setup.sh" ]; then
    echo "[INFO] Virtual environment (.venv) belum ada. Menjalankan setup.sh otomatis..."
    bash setup.sh
    . .venv/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true
else
    echo "[ERROR] .venv tidak ditemukan dan setup.sh tidak ada!"
    exit 1
fi

PYTHON_BIN=".venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Print clear hardware accelerator diagnostics (CUDA / MPS / CPU)
$PYTHON_BIN -c '
import torch, platform

os_name = f"{platform.system()} ({platform.machine()})"
torch_ver = f"v{torch.__version__}"

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    dev_title = f"NVIDIA {gpu_name} ({vram} GB VRAM) 🚀"
    dev_type = "cuda:0"
    cuda_ver = torch.version.cuda or "N/A"
    status_str = f"High-Performance Tensor Core Active (CUDA {cuda_ver})"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    dev_title = "Apple Silicon GPU (MPS Metal Acceleration) ⚡"
    dev_type = "mps"
    status_str = "Apple Metal Performance Shaders (Unified Memory) Active"
else:
    dev_title = "CPU Mode (Fallback - No GPU accelerator) ⚠️"
    dev_type = "cpu"
    status_str = "Standard Multi-core CPU Processing"

cyan = "\033[1;36m"
green = "\033[1;32m"
yellow = "\033[1;33m"
bold = "\033[1m"
reset = "\033[0m"

print("=============================================================")
print(f"   {bold}🔍 HARDWARE ACCELERATION DETECTED{reset}")
print("=============================================================")
print(f"   • {bold}OS Platform    :{reset} {os_name}")
print(f"   • {bold}Compute Device :{reset} {green}{dev_title}{reset}")
print(f"   • {bold}Device Code    :{reset} {cyan}{dev_type}{reset}")
print(f"   • {bold}PyTorch Version:{reset} {torch_ver}")
print(f"   • {bold}Status         :{reset} {yellow}{status_str}{reset}")
print("=============================================================")
'

# Cek apakah port 8000 sedang digunakan
PORT=8000
PID=$(lsof -ti :$PORT || true)
if [ -n "$PID" ]; then
    echo "[INFO] Membebaskan port $PORT (menghentikan proses lama PID: $PID)..."
    kill -9 $PID 2>/dev/null || true
    sleep 1
fi

echo "[INFO] Menjalankan server FastAPI di http://localhost:$PORT"
echo "[INFO] Buka http://localhost:$PORT di browser kamu"
echo "[INFO] Tekan CTRL+C untuk menghentikan server."
echo "============================================================="

# Jalankan uvicorn dengan auto-reload
exec $PYTHON_BIN -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT --reload

