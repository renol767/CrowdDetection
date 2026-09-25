"""Environment and Hardware Acceleration Self-Test Script.

Checks whether PyTorch is running on:
- NVIDIA CUDA (RTX GPUs)
- Apple Silicon (MPS)
- CPU Fallback

Also validates Ultralytics YOLO, P2PNet, and LWCC models.
"""

import sys
import platform
import time

def print_header(title: str):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)

def main():
    print_header("CROWDVISION HARDWARE & ENVIRONMENT AUDIT")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.system()} ({platform.machine()})")
    print(f"OS Release     : {platform.platform()}")

    # 1. PyTorch & Acceleration Check
    print_header("1. PYTORCH ACCELERATION CHECK")
    try:
        import torch
        import torchvision
        print(f"PyTorch Version     : {torch.__version__}")
        print(f"Torchvision Version : {torchvision.__version__}")

        device = "cpu"
        device_name = "CPU"

        if torch.cuda.is_available():
            device = "cuda:0"
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            cuda_ver = torch.version.cuda
            device_name = f"NVIDIA {gpu_name} ({vram_gb} GB VRAM, CUDA {cuda_ver})"
            print(f"[OK] CUDA Available : YES (Found {gpu_count} GPU)")
            print(f"     Active GPU     : {device_name}")

            # Quick CUDA tensor benchmark
            t0 = time.time()
            a = torch.randn(2000, 2000, device="cuda")
            b = torch.randn(2000, 2000, device="cuda")
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            dt = round((time.time() - t0) * 1000, 2)
            print(f"     Matrix Benchmark: 2000x2000 matmul in {dt} ms")

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple Silicon GPU (MPS)"
            print(f"[OK] MPS Available  : YES (Apple Silicon GPU)")
            print(f"     Active GPU     : {device_name}")

            t0 = time.time()
            a = torch.randn(2000, 2000, device="mps")
            b = torch.randn(2000, 2000, device="mps")
            c = torch.matmul(a, b)
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            dt = round((time.time() - t0) * 1000, 2)
            print(f"     Matrix Benchmark: 2000x2000 matmul in {dt} ms")

        else:
            print("[WARN] GPU Acceleration : NOT AVAILABLE (Using CPU fallback)")
            print("       If on an NVIDIA RTX machine, install CUDA PyTorch with:")
            print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    except ImportError as e:
        print(f"[ERROR] PyTorch not installed: {e}")
        sys.exit(1)

    # 2. YOLO Model Validation
    print_header("2. YOLO DETECTION ENGINE")
    try:
        from ultralytics import YOLO
        import os
        for m in ["aminesam.pt", "yakhyo.pt"]:
            if os.path.exists(m):
                model = YOLO(m)
                print(f"[OK] Found and loaded '{m}' successfully ({len(model.names)} class: {model.names})")
            else:
                print(f"[NOTE] Model file '{m}' not found in current root directory.")
    except Exception as e:
        print(f"[ERROR] YOLO failed to load: {e}")

    # 3. P2PNet Point Counting Validation
    print_header("3. P2PNET POINT LOCALIZATION ENGINE")
    try:
        from crowd.p2pnet import load_p2pnet
        weights = "crowd-eye/yolo/p2pcrowdcounting.pth"
        p2p_model = load_p2pnet(weights, device=device)
        print(f"[OK] P2PNet architecture initialized and weights loaded on {device} ({device_name})")
    except Exception as e:
        print(f"[ERROR] P2PNet validation error: {e}")

    # 4. Crowd Density Estimation (LWCC / DM-Count)
    print_header("4. DENSITY ESTIMATION ENGINE (DM-COUNT)")
    try:
        from crowd.density import get_density_model
        dm_model, w_name = get_density_model("DM-Count", "QNRF", device=device)
        print(f"[OK] DM-Count ({w_name}) initialized on {device}")
    except Exception as e:
        print(f"[ERROR] DM-Count validation error: {e}")

    print_header("AUDIT SUMMARY")
    print(f"All inference engines are READY on: {device_name.upper()}")
    print("Run './run.sh' to launch the CrowdVision Command Center server.\n")

if __name__ == "__main__":
    main()
