# 🚁 CrowdVision - Drone AI Crowd Detection & Tactical Command Center

Sistem monitoring kerumunan (*Crowd Detection & Analytics*) berbasis AI real-time berkecepatan tinggi yang dirancang untuk analisis video CCTV dan aerial drone. Dilengkapi dengan **Dual-Engine Architecture** (YOLO Head Detection + P2PNet Point Localization + DM-Count Density Estimation) serta sistem **Deteksi Akselerasi Hardware Otomatis** (**NVIDIA CUDA RTX** 🚀, **Apple Silicon MPS** ⚡, dan **CPU Fallback**).

---

## 📋 Daftar Isi
1. [Fitur Utama](#-fitur-utama)
2. [Arsitektur Hardware & Akselerasi](#-arsitektur-hardware--akselerasi)
3. [Kebutuhan Sistem](#-kebutuhan-sistem)
4. [Cara Install & Setup Otomatis](#-cara-install--setup-otomatis)
5. [Cara Menjalankan Server](#-cara-menjalankan-server)
6. [Validasi Akselerasi Hardware (CUDA / MPS)](#-validasi-akselerasi-hardware-cuda--mps)
7. [Panduan Migrasi Mac ke PC RTX (PENTING)](#-panduan-migrasi-mac-ke-pc-rtx-penting)
8. [Struktur Model & Engine AI](#-struktur-model--engine-ai)
9. [Struktur Direktori Project](#-struktur-direktori-project)

---

## 🌟 Fitur Utama

- **Real-Time AI Multi-Engine**:
  - **YOLO Head Detection** (`aminesam.pt`): Deteksi kepala manusia dari sudut pandang drone / atas (*top-down aerial view*).
  - **P2PNet Point Localization** (`p2pcrowdcounting.pth`): Menghitung titik pusat kepala (*tactical dots*) secara akurat pada kerumunan sangat padat.
  - **SAHI Sliced Inference**: Deteksi objek kecil pada resolusi tinggi tanpa kehilangan detail.
  - **DM-Count (QNRF)**: Estimasi kepadatan (*density map heatmap*) untuk crowd berskala masif.
  - **Crowd Eye Auto Routing**: Otomatis memilih antara metode detection vs density berdasarkan rasio kepadatan visual.
- **Interactive Tactical Zone**: Gambar polygon multi-zone bebas (Zona A, B, C) langsung di atas canvas video untuk counting spesifik per area.
- **Auto Hardware Fallback**: Otomatis mendeteksi GPU terbaik yang tersedia (`CUDA` $\rightarrow$ `MPS` $\rightarrow$ `CPU`).
- **Tactical Dark HUD Dashboard**: Dashboard web real-time responsif berbasis FastAPI dan WebSocket.

---

## ⚡ Arsitektur Hardware & Akselerasi

Aplikasi ini secara otomatis mendeteksi hardware yang tersedia dengan prioritas:

| Prioritas | Device | Hardware Target | Status Acceleration |
|---|---|---|---|
| **1 (Utama)** | `cuda:0` | **NVIDIA RTX Series** (misal RTX 3060, 4060, 4080, 4090) | Ultra-fast Tensor Core (CUDA 12.1) |
| **2** | `mps` | **Apple Silicon Mac** (M1, M2, M3, M4 series) | Apple Metal Performance Shaders |
| **3 (Fallback)** | `cpu` | Multi-core CPU (Intel / AMD) | CPU Threading |

---

## 💻 Kebutuhan Sistem

- **Sistem Operasi**:
  - Linux (Ubuntu 20.04/22.04 LTS / Debian) — *Sangat direkomendasikan untuk NVIDIA CUDA*
  - macOS (Apple Silicon M-Series)
  - Windows 10/11 (Menggunakan WSL2 Ubuntu direkomendasikan untuk CUDA)
- **Python**: Python **3.10**, **3.11**, atau **3.12**
- **NVIDIA GPU (Opsional untuk CUDA)**: Driver NVIDIA terbaru + `nvidia-smi` terdeteksi.

---

## 🚀 Cara Install & Setup Otomatis

Script [`setup.sh`](setup.sh) sudah dirancang untuk mendeteksi sistem operasi dan GPU Anda secara otomatis, mengunduh build PyTorch yang tepat (CUDA 12.1 untuk RTX atau native untuk Mac MPS), serta menginstall seluruh dependensi:

```bash
# 1. Masuk ke direktori project
cd CrowdDetection

# 2. Berikan izin eksekusi pada script (hanya perlu sekali)
chmod +x setup.sh run.sh

# 3. Jalankan script setup otomatis
./setup.sh
```

> **Apa yang dilakukan oleh `setup.sh`?**
> 1. Mendeteksi OS (Linux/macOS) dan arsitektur CPU.
> 2. Memeriksa `nvidia-smi` untuk mendeteksi GPU NVIDIA RTX.
> 3. Membuat virtual environment `.venv` secara bersih.
> 4. Menginstall PyTorch dengan build yang sesuai:
>    - Jika ada GPU NVIDIA: Menginstall `torch torchvision --index-url https://download.pytorch.org/whl/cu121`
>    - Jika di Apple Silicon: Menginstall PyTorch native Metal MPS.
>    - Jika tanpa GPU: Menginstall PyTorch CPU-optimized.
> 5. Menginstall seluruh library di `requirements.txt`.
> 6. Menjalankan audit validasi mandiri `test_env.py`.

---

## 🎬 Cara Menjalankan Server

Setelah setup selesai, jalankan server dengan:

```bash
./run.sh
```

Atau jika menggunakan `bash`:
```bash
bash run.sh
```

### Banner Log Otomatis di Terminal
Saat `run.sh` dijalankan, terminal akan mencetak konfirmasi hardware yang sedang aktif:

```text
=============================================================
   🔍 HARDWARE ACCELERATION DETECTED
=============================================================
   • OS Platform    : Linux (x86_64)
   • Compute Device : NVIDIA GeForce RTX 4090 (24.0 GB VRAM) 🚀
   • Device Code    : cuda:0
   • PyTorch Version: v2.2.1+cu121
   • Status         : High-Performance Tensor Core Active (CUDA 12.1)
=============================================================
[INFO] Menjalankan server FastAPI di http://localhost:8000
[INFO] Buka http://localhost:8000 di browser kamu
```

Buka browser Anda dan akses:
👉 **`http://localhost:8000`**

---

## 🔍 Validasi Akselerasi Hardware (CUDA / MPS)

Untuk memastikan bahwa PyTorch benar-benar menggunakan GPU NVIDIA RTX Anda (bukan CPU), tersedia beberapa metode validasi:

### Metode 1: Menggunakan Script Audit Lengkap (`test_env.py`)
Jalankan script verifikasi mandiri di dalam virtual environment:

```bash
python test_env.py
```

**Contoh Output jika CUDA Berhasil:**
```text
==================================================
  CROWDVISION HARDWARE & ENVIRONMENT AUDIT
==================================================
Python Version : 3.12.x
Platform       : Linux (x86_64)

==================================================
  1. PYTORCH ACCELERATION CHECK
==================================================
PyTorch Version     : 2.2.1+cu121
Torchvision Version : 0.17.1+cu121
[OK] CUDA Available : YES (Found 1 GPU)
     Active GPU     : NVIDIA GeForce RTX 4090 (24.0 GB VRAM, CUDA 12.1)
     Matrix Benchmark: 2000x2000 matmul in 1.42 ms

==================================================
  2. YOLO DETECTION ENGINE
==================================================
[OK] Found and loaded 'aminesam.pt' successfully (1 class: {0: 'head'})
[OK] Found and loaded 'yakhyo.pt' successfully (1 class: {0: 'person'})

==================================================
  3. P2PNET POINT LOCALIZATION ENGINE
==================================================
[OK] P2PNet architecture initialized and weights loaded on cuda:0

==================================================
  4. DENSITY ESTIMATION ENGINE (DM-COUNT)
==================================================
[OK] DM-Count (QNRF) initialized on cuda:0

==================================================
  AUDIT SUMMARY
==================================================
All inference engines are READY on: NVIDIA GEFORCE RTX 4090
```

---

### Metode 2: Cek Singkat via Python One-Liner
Jalankan perintah ini di terminal:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Active Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- Jika berhasil di RTX: `CUDA Available: True | Active Device: NVIDIA GeForce RTX ...`
- Jika di Mac:
  ```bash
  python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
  ```
  Output: `MPS Available: True`

---

### Metode 3: Cek Lewat Dashboard Web
1. Buka dashboard di `http://localhost:8000`.
2. Lihat indikator badge di bagian atas:
   - `HW: CUDA` (Berwarna Hijau Neon di PC RTX)
   - `HW: MPS` (Berwarna Cyan di Apple Silicon)
3. Di sudut kiri bawah video stream pada Tactical HUD, terdapat teks:
   `ENGINE: YOLO (HEAD) | HW: CUDA | CROWD-EYE: DETECTION | RATIO: 1.00`

---

## ⚠️ Panduan Migrasi Mac ke PC RTX (PENTING)

Jika Anda memindahkan codebase ini dari Mac ke PC Windows/Linux dengan GPU NVIDIA RTX:

### ❌ JANGAN Meng-copy folder `.venv`!
Folder virtual environment (`.venv`) berisi file biner khusus OS:
- Di Mac: File biner `.dylib` dan PyTorch versi macOS (MPS).
- Di Linux: Membutuhkan file biner `.so` dan PyTorch CUDA wheel.
Meng-copy folder `.venv` antar OS yang berbeda **pasti akan gagal** (*corrupt*).

### ✅ Langkah Benar Saat Pindah ke PC RTX:
1. Copy seluruh folder project, **KECUALI** folder `.venv`.
   (Pastikan file bobot `aminesam.pt`, `yakhyo.pt`, dan `crowd-eye/yolo/p2pcrowdcounting.pth` ikut ter-copy).
2. Di PC RTX (Linux / WSL2):
   ```bash
   cd CrowdDetection
   bash setup.sh
   ```
3. `setup.sh` akan:
   - Mendeteksi `nvidia-smi`.
   - Mengambil wheel PyTorch CUDA 12.1 resmi dari PyTorch repository.
   - Menguji matmul benchmark pada GPU NVIDIA RTX Anda.
4. Jalankan aplikasi:
   ```bash
   ./run.sh
   ```

---

## 🧠 Struktur Model & Engine AI

| Model File | Engine | Karakteristik / Target Objek | Kapan Digunakan |
|---|---|---|---|
| `aminesam.pt` | **YOLOv8** | Deteksi Kepala (*Head Detection*) | **Rekomendasi Utama** untuk Drone / CCTV sudut tinggi (*top-down*). |
| `yakhyo.pt` | **YOLOv8** | Deteksi Badan Manusia (*Full Person*) | Sudut pandang horizontal (*ground-level angle*). |
| `p2pcrowdcounting.pth` | **P2PNet** | Point Localization (*Head Dots*) | Kerumunan sangat padat (*dense crowd*) di mana bounding box saling bertumpuk. |
| `DM-Count (QNRF)` | **LWCC** | Density Map Heatmap | Kerumunan masif (konser, stadion, demonstrasi) dengan puluhan ribu orang. |

> **Mode Crowd Eye (Auto Engine):**
> Mode **AUTO** akan menjalankan YOLO + SAHI secara berkala dan membandingkan estimasi deteksi dengan density model. Jika kepadatan visual melewati ambang batas rasio, sistem akan otomatis beralih (*fallback*) ke estimasi kepadatan untuk mencegah *under-counting*.

---

## 📁 Struktur Direktori Project

```text
CrowdDetection/
├── backend/
│   ├── main.py             # FastAPI App, WebSocket handler, Zone API
│   ├── detector.py         # Engine Wrapper (YOLO, P2PNet, Crowd Eye, HUD overlay)
│   └── streamer.py         # Video streaming & frame processing loop
├── crowd/
│   ├── __init__.py         # CrowdService entrypoint
│   ├── detector.py         # YOLO + SAHI slice inference logic
│   ├── density.py          # DM-Count density map estimation
│   ├── models.py           # Dataclass settings & results
│   └── p2pnet.py           # P2PNet architecture & point prediction
├── frontend/
│   ├── index.html          # Tactical HUD Dashboard UI
│   ├── style.css           # Drone Command Center styling (Glassmorphism)
│   └── app.js              # WebSocket consumer, video canvas & polygon zone editor
├── sample_data/
│   └── create_demo.py      # Generator video demo crowd sintetis
├── aminesam.pt             # Bobot YOLO Head Detection
├── yakhyo.pt               # Bobot YOLO Person Detection
├── crowd-eye/
│   └── yolo/
│       └── p2pcrowdcounting.pth # Bobot P2PNet Point Counting
├── requirements.txt        # Dependensi cross-platform Python
├── setup.sh                # Script installer otomatis (CUDA / MPS / CPU)
├── run.sh                  # Script launcher server dengan hardware check
├── test_env.py             # Script audit validasi PyTorch & GPU benchmark
└── README.md               # Dokumentasi lengkap
```

---

## 🛠️ Troubleshooting

- **Error: `CUDA out of memory`**
  - Turunkan ukuran resolusi inferensi pada dashboard UI (misal dari `1280` ke `640` atau `960`).
  - Matikan fitur SAHI jika video berukuran sangat besar.
- **Port 8000 Sedang Digunakan**
  - Script `run.sh` secara otomatis membebaskan port 8000 sebelum menyalakan server baru. Jika ingin menghentikan manual:
    ```bash
    kill -9 $(lsof -ti :8000)
    ```
- **NVIDIA GPU Tidak Terdeteksi di `setup.sh`**
  - Pastikan driver NVIDIA sudah terpasang dengan menjalankan `nvidia-smi` di terminal.
  - Jika `nvidia-smi` tidak ada, install driver NVIDIA terlebih dahulu melalui package manager OS Anda.

---

⭐ **CrowdVision Pro** — *Precision Aerial Drone Intelligence & Dense Crowd Monitoring.*
