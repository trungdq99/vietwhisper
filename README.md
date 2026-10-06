# VietWhisper 🎙️

**VietWhisper** là ứng dụng chuyển đổi âm thanh tiếng Việt sang văn bản (**Vietnamese Speech-to-Text Transcriber**) hiệu năng cao với giao diện trực quan xây dựng trên **Streamlit** (phong cách Glassmorphism hiện đại). 

Dự án hỗ trợ xử lý hàng đợi ngầm đa tác vụ ([`QueueManager`](utils/queue_manager.py)), hiển thị tiến trình thời gian thực, và xuất kết quả đa định dạng (Markdown, Word DOCX, ZIP kèm/không kèm timestamps).

---

## 🏗️ Kiến trúc & Hỗ trợ phần cứng

VietWhisper hỗ trợ 2 nền tảng phần cứng chính:
1. **macOS Apple Silicon (M1/M2/M3/M4)**: Tối ưu hoá thông qua thư viện `mlx-whisper` (Apple Metal/Neural Engine).
2. **Ubuntu / Linux với Card đồ họa NVIDIA (RTX 2060, CUDA)**: Sử dụng backend `faster-whisper` (CTranslate2) hoặc PyTorch CUDA để tăng tốc độ nhận diện trên GPU.

---

## 🖥️ Hướng dẫn cài đặt trên Ubuntu (NVIDIA RTX 2060 / CUDA)

> [!NOTE]
> Card đồ họa **NVIDIA GeForce RTX 2060** trang bị **6GB VRAM**. Để tránh lỗi CUDA Out of Memory (OOM), cấu hình khuyến nghị:
> - **Mô hình khuyến nghị**: `large-v3` với `compute_type="float16"` hoặc `compute_type="int8_float16"` (~2.5GB - 4.5GB VRAM), hoặc `medium`/`small` (~1.5GB - 2.5GB VRAM).

### 1. Cài đặt các gói hệ thống
Cài đặt `ffmpeg` (bắt buộc để xử lý âm thanh) và kiểm tra driver NVIDIA:
```bash
sudo apt update
sudo apt install -y ffmpeg

# Kiểm tra driver GPU và phiên bản CUDA
nvidia-smi
```

### 2. Thiết lập môi trường Python

#### Cách 1: Sử dụng Conda (Khuyên dùng)
```bash
# Tạo môi trường từ file environment-cuda.yml
conda env create -f environment-cuda.yml

# Kích hoạt môi trường
conda activate whisper-cuda

# Cài đặt PyTorch hỗ trợ CUDA (nếu chưa cài):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### Cách 2: Sử dụng venv
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Cài đặt PyTorch với CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện cần thiết
pip install -r requirements-cuda.txt
```

---

## 🤖 Hướng dẫn dành cho AI Agent / Lập trình viên tiếp quản trên Ubuntu

Dự án có cấu trúc module hoá rõ ràng:
- [`app.py`](app.py): Giao diện web Streamlit (Glassmorphism UI, quản lý upload file, hiển thị trạng thái queue và kết quả).
- [`utils/queue_manager.py`](utils/queue_manager.py): Singleton quản lý hàng đợi xử lý âm thanh ngầm (worker thread), không làm block giao diện Streamlit.
- [`utils/transcriber.py`](utils/transcriber.py): Module thực thi nhận diện âm thanh. Hiện tại chứa logic tích hợp `mlx_whisper` (dành cho Mac).
- [`utils/exporter.py`](utils/exporter.py): Tiện ích xuất file Markdown, DOCX, ZIP và định dạng timestamps.
- [`tests/`](tests/): Bộ unit test với pytest.

### Gợi ý chuyển đổi backend sang `faster-whisper` trên Ubuntu:
Khi triển khai trên Ubuntu với card RTX 2060, Agent/Developer có thể điều chỉnh [`utils/transcriber.py`](utils/transcriber.py) sang backend `faster-whisper`:
```python
from faster_whisper import WhisperModel

# Khởi tạo mô hình tối ưu cho RTX 2060 6GB VRAM
model = WhisperModel(
    "large-v3", 
    device="cuda", 
    compute_type="float16" # hoặc "int8_float16" để tiết kiệm thêm VRAM
)

segments, info = model.transcribe(
    file_path, 
    language="vi", 
    beam_size=5
)
```

---

## 🍏 Hướng dẫn cài đặt trên macOS (Apple Silicon M1/M2/M3)

1. **Tạo môi trường Conda:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Kích hoạt môi trường:**
   ```bash
   conda activate whisper-mac
   ```

3. **Cài đặt thư viện (nếu cần cập nhật):**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Khởi chạy ứng dụng

Kích hoạt môi trường tương ứng (`whisper-cuda` trên Ubuntu hoặc `whisper-mac` trên macOS), sau đó chạy:

```bash
streamlit run app.py --server.port 8501
```

Mở trình duyệt tại địa chỉ: `http://localhost:8501`

---

## 🧪 Chạy Kiểm thử (Unit Tests)

Bộ kiểm thử đã được cấu hình đường dẫn chuẩn trong `pytest.ini`. Để chạy kiểm thử:

```bash
pytest
```
hoặc:
```bash
python -m pytest
```

---

## 📦 Xuất dữ liệu & Định dạng hỗ trợ
- **Âm thanh đầu vào**: `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`.
- **Định dạng xuất**:
  - Markdown (`.md`)
  - Microsoft Word (`.docx`)
  - File nén tổng hợp (`.zip`) cho nhiều file cùng lúc
  - Tùy chọn hiển thị hoặc ẩn Timestamp (`[HH:MM:SS]`).
