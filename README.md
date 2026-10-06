# VietWhisper 🎙️

**VietWhisper** là ứng dụng chuyển đổi âm thanh tiếng Việt sang văn bản (**Vietnamese Speech-to-Text Transcriber**) hiệu năng cao với giao diện trực quan xây dựng trên **Streamlit** (phong cách Glassmorphism hiện đại). 

Dự án hỗ trợ xử lý hàng đợi ngầm đa tác vụ ([`QueueManager`](utils/queue_manager.py)), hiển thị tiến trình thời gian thực, và xuất kết quả đa định dạng (Markdown, Word DOCX, ZIP kèm/không kèm timestamps).

---

## 🏗️ Kiến trúc & Hỗ trợ phần cứng

VietWhisper hỗ trợ 2 nền tảng phần cứng chính:
1. **macOS Apple Silicon (M1/M2/M3/M4)**: Tối ưu hoá thông qua thư viện `mlx-whisper` (Apple Metal/Neural Engine).
2. **Ubuntu / Linux với Card đồ họa NVIDIA (RTX 2060, CUDA)**: Sử dụng backend `faster-whisper` (CTranslate2) hoặc PyTorch CUDA để tăng tốc độ nhận diện trên GPU.

---

## 🖥️ Hướng dẫn cài đặt trên PC / Ubuntu (NVIDIA RTX 2060 12GB VRAM / CUDA)

> [!TIP]
> Card đồ họa **NVIDIA GeForce RTX 2060 12GB VRAM** sở hữu kiến trúc Turing với Tensor Cores FP16 mạnh mẽ. Với dung lượng 12GB VRAM, hệ thống vận hành cực kỳ tối ưu:
> - **Mô hình khuyến nghị**: `large-v3` với `compute_type="float16"` (~3.5GB - 5.0GB VRAM) cho chất lượng nhận diện tiếng Việt cao nhất, hoặc `large-v3-turbo` (~2.5GB VRAM) cho tốc độ siêu nhanh.
> - **Tối ưu VRAM & Ngăn ngừa OOM (BatchedInferencePipeline)**: Ứng dụng tích hợp `BatchedInferencePipeline` từ `faster-whisper` với kích thước lô (batch size `8` hoặc `16`), giúp tăng tốc xử lý gấp 3-5 lần và tận dụng hiệu quả 12GB VRAM.
> - **Chống phân mảnh bộ nhớ (CUB Caching Allocator)**: Hệ thống tự động cấu hình `CT2_CUDA_ALLOCATOR=cub_caching` giúp tái sử dụng khối bộ nhớ CUDA, xử lý mượt mà các tệp ghi âm cực dài (trên 3-4 giờ liên tục) mà không bị lỗi tràn bộ nhớ `CUDA failed with error out of memory`.

### 1. Cài đặt các gói hệ thống
Cài đặt `ffmpeg` (bắt buộc để xử lý âm thanh) và kiểm tra driver NVIDIA:
```bash
sudo apt update
sudo apt install -y ffmpeg

# Kiểm tra driver GPU và dung lượng VRAM
nvidia-smi
```

### 2. Thiết lập môi trường Python Conda (Khuyên dùng)

```bash
# 1. Tạo môi trường từ file environment-cuda.yml
conda env create -f environment-cuda.yml

# Hoặc tạo thủ công:
# conda create -n whisper-cuda python=3.10 -y
# conda activate whisper-cuda
# pip install -r requirements-cuda.txt

# 2. Kích hoạt môi trường
conda activate whisper-cuda
```

---

## 🤖 Kiến trúc xử lý đa nền tảng (Dual-Backend)

Dự án được thiết kế module hoá rõ ràng, tự động nhận diện phần cứng khi khởi chạy:
- [`app.py`](app.py): Giao diện web Streamlit (Glassmorphism UI, tự động hiển thị card GPU, quản lý upload file, hiển thị trạng thái queue và kết quả).
- [`utils/transcriber.py`](utils/transcriber.py): Module thực thi nhận diện âm thanh đa nền tảng:
  - **Trên PC với GPU NVIDIA (RTX 2060)**: Tự động chạy backend `faster-whisper` (CTranslate2) với CUDA và Tensor Cores FP16, kèm bộ lọc VAD (Voice Activity Detection) khử nhiễu/khoảng lặng.
  - **Trên macOS Apple Silicon**: Tự động chuyển sang backend `mlx-whisper` với tăng tốc Metal.
- [`utils/queue_manager.py`](utils/queue_manager.py): Singleton quản lý hàng đợi xử lý âm thanh ngầm (worker thread), không làm block giao diện Streamlit.
- [`utils/exporter.py`](utils/exporter.py): Tiện ích xuất file Markdown, DOCX, ZIP và định dạng timestamps.
- [`tests/`](tests/): Bộ kiểm thử tự động toàn diện với pytest.

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
