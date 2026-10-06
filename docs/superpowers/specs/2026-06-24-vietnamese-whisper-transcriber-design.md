# Đặc tả Thiết kế: Ứng dụng Chuyển đổi Âm thanh tiếng Việt sang Văn bản (VietWhisper)

Tài liệu này đặc tả thiết kế chi tiết cho dự án **VietWhisper** - ứng dụng Web chạy cục bộ trên macOS sử dụng Streamlit và thư viện `mlx-whisper` tối ưu cho Apple Silicon.

---

## 1. Mục tiêu Dự án (Project Goal)
*   Xây dựng một ứng dụng Web UI cục bộ dễ dùng bằng Streamlit để nhận diện tiếng Việt từ các file âm thanh đa định dạng (m4a, mp3, wav, flac...).
*   Sử dụng các mô hình **Whisper** tối ưu hóa thông qua framework **MLX** (`mlx-whisper`) giúp chạy trực tiếp trên GPU của chip Apple Silicon.
*   Hỗ trợ người dùng lựa chọn linh hoạt kích thước mô hình (lượng tử hóa 4-bit hoặc bản đầy đủ) để tối ưu hóa tài nguyên RAM 16GB trên macOS.
*   Hỗ trợ xuất kết quả ra định dạng Word (`.docx`) hoặc Markdown (`.md`), hỗ trợ bật/tắt mốc thời gian (timestamps) với độ chính xác cao và xử lý nhiều file tuần tự bằng hàng đợi, tránh tràn bộ nhớ GPU.

---

## 2. Thiết kế Giao diện (UI/UX Specification - Glassmorphism Style)
Ứng dụng sử dụng giao diện tối (Dark Mode) làm chủ đạo kết hợp hiệu ứng kính mờ (Glassmorphism) để đem lại trải nghiệm cao cấp trên macOS.

### 2.1. Phối màu & Phông chữ (Colors & Typography)
*   **Font Family**: `Inter` (được tải trực tiếp từ Google Fonts).
*   **Màu nền chính**: Midnight Blue `#0F0F23` kết hợp quầng sáng mờ (`radial-gradient` với các đốm sáng mờ - glow orbs màu xanh/cam phía sau) để làm nổi bật hiệu ứng chiều sâu cho lớp kính mờ.
*   **Kính mờ (Glass Cards)**: Lớp phủ `rgba(255, 255, 255, 0.05)` với `backdrop-filter: blur(15px)` và viền mỏng `border: 1px solid rgba(255, 255, 255, 0.1)`. Reset CSS các container mặc định của Streamlit thành `transparent !important`.
*   **Nút nhấn (CTA Button)**: Gradient Cam ấm `#FF7A00` sang Hồng tím, thay đổi kích thước nhẹ khi hover.

### 2.2. Bố cục Widget trong Streamlit
1.  **Cấu hình Theme gốc**: Thiết lập trước Dark Mode trong `.streamlit/config.toml` để tránh hiện tượng giật sáng khi load trang.
2.  **Tiêu đề**: Dạng chữ lớn có bóng mờ (shadow glow).
3.  **Hộp lựa chọn mô hình (`st.selectbox`)**: Cho phép người dùng chọn:
    *   `mlx-community/whisper-large-v3-4bit` (Lớn, lượng tử hóa 4-bit, khuyên dùng)
    *   `mlx-community/whisper-large-v3-turbo-4bit` (Turbo, lượng tử hóa 4-bit, tối ưu tốc độ)
    *   `mlx-community/whisper-base-4bit` (Nhỏ, lượng tử hóa 4-bit, để thử nghiệm nhanh)
    *   `mlx-community/whisper-large-v3` (Lớn, bản gốc FP16, độ chính xác cao nhất, yêu cầu nhiều RAM)
4.  **Khung Upload File (`st.file_uploader`)**: Hỗ trợ kéo thả nhiều file cùng lúc, hỗ trợ file âm thanh định dạng: `mp3, m4a, wav, flac, ogg`. Tăng cấu hình `maxUploadSize = 1024` (1GB) để hỗ trợ file tải lên lớn.
5.  **Dropdown định dạng xuất (`st.selectbox`)**: Lựa chọn output là `.docx` hoặc `.md` (mặc định là `.docx`).
6.  **Nút gạt timestamps (`st.toggle`)**: Tuỳ chọn xuất kết quả có mốc thời gian (bật/tắt).
7.  **Nút xử lý (`st.button`)**: Bắt đầu chuyển đổi dữ liệu âm thanh.
8.  **Nút giải phóng GPU**: Cho phép người dùng dọn dẹp bộ nhớ đệm Metal GPU thủ công.
9.  **Khu vực hiển thị kết quả (Results Area)**:
    *   Thanh tiến trình chạy tổng hợp kèm danh sách badge trạng thái (đang chờ, đang chạy, hoàn thành) cho từng file âm thanh.
    *   Xem trước nội dung (Preview): Hiển thị đoạn hội thoại dạng text với các badge mốc thời gian HTML tự thiết kế đẹp mắt trong khung kính mờ cuộn dọc.
    *   Nút **Sao chép vào clipboard** trực tiếp dưới khung preview (sử dụng HTML/JS an toàn qua `st.components.v1.html`).
    *   Nút tải file riêng lẻ (`st.download_button`) và tải toàn bộ dạng `.zip` nếu xử lý nhiều file (tải trực tiếp dưới dạng luồng dữ liệu `io.BytesIO` để không ghi file vật lý xuống ổ đĩa).

---

## 3. Kiến trúc Hệ thống & Luồng Dữ liệu (Backend Logic & Data Flow)

### 3.1. Sơ đồ Luồng dữ liệu (Data Flow Diagram)
```
[User Uploads Audio Files] 
       │
       ▼ (Kiểm tra định dạng & tính hợp lệ của file ở backend)
[Valid Audio Files Queue]
       │
       ▼ (Xử lý tuần tự hàng đợi để tránh GPU OOM)
[Temp Audio In Memory/Tempfile]
       │
       ▼ (Tải/Cache Model từ Hugging Face bằng @st.cache_resource)
[mlx-whisper Model Inference] -> (Sử dụng Model đã chọn trên GPU)
       │
       ▼ (Trả về dict kết quả: text & segments)
[Post-Processing & Caching] -> (Lưu kết quả thô vào st.session_state)
       ├─► (Nếu có timestamps: Định dạng thời gian dạng [HH:MM:SS.mmm])
       └─► (Xuất ra BytesIO dưới dạng docx/md tương ứng)
       │
       ▼ (Nén các luồng BytesIO thành ZIP trong RAM nếu có nhiều file)
[Ready for Download] -> (Truyền trực tiếp BytesIO vào st.download_button trên UI)
```

### 3.2. Cấu trúc Thư mục Dự án (Project Directory Structure)
```
/mmm
├── .streamlit/
│   └── config.toml             # Cấu hình theme tối mặc định cho Streamlit
├── app.py                      # File ứng dụng Streamlit chính (UI + Controller + Session State)
├── utils/
│   ├── __init__.py
│   ├── transcriber.py          # Logic load và cache model, nhận diện âm thanh
│   └── exporter.py             # Logic xuất định dạng file và zip trong RAM (BytesIO)
├── tests/
│   ├── test_transcriber.py     # Unit test cho module nhận diện
│   └── test_exporter.py        # Unit test cho module xuất file
├── environment.yml             # Cấu hình môi trường Anaconda
├── requirements.txt            # Danh sách thư viện Python phụ thuộc
└── .gitignore                  # Bỏ qua tệp tạm, cache python và pytest
```

---

## 4. Đặc tả Kỹ thuật Chi tiết (Technical Specification)

### 4.1. Môi trường phát triển (Anaconda Environment)
Tệp `environment.yml` thiết lập môi trường Python 3.10 có cài sẵn ffmpeg từ kênh `conda-forge` và gọi trực tiếp `requirements.txt` để quản lý các package phụ thuộc tập trung:
*   **environment.yml**:
    ```yaml
    name: whisper-mac
    channels:
      - conda-forge
      - defaults
    dependencies:
      - python=3.10
      - ffmpeg
      - lxml
      - pip
      - pip:
          - -r requirements.txt
    ```
*   **requirements.txt**:
    ```text
    streamlit>=1.30.0
    mlx-whisper>=0.1.0
    python-docx>=1.1.0
    numpy>=1.24.0
    pytest>=7.4.0
    pytest-mock>=3.12.0
    ```

### 4.2. Logic Transcription (`utils/transcriber.py`)
Mã nguồn xử lý load model và nhận diện âm thanh, có tích hợp bộ lọc nhiễu và cache:
```python
import mlx_whisper
import mlx.core as mx
import os
import gc
import streamlit as st

def validate_audio_file(file_path: str):
    """Kiểm tra định dạng và tính toàn vẹn của tệp âm thanh."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    valid_exts = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
    _, ext = os.path.splitext(file_path.lower())
    if ext not in valid_exts:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")

@st.cache_resource
def get_whisper_model(model_name: str):
    """Load và cache mô hình Whisper trong bộ nhớ để tái sử dụng."""
    # Trực tiếp sử dụng mô hình được cache thông qua load_model của mlx_whisper
    from mlx_whisper.load_models import load_model
    return load_model(model_name)

def transcribe_audio(file_path: str, model_name: str = "mlx-community/whisper-large-v3-4bit") -> dict:
    """
    Gọi thư viện mlx-whisper để nhận diện tiếng Việt từ file audio.
    Hỗ trợ cache model và giải phóng Metal cache để tránh lỗi OOM.
    """
    validate_audio_file(file_path)
    
    # Load model từ cache
    model = get_whisper_model(model_name)
    
    try:
        # Cấu hình các tham số lọc khoảng lặng tránh lặp từ vô hạn (hallucination)
        result = mlx_whisper.transcribe(
            file_path,
            path_or_hf_repo=model,
            verbose=False,
            language="vi",
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )
    finally:
        # Giải phóng cache Metal sau khi chạy
        mx.metal.clear_cache()
        gc.collect()
        
    return result

def force_clear_gpu_cache():
    """Giải phóng hoàn toàn bộ nhớ cache của mô hình và Metal GPU."""
    from mlx_whisper.transcribe import ModelHolder
    ModelHolder.model = None
    ModelHolder.model_path = None
    get_whisper_model.clear()
    gc.collect()
    mx.metal.clear_cache()
```

### 4.3. Logic Định dạng & Xuất File (`utils/exporter.py`)
*   **Hàm định dạng thời gian có mili giây**:
    ```python
    def format_timestamp(seconds: float) -> str:
        if seconds < 0:
            raise ValueError("Thời gian không được âm")
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"[{h:02d}:{m:02d}:{s:02d}.{ms:03d}]"
    ```
*   **Xuất file Word (`.docx`) trong bộ nhớ**:
    Sử dụng `docx.Document` ghi vào đối tượng `io.BytesIO`. Nếu có mốc thời gian, tạo bảng gồm 2 cột (Cột 1: Mốc thời gian, Cột 2: Văn bản nói).
*   **Xuất file Markdown (`.md`) trong bộ nhớ**:
    Ghi trực tiếp nội dung văn bản sử dụng mã hóa `utf-8` vào đối tượng `io.BytesIO`. Định dạng mốc thời gian dạng `**[HH:MM:SS.mmm]**`.
*   **Tạo file ZIP lưu trong RAM**:
    Nén danh sách các luồng dữ liệu byte (`io.BytesIO`) thành một file `.zip` hoàn chỉnh trong RAM và truyền trực tiếp vào UI download để không ghi file vật lý ra đĩa cứng.

---

## 5. Kế hoạch Kiểm thử & Xác minh (Verification Plan)

### 5.1. Kiểm thử Tự động (Automated Verification)
Sử dụng bộ công cụ `pytest` để chạy các unit test tự động được định nghĩa trong thư mục `tests/`:
*   `tests/test_transcriber.py`: Kiểm thử việc xử lý ngoại lệ khi file không tồn tại, validate định dạng tệp và chạy mock cho hàm `mlx_whisper.transcribe`.
*   `tests/test_exporter.py`: Kiểm thử logic hàm `format_timestamp` (bao gồm cả trường hợp lỗi thời gian âm và định dạng mili giây), xuất markdown với encoding `utf-8` và tạo tệp nén ZIP từ bộ nhớ đệm.

Chạy lệnh kiểm thử:
```bash
pytest tests/
```

### 5.2. Kiểm thử Thủ công (Manual Verification)
1.  **Kiểm tra môi trường**: Khởi tạo thành công môi trường Anaconda, kiểm tra `ffmpeg` hệ thống.
2.  **Kiểm tra tính năng chọn mô hình**: Thay đổi các mô hình khác nhau từ selectbox và kiểm tra xem ứng dụng có chuyển đổi tương ứng hay không.
3.  **Kiểm tra tính năng nhận diện**:
    *   Tải lên file ghi âm ngắn tiếng Việt `.mp3` để kiểm thử độ chính xác.
    *   Tải lên file lớn `.m4a` từ thư mục dự án và kiểm tra hàng đợi xử lý tuần tự.
4.  **Kiểm tra lưu trạng thái (State Retention)**: Chạy dịch xong một tệp âm thanh, sau đó bật/tắt nút gạt timestamps hoặc đổi định dạng tải xuống (.docx sang .md) và xác minh kết quả cập nhật ngay lập tức mà không chạy lại mô hình (không tăng bộ nhớ GPU).
5.  **Kiểm tra an toàn lỗi**: Ngắt kết nối mạng và thử tải một model chưa có sẵn để kiểm tra xem cảnh báo kết nối internet có hiển thị trực quan không. Tải lên một file không phải audio để kiểm tra thông báo cảnh báo định dạng.
6.  **Theo dõi hiệu năng**: Theo dõi lượng RAM và mức sử dụng GPU Metal bằng Activity Monitor trên Mac để đảm bảo bộ nhớ đệm được giải phóng đúng cách sau khi hoàn thành.
