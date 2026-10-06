# VietWhisper: Vietnamese Audio to Text Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a macOS local Streamlit web application using `mlx-whisper` optimized for Apple Silicon to transcribe Vietnamese audio files sequentially.

**Architecture:** A Streamlit UI captures uploaded audio files and schedules them in a queue, passing each to an MLX-based transcriber module. Transcribed segments are cached in Streamlit's session state and formatted using an exporter module to return Word docx, Markdown, or zipped BytesIO archives directly to the user.

**Tech Stack:** Python 3.10, Streamlit, mlx-whisper, python-docx, Conda/Anaconda environment, pytest.

## Global Constraints

- macOS local Web UI using Streamlit and `mlx-whisper` library optimized for Apple Silicon.
- Python 3.10 with `ffmpeg` installed from `conda-forge`.
- Core library requirements: `streamlit>=1.30.0`, `mlx-whisper>=0.1.0`, `python-docx>=1.1.0`, `numpy>=1.24.0`, `pytest>=7.4.0`, `pytest-mock>=3.12.0`.
- Supported audio file extensions: `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`.
- Max upload size of 1GB (`maxUploadSize = 1024` in Streamlit server configuration).
- Midnight Blue background `#0F0F23` with Glassmorphic styles (`rgba(255, 255, 255, 0.05)`, `backdrop-filter: blur(15px)`, viền `border: 1px solid rgba(255, 255, 255, 0.1)`).
- CPU/GPU cache cleanup after each execution: `mx.metal.clear_cache()` and `gc.collect()`.

---

### Task 1: Environment Setup & Project Scaffolding

**Files:**
- Create: `environment.yml`
- Create: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `.gitignore`

**Interfaces:**
- Consumes: None
- Produces: Project dependencies, environment, base configurations, and `.gitignore`.

- [ ] **Step 1: Write `environment.yml` and `requirements.txt`**

Create `environment.yml`:
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

Create `requirements.txt`:
```text
streamlit>=1.30.0
mlx>=0.13.0
mlx-whisper>=0.1.0
python-docx>=1.1.0
numpy>=1.24.0,<2
pytest>=7.4.0
pytest-mock>=3.12.0
```

- [ ] **Step 2: Create Streamlit configuration and theme**

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF7A00"
backgroundColor = "#0F0F23"
secondaryBackgroundColor = "rgba(255, 255, 255, 0.05)"
textColor = "#FFFFFF"
font = "sans serif"

[server]
maxUploadSize = 1024
```

- [ ] **Step 3: Create `.gitignore` file**

Create `.gitignore`:
```text
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
*.egg-info/
dist/
build/
.DS_Store
*.m4a
*.mp3
*.wav
*.flac
*.ogg
*.zip
*.docx
*.md
```

- [ ] **Step 4: Create Conda environment**

Run:
```bash
conda env create -f environment.yml
```
Expected: Conda environment `whisper-mac` is successfully created with Python 3.10 and all required packages.

- [ ] **Step 5: Verify environment dependencies**

Run:
```bash
conda activate whisper-mac
python -c "import streamlit, docx, numpy, pytest; print('Dependencies verified!')"
```
Expected: `Dependencies verified!` output.

- [ ] **Step 6: Commit**

Run:
```bash
git add environment.yml requirements.txt .streamlit/config.toml .gitignore
git commit -m "chore: setup environment environment.yml requirements.txt and streamlit configs"
```

---

### Task 2: Utility: Transcriber Module

**Files:**
- Create: `utils/transcriber.py`
- Create: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: Environment dependencies (`mlx-whisper`, `mlx.core`, `streamlit`)
- Produces:
  - `validate_audio_file(file_path: str) -> None`
  - `get_whisper_model(model_name: str) -> Any`
  - `transcribe_audio(file_path: str, model_name: str) -> dict`
  - `force_clear_gpu_cache() -> None`

- [ ] **Step 1: Write the failing tests in `tests/test_transcriber.py`**

Create `tests/test_transcriber.py`:
```python
import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Mock mlx and mlx_whisper modules for non-macOS/non-MLX testing environment compatibility
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()
sys.modules['mlx_whisper'] = MagicMock()
sys.modules['mlx_whisper.transcribe'] = MagicMock()

from utils.transcriber import validate_audio_file, transcribe_audio, force_clear_gpu_cache

def test_validate_audio_file_non_existent():
    with pytest.raises(FileNotFoundError):
        validate_audio_file("non_existent_file.mp3")

def test_validate_audio_file_invalid_extension(tmp_path):
    temp_file = tmp_path / "test.txt"
    temp_file.write_text("dummy")
    with pytest.raises(ValueError, match="Định dạng file không được hỗ trợ"):
        validate_audio_file(str(temp_file))

def test_validate_audio_file_success(tmp_path):
    temp_file = tmp_path / "test.mp3"
    temp_file.write_bytes(b"dummy audio content")
    # Stub ffprobe check to pass
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        validate_audio_file(str(temp_file))

@patch("mlx_whisper.transcribe")
@patch("mlx.core.metal.clear_cache")
@patch("utils.transcriber.validate_audio_file")
def test_transcribe_audio(mock_validate, mock_clear_cache, mock_transcribe, tmp_path):
    temp_file = tmp_path / "test.mp3"
    temp_file.write_text("dummy")
    
    mock_transcribe.return_value = {"text": "Xin chào", "segments": []}
    
    result = transcribe_audio(str(temp_file), "mlx-community/whisper-base-4bit")
    
    assert result == {"text": "Xin chào", "segments": []}
    mock_validate.assert_called_once_with(str(temp_file))
    mock_transcribe.assert_called_once_with(
        str(temp_file),
        path_or_hf_repo="mlx-community/whisper-base-4bit",
        verbose=False,
        language="vi",
        no_speech_threshold=0.6,
        logprob_threshold=-1.5,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False
    )
    mock_clear_cache.assert_called_once()

@patch("mlx_whisper.transcribe.ModelHolder")
@patch("mlx.core.metal.clear_cache")
def test_force_clear_gpu_cache(mock_clear_cache, mock_model_holder):
    force_clear_gpu_cache()
    assert mock_model_holder.model is None
    assert mock_model_holder.model_path is None
    mock_clear_cache.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_transcriber.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'utils'` (or similar import error since the transcriber code doesn't exist).

- [ ] **Step 3: Implement minimal code in `utils/transcriber.py`**

Create `utils/transcriber.py`:
```python
import mlx_whisper
import mlx.core as mx
import os
import gc
import subprocess

def validate_audio_file(file_path: str):
    """Kiểm tra định dạng, dung lượng và tính toàn vẹn của tệp âm thanh."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
        
    valid_exts = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}
    _, ext = os.path.splitext(file_path.lower())
    if ext not in valid_exts:
        raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")
        
    # Kiểm tra tệp rỗng
    if os.path.getsize(file_path) == 0:
        raise ValueError("Tệp âm thanh bị trống (0 bytes)")
        
    # Kiểm tra tính toàn vẹn bằng ffprobe (đã có sẵn trong conda environment)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise ValueError("Tệp tin âm thanh bị hỏng hoặc không có dữ liệu hợp lệ.")
    except FileNotFoundError:
        # Nếu máy dev chưa cài ffprobe, bỏ qua kiểm tra sâu để tránh crash ứng dụng
        pass
    except Exception as e:
        raise ValueError(f"Kiểm tra tính toàn vẹn thất bại: {str(e)}")

def transcribe_audio(file_path: str, model_name: str = "mlx-community/whisper-large-v3-4bit") -> dict:
    """
    Gọi thư viện mlx-whisper để nhận diện tiếng Việt từ file audio.
    Giải phóng Metal cache sau khi chạy để tránh lỗi OOM.
    """
    validate_audio_file(file_path)
    
    try:
        # Cấu hình các tham số lọc khoảng lặng và phòng lặp từ ảo giác (hallucination)
        result = mlx_whisper.transcribe(
            file_path,
            path_or_hf_repo=model_name,
            verbose=False,
            language="vi",
            no_speech_threshold=0.6,
            logprob_threshold=-1.5,      # Nới lỏng hơn cho tiếng Việt
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False # Ngăn chặn hiện tượng lặp từ
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
    gc.collect()
    mx.metal.clear_cache()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_transcriber.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add utils/transcriber.py tests/test_transcriber.py
git commit -m "feat: add transcriber utility module with audio validation and cache handling"
```

---

### Task 3: Utility: Exporter Module

**Files:**
- Create: `utils/exporter.py`
- Create: `tests/test_exporter.py`

**Interfaces:**
- Consumes: `python-docx`, `zipfile`, `io`
- Produces:
  - `format_timestamp(seconds: float) -> str`
  - `export_to_markdown(segments: list, include_timestamps: bool) -> io.BytesIO`
  - `export_to_docx(segments: list, include_timestamps: bool) -> io.BytesIO`
  - `export_to_zip(files: dict) -> io.BytesIO`

- [ ] **Step 1: Write the failing tests in `tests/test_exporter.py`**

Create `tests/test_exporter.py`:
```python
import io
import pytest
from utils.exporter import format_timestamp, export_to_markdown, export_to_docx, export_to_zip

def test_format_timestamp_valid():
    assert format_timestamp(0.0) == "[00:00:00.000]"
    assert format_timestamp(3661.123) == "[01:01:01.123]"
    assert format_timestamp(59.999) == "[00:00:59.999]"

def test_format_timestamp_negative():
    with pytest.raises(ValueError, match="Thời gian không được âm"):
        format_timestamp(-1.0)

def test_export_to_markdown_with_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_markdown(segments, include_timestamps=True)
    content = bio.getvalue().decode("utf-8")
    assert "**[00:00:00.000]** Xin chào." in content
    assert "**[00:00:03.000]** Tôi là trợ lý AI." in content

def test_export_to_markdown_without_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_markdown(segments, include_timestamps=False)
    content = bio.getvalue().decode("utf-8")
    assert "Xin chào." in content
    assert "Tôi là trợ lý AI." in content
    assert "[" not in content

def test_export_to_docx_with_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_docx(segments, include_timestamps=True)
    assert bio.getvalue().startswith(b"PK")  # DOCX files are zip archives under the hood

def test_export_to_docx_without_timestamps():
    segments = [
        {"start": 0.0, "end": 2.5, "text": " Xin chào."},
        {"start": 3.0, "end": 5.5, "text": " Tôi là trợ lý AI."}
    ]
    bio = export_to_docx(segments, include_timestamps=False)
    assert bio.getvalue().startswith(b"PK")

def test_export_to_zip():
    files = {
        "file1.md": io.BytesIO(b"Hello MD"),
        "file2.docx": io.BytesIO(b"Hello DOCX")
    }
    zip_bio = export_to_zip(files)
    assert zip_bio.getvalue().startswith(b"PK")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_exporter.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.exporter'`.

- [ ] **Step 3: Implement minimal code in `utils/exporter.py`**

Create `utils/exporter.py`:
```python
import io
import zipfile
import docx

def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("Thời gian không được âm")
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"[{h:02d}:{m:02d}:{s:02d}.{ms:03d}]"

def export_to_markdown(segments: list, include_timestamps: bool) -> io.BytesIO:
    bio = io.BytesIO()
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if include_timestamps:
            ts = format_timestamp(seg.get("start", 0.0))
            lines.append(f"**{ts}** {text}")
        else:
            lines.append(text)
    content = "\n\n".join(lines)
    bio.write(content.encode("utf-8"))
    bio.seek(0)
    return bio

def export_to_docx(segments: list, include_timestamps: bool) -> io.BytesIO:
    doc = docx.Document()
    doc.add_heading("Kết quả chuyển đổi âm thanh", level=1)
    
    if include_timestamps:
        table = doc.add_table(rows=1, cols=2)
        try:
            table.style = 'Light Shading Accent 1'
        except KeyError:
            table.style = 'Table Grid' # Fallback style
            
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Mốc thời gian'
        hdr_cells[1].text = 'Văn bản'
        
        for seg in segments:
            row_cells = table.add_row().cells
            row_cells[0].text = format_timestamp(seg.get("start", 0.0))
            row_cells[1].text = seg.get("text", "").strip()
    else:
        for seg in segments:
            doc.add_paragraph(seg.get("text", "").strip())
            
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def export_to_zip(files: dict) -> io.BytesIO:
    zip_bio = io.BytesIO()
    with zipfile.ZipFile(zip_bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data.getvalue())
    zip_bio.seek(0)
    return zip_bio
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_exporter.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add utils/exporter.py tests/test_exporter.py
git commit -m "feat: add exporter module with markdown, word and zip support"
```

---

### Task 4: Main Application: Streamlit Frontend Interface

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `utils/transcriber.py`, `utils/exporter.py`
- Produces: Streamlit Frontend Web App

- [ ] **Step 1: Write the failing test in `tests/test_app.py`**

Create `tests/test_app.py`:
```python
import sys
from unittest.mock import MagicMock

# Mock mlx and mlx_whisper modules for non-macOS/non-MLX testing environment compatibility
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()
sys.modules['mlx_whisper'] = MagicMock()
sys.modules['mlx_whisper.load_models'] = MagicMock()

import pytest
from streamlit.testing.v1 import AppTest

def test_app_renders():
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) > 0
    assert "VietWhisper" in at.title[0].value
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_app.py -v
```
Expected: FAIL with `FileNotFoundError: app.py not found`.

- [ ] **Step 3: Write minimal implementation in `app.py`**

Create `app.py`:
```python
import streamlit as st
import tempfile
import os
import io
from utils.transcriber import transcribe_audio, force_clear_gpu_cache
from utils.exporter import export_to_markdown, export_to_docx, export_to_zip, format_timestamp

# 1. Page Configuration
st.set_page_config(
    page_title="VietWhisper - Chuyển đổi Âm thanh tiếng Việt sang Văn bản",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styles (Glassmorphism & Aesthetics)
glass_css = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* Reset and custom font */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0F0F23;
        background-image: radial-gradient(circle at 10% 20%, rgba(255, 122, 0, 0.08) 0%, transparent 40%),
                          radial-gradient(circle at 90% 80%, rgba(255, 0, 122, 0.08) 0%, transparent 40%);
        background-attachment: fixed;
        color: #FFFFFF;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Overwrite Streamlit container default styling with Glassmorphism */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.07) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF7A00 0%, #FF007A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 122, 0, 0.2);
        margin-bottom: 8px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 30px;
    }
    
    /* Button Custom styling */
    div.stButton > button {
        background: linear-gradient(135deg, #FF7A00 0%, #FF007A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(255, 122, 0, 0.3) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 122, 0, 0.5) !important;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .status-pending { background: rgba(255, 193, 7, 0.15); color: #FFC107; border: 1px solid rgba(255, 193, 7, 0.3); }
    .status-processing { background: rgba(0, 123, 255, 0.15); color: #007BFF; border: 1px solid rgba(0, 123, 255, 0.3); }
    .status-done { background: rgba(40, 167, 69, 0.15); color: #28A745; border: 1px solid rgba(40, 167, 69, 0.3); }
    .status-error { background: rgba(220, 53, 69, 0.15); color: #DC3545; border: 1px solid rgba(220, 53, 69, 0.3); }
    
    .timestamp-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #FF7A00;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-right: 10px;
    }
    
    .transcript-line {
        margin-bottom: 12px;
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.02);
    }
    
    .scroll-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    .scroll-container::-webkit-scrollbar {
        width: 6px;
    }
    .scroll-container::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)

# 3. Session State Initialization
if "queue" not in st.session_state:
    st.session_state.queue = []
if "results" not in st.session_state:
    st.session_state.results = {}
if "processing" not in st.session_state:
    st.session_state.processing = False

# 4. Header Section
st.title("🎙️ VietWhisper")
st.markdown('<p class="subtitle">Ứng dụng chuyển đổi âm thanh tiếng Việt sang văn bản chất lượng cao, tối ưu Apple Silicon GPU</p>', unsafe_allow_html=True)

# 5. UI Layout - Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình hệ thống")
    
    model_name = st.selectbox(
        "Mô hình Whisper (MLX):",
        options=[
            "mlx-community/whisper-large-v3-4bit",
            "mlx-community/whisper-large-v3-turbo-4bit",
            "mlx-community/whisper-base-4bit",
            "mlx-community/whisper-large-v3"
        ],
        index=0
    )
    
    export_format = st.selectbox(
        "Định dạng xuất:",
        options=[".docx", ".md"],
        index=0
    )
    
    include_timestamps = st.toggle("Bật mốc thời gian (Timestamps)", value=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Giải phóng GPU Metal Cache", key="clear_gpu_btn"):
        force_clear_gpu_cache()
        st.success("Đã giải phóng bộ nhớ cache của mô hình và Metal GPU.")

# 6. Main UI Content - File Upload & Processing
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📤 Tải lên tệp âm thanh")
    
    uploaded_files = st.file_uploader(
        "Kéo thả các file âm thanh vào đây (Hỗ trợ: mp3, m4a, wav, flac, ogg - Tối đa 1GB):",
        type=["mp3", "m4a", "wav", "flac", "ogg"],
        accept_multiple_files=True
    )
    
    # Synchronize uploaded files and session queue to prevent progress wipeouts on rerun
    uploaded_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    current_queue_names = [item["name"] for item in st.session_state.queue]
    
    if uploaded_file_names != current_queue_names:
        st.session_state.queue = []
        if uploaded_files:
            for uf in uploaded_files:
                status = "Chờ xử lý"
                if uf.name in st.session_state.results:
                    status = "Hoàn thành"
                st.session_state.queue.append({"name": uf.name, "file": uf, "status": status})
                
    if uploaded_files:
        queue_placeholder = st.empty()
        
        def render_queue():
            with queue_placeholder.container():
                st.markdown("#### Hàng đợi xử lý:")
                for item in st.session_state.queue:
                    status_class = "status-pending"
                    if item["status"] == "Đang xử lý":
                        status_class = "status-processing"
                    elif item["status"] == "Hoàn thành":
                        status_class = "status-done"
                    elif item["status"] == "Lỗi":
                        status_class = "status-error"
                        
                    st.markdown(
                        f'<div><span class="status-badge {status_class}">{item["status"]}</span> {item["name"]}</div>',
                        unsafe_allow_html=True
                    )
        
        render_queue()
        start_btn = st.button("🚀 Bắt đầu nhận diện")
        
        if start_btn and not st.session_state.processing:
            st.session_state.processing = True
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, item in enumerate(st.session_state.queue):
                if item["status"] == "Hoàn thành":
                    continue
                
                # Update status and render UI live at the spot
                st.session_state.queue[idx]["status"] = "Đang xử lý"
                render_queue()
                status_text.text(f"Đang xử lý: {item['name']}...")
                
                # Write uploaded file in chunks to prevent OOM
                suffix = os.path.splitext(item["name"])[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    CHUNK_SIZE = 1024 * 1024  # 1MB
                    file_data = item["file"]
                    file_data.seek(0)
                    while chunk := file_data.read(CHUNK_SIZE):
                        tmp_file.write(chunk)
                    tmp_file_path = tmp_file.name
                
                try:
                    result = transcribe_audio(tmp_file_path, model_name=model_name)
                    st.session_state.results[item["name"]] = result
                    st.session_state.queue[idx]["status"] = "Hoàn thành"
                except Exception as e:
                    st.session_state.queue[idx]["status"] = "Lỗi"
                    st.error(f"Lỗi khi xử lý file {item['name']}: {str(e)}")
                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
                
                render_queue()
                progress = int((idx + 1) / len(st.session_state.queue) * 100)
                progress_bar.progress(progress)
            
            st.session_state.processing = False
            status_text.text("Đã hoàn thành nhận dạng tất cả các tệp âm thanh.")
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

with col_right:
    st.markdown("### 📝 Kết quả & Tải về")
    
    if st.session_state.results:
        selected_file_name = st.selectbox(
            "Chọn file để xem kết quả:",
            options=list(st.session_state.results.keys())
        )
        
        if selected_file_name:
            res = st.session_state.results[selected_file_name]
            segments = res.get("segments", [])
            
            # Formatted Preview
            st.markdown("#### Xem trước đoạn hội thoại:")
            st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
            
            html_transcript = []
            plain_text_lines = []
            for seg in segments:
                ts = format_timestamp(seg.get("start", 0.0))
                text = seg.get("text", "").strip()
                
                if include_timestamps:
                    html_line = f'<div class="transcript-line"><span class="timestamp-badge">{ts}</span>{text}</div>'
                    plain_line = f"{ts} {text}"
                else:
                    html_line = f'<div class="transcript-line">{text}</div>'
                    plain_line = text
                
                html_transcript.append(html_line)
                plain_text_lines.append(plain_line)
                
            st.markdown("\n".join(html_transcript), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clipboard Copy native component
            full_plain_text = "\n".join(plain_text_lines)
            st.markdown("#### Bản thô để sao chép nhanh (Nhấp nút copy ở góc trên bên phải):")
            st.code(full_plain_text, language="text")
            
            # Lazy export formatting
            export_files = {}
            for name, r in st.session_state.results.items():
                segs = r.get("segments", [])
                if export_format == ".docx":
                    out_bio = export_to_docx(segs, include_timestamps)
                    export_files[os.path.splitext(name)[0] + ".docx"] = out_bio
                else:
                    out_bio = export_to_markdown(segs, include_timestamps)
                    export_files[os.path.splitext(name)[0] + ".md"] = out_bio
            
            single_file_name = os.path.splitext(selected_file_name)[0] + export_format
            single_file_data = export_files[single_file_name]
            
            st.download_button(
                label=f"📥 Tải xuống tệp {single_file_name}",
                data=single_file_data,
                file_name=single_file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document" if export_format == ".docx" else "text/markdown"
            )
            
            if len(st.session_state.results) > 1:
                zip_bio = export_to_zip(export_files)
                st.download_button(
                    label="🗜️ Tải xuống toàn bộ tệp (.zip)",
                    data=zip_bio,
                    file_name="vietwhisper_transcripts.zip",
                    mime="application/zip"
                )
    else:
        st.info("Chưa có kết quả. Vui lòng tải file lên và ấn nút Bắt đầu nhận diện.")

## Verification Plan

### Automated Tests

Activate the environment and run pytest across the suite:
```bash
conda activate whisper-mac
pytest tests/ -v
```

Expected outputs:
```text
tests/test_transcriber.py::test_validate_audio_file_non_existent PASSED
tests/test_transcriber.py::test_validate_audio_file_invalid_extension PASSED
tests/test_transcriber.py::test_validate_audio_file_success PASSED
tests/test_transcriber.py::test_get_whisper_model PASSED
tests/test_transcriber.py::test_transcribe_audio PASSED
tests/test_transcriber.py::test_force_clear_gpu_cache PASSED
tests/test_exporter.py::test_format_timestamp_valid PASSED
tests/test_exporter.py::test_format_timestamp_negative PASSED
tests/test_exporter.py::test_export_to_markdown_with_timestamps PASSED
tests/test_exporter.py::test_export_to_markdown_without_timestamps PASSED
tests/test_exporter.py::test_export_to_docx_with_timestamps PASSED
tests/test_exporter.py::test_export_to_docx_without_timestamps PASSED
tests/test_exporter.py::test_export_to_zip PASSED
tests/test_app.py::test_app_renders PASSED
```

### Manual & Robustness Verification

1. **Verify Environment Setup:**
   Confirm that `ffmpeg` is available on the path and runs successfully:
   ```bash
   ffmpeg -version
   ```

2. **Run the Streamlit Application:**
   Execute the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` and verify:
   - The UI correctly displays with the high-end Glassmorphic Dark theme (Midnight Blue background, radial-glow accents, translucent cards).
   - Change models and configurations in the sidebar.
   - Upload small/large audio files, and check that the queue shows status badges correctly.
   - Run transcription and check live updates of the status indicators and progress bar.
   - Verify that output previews render properly with timestamp badges.
   - Verify code container's native copy-to-clipboard button copies text correctly.
   - Click "Giải phóng GPU Metal Cache" and verify success message.
   - Test downloading `.docx`, `.md`, and `.zip` archives.

3. **Verify Corrupt / Empty File Rejection:**
   - Prepare a 0-byte file: `touch zero_byte.mp3`.
   - Prepare a corrupt text file disguised as audio: `echo "not an audio" > fake.mp3`.
   - Upload both files in Streamlit.
   - Click "Bắt đầu nhận diện" and verify they are marked as `Lỗi` with appropriate error messages in the UI.

4. **Verify Queue State Resiliency:**
   - Upload 3 files.
   - Start the transcription.
   - While file 1 is transcribing, toggle the "Bật mốc thời gian (Timestamps)" or change the "Định dạng xuất" in the sidebar.
   - Verify that the transcription queue is **not** wiped out or reset, and the current run continues processing correctly.

5. **Verify Memory Recovery:**
   - Monitor macOS memory usage in Terminal: `top -o mem` or via Activity Monitor.
   - Transcribe a large audio file.
   - Note the memory spike.
   - Click "Giải phóng GPU Metal Cache" in the sidebar.
   - Verify that memory drops back to near baseline levels.
