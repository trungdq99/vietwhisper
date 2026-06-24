import mlx_whisper
import mlx.core as mx
import os
import gc
import subprocess
import streamlit as st

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

@st.cache_resource
def get_whisper_model(model_name: str):
    """Load và cache mô hình Whisper trong bộ nhớ để tái sử dụng."""
    from mlx_whisper.load_models import load_model
    return load_model(model_name)

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
    
    try:
        get_whisper_model.clear()
    except AttributeError:
        # Trong môi trường test nếu mock không có clear method
        pass
        
    gc.collect()
    mx.metal.clear_cache()
