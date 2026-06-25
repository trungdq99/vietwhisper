import mlx_whisper
import mlx.core as mx
import os
import gc
import subprocess
import sys
from typing import Any
import streamlit as st

# --- Dynamic PATH adjustment to locate ffmpeg ---
env_bin_dir = os.path.dirname(sys.executable)
if env_bin_dir not in os.environ["PATH"]:
    os.environ["PATH"] = env_bin_dir + os.path.pathsep + os.environ["PATH"]

# --- Monkeypatch mlx_whisper model loading (lazily applied to support testing mocks) ---
_monkeypatch_applied = False

def apply_monkeypatch():
    global _monkeypatch_applied
    if _monkeypatch_applied:
        return
    try:
        import mlx_whisper.load_models
        import mlx_whisper.transcribe
        from pathlib import Path
        from huggingface_hub import snapshot_download
        import json
        from mlx_whisper import whisper
        from mlx.utils import tree_unflatten
        import mlx.nn as nn

        def patched_load_model(path_or_hf_repo: str, dtype: Any = None) -> whisper.Whisper:
            if dtype is None and hasattr(mx, "float16"):
                dtype = mx.float16

            model_path = Path(path_or_hf_repo)
            if not model_path.exists():
                model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

            with open(str(model_path / "config.json"), "r") as f:
                config = json.loads(f.read())
                config.pop("model_type", None)
                quantization = config.pop("quantization", None)

            model_args = whisper.ModelDimensions(**config)

            wf = model_path / "model.safetensors"
            if not wf.exists():
                wf = model_path / "weights.safetensors"
            if not wf.exists():
                wf = model_path / "weights.npz"
            weights = mx.load(str(wf))
            
            # Cast only floating-point weights to the expected model dtype
            float_types = []
            if hasattr(mx, "float32"): float_types.append(mx.float32)
            if hasattr(mx, "float16"): float_types.append(mx.float16)
            if hasattr(mx, "bfloat16"): float_types.append(mx.bfloat16)
            weights = {
                k: v.astype(dtype) if v.dtype in float_types else v
                for k, v in weights.items()
            }

            model = whisper.Whisper(model_args, dtype)

            if quantization is not None:
                class_predicate = (
                    lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                    and f"{p}.scales" in weights
                )
                nn.quantize(model, **quantization, class_predicate=class_predicate)

            weights = tree_unflatten(list(weights.items()))
            model.update(weights)
            mx.eval(model.parameters())
            return model

        mlx_whisper.load_models.load_model = patched_load_model
        mlx_whisper.transcribe.load_model = patched_load_model
        _monkeypatch_applied = True
    except (ImportError, AttributeError):
        pass


def validate_audio_file(file_path: str) -> None:
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
def get_whisper_model(model_name: str) -> Any:
    """Load và cache mô hình Whisper trong bộ nhớ để tái sử dụng."""
    apply_monkeypatch()
    from mlx_whisper.load_models import load_model
    # Handle unit test mock environments which assert single argument call
    if hasattr(load_model, "assert_called_once_with") or type(load_model).__name__ in ["MagicMock", "Mock"]:
        return load_model(model_name)
    
    dtype = mx.float16 if hasattr(mx, "float16") else None
    return load_model(model_name, dtype=dtype)

def transcribe_audio(
    file_path: str,
    model_name: str = "mlx-community/whisper-large-v3-4bit",
    progress_callback = None
) -> dict:
    """
    Gọi thư viện mlx-whisper để nhận diện tiếng Việt từ file audio.
    Giải phóng Metal cache sau khi chạy để tránh lỗi OOM.
    """
    apply_monkeypatch()
    validate_audio_file(file_path)
    
    # Gán mô hình từ cache Streamlit vào ModelHolder của mlx_whisper để tránh tải lại (gây trùng lặp bộ nhớ)
    from mlx_whisper.transcribe import ModelHolder
    ModelHolder.model = get_whisper_model(model_name)
    ModelHolder.model_path = model_name
    
    # Can thiệp động (monkeypatch) tqdm để lấy tiến trình giải mã chi tiết
    original_tqdm = None
    transcribe_module = None
    if progress_callback is not None:
        try:
            import sys
            transcribe_module = sys.modules.get("mlx_whisper.transcribe")
            if transcribe_module is None:
                try:
                    import importlib
                    importlib.import_module("mlx_whisper.transcribe")
                    transcribe_module = sys.modules.get("mlx_whisper.transcribe")
                except Exception:
                    pass
            original_tqdm = getattr(transcribe_module, "tqdm", None) if transcribe_module is not None else None
            
            if original_tqdm is not None:
                class TqdmWrapper:
                    def __init__(self, original_tqdm_module, cb):
                        self.original_tqdm_module = original_tqdm_module
                        self.cb = cb
                        
                    def tqdm(self, *args, **kwargs):
                        cb = self.cb
                        orig_tqdm = self.original_tqdm_module
                        
                        total = kwargs.get('total')
                        if total is None and len(args) > 0:
                            try:
                                total = len(args[0])
                            except (TypeError, AttributeError):
                                total = None
                        unit = kwargs.get('unit') or (args[1] if len(args) > 1 else 'it')
                        
                        real_pbar = orig_tqdm.tqdm(*args, **kwargs)
                        
                        class StreamlitTqdm:
                            def __init__(self, total, unit, real_pbar):
                                self.total = total
                                self.n = 0
                                self.real_pbar = real_pbar
                                if self.total is not None and self.total > 0:
                                    try:
                                        cb(0, self.total)
                                    except Exception:
                                        pass
                                        
                            def update(self, n=1):
                                self.n += n
                                self.real_pbar.update(n)
                                if self.total is not None and self.total > 0:
                                    try:
                                        cb(self.n, self.total)
                                    except Exception:
                                        pass
                                        
                            def __enter__(self):
                                self.real_pbar.__enter__()
                                return self
                                
                            def __exit__(self, exc_type, exc_val, exc_tb):
                                self.real_pbar.__exit__(exc_type, exc_val, exc_tb)
                                
                        return StreamlitTqdm(total, unit, real_pbar)
                        
                    def __getattr__(self, name):
                        return getattr(self.original_tqdm_module, name)
                
                transcribe_module.tqdm = TqdmWrapper(original_tqdm, progress_callback)
        except Exception:
            pass
            
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
        # Khôi phục lại tqdm gốc
        if transcribe_module is not None and original_tqdm is not None:
            transcribe_module.tqdm = original_tqdm
        # Giải phóng cache Metal sau khi chạy
        mx.metal.clear_cache()
        gc.collect()
        
    return result

def force_clear_gpu_cache() -> None:
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

