import pytest
import os
import sys
from unittest.mock import MagicMock, patch

import types

if 'mlx' not in sys.modules:
    # Mock mlx and mlx_whisper modules for non-macOS/non-MLX testing environment compatibility
    mlx = types.ModuleType('mlx')
    mlx_core = types.ModuleType('mlx.core')
    mlx_core_metal = types.ModuleType('mlx.core.metal')
    mlx_whisper = types.ModuleType('mlx_whisper')
    mlx_whisper_transcribe = types.ModuleType('mlx_whisper.transcribe')
    mlx_whisper_load_models = types.ModuleType('mlx_whisper.load_models')

    # Wire relationships
    sys.modules['mlx'] = mlx
    sys.modules['mlx.core'] = mlx_core
    mlx.core = mlx_core
    sys.modules['mlx.core.metal'] = mlx_core_metal
    mlx_core.metal = mlx_core_metal

    sys.modules['mlx_whisper'] = mlx_whisper
    sys.modules['mlx_whisper.transcribe'] = mlx_whisper_transcribe
    mlx_whisper.transcribe = mlx_whisper_transcribe
    sys.modules['mlx_whisper.load_models'] = mlx_whisper_load_models
    mlx_whisper.load_models = mlx_whisper_load_models

    # Add default mocks / classes
    mlx_core_metal.clear_cache = MagicMock()
    mlx_whisper_transcribe.transcribe = MagicMock()
    mlx_whisper_load_models.load_model = MagicMock()

    class MockModelHolder:
        model = "some_model"
        model_path = "some_path"

    mlx_whisper_transcribe.ModelHolder = MockModelHolder


from utils.transcriber import (
    validate_audio_file,
    transcribe_audio,
    force_clear_gpu_cache,
    get_whisper_model,
    get_batched_pipeline,
    detect_backend,
    get_hardware_status,
    get_available_models,
    is_cuda_available,
    is_mlx_available
)

def test_validate_audio_file_non_existent():
    with pytest.raises(FileNotFoundError):
        validate_audio_file("non_existent_file.mp3")

def test_validate_audio_file_invalid_extension(tmp_path):
    temp_file = tmp_path / "test.txt"
    temp_file.write_text("dummy")
    with pytest.raises(ValueError, match="Định dạng file không được hỗ trợ"):
        validate_audio_file(str(temp_file))

def test_validate_audio_file_empty(tmp_path):
    temp_file = tmp_path / "test.mp3"
    temp_file.write_bytes(b"")  # 0 bytes
    with pytest.raises(ValueError, match="Tệp âm thanh bị trống"):
        validate_audio_file(str(temp_file))

def test_validate_audio_file_corrupt(tmp_path):
    temp_file = tmp_path / "test.mp3"
    temp_file.write_bytes(b"corrupt content")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        with pytest.raises(ValueError, match="Tệp tin âm thanh bị hỏng"):
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

@patch("utils.transcriber.get_whisper_model")
@patch("mlx_whisper.transcribe.ModelHolder")
@patch("mlx.core.metal.clear_cache")
def test_force_clear_gpu_cache(mock_clear_cache, mock_model_holder, mock_get_model):
    force_clear_gpu_cache()
    assert mock_model_holder.model is None
    assert mock_model_holder.model_path is None
    mock_get_model.clear.assert_called_once()
    mock_clear_cache.assert_called_once()

@patch("mlx_whisper.load_models.load_model")
def test_get_whisper_model(mock_load_model):
    mock_load_model.return_value = "mocked_model"
    result = get_whisper_model("some-model")
    assert result == "mocked_model"
    mock_load_model.assert_called_once_with("some-model")

def test_validate_audio_file_ffprobe_missing(tmp_path):
    temp_file = tmp_path / "test.mp3"
    temp_file.write_bytes(b"dummy audio content")
    with patch("subprocess.run", side_effect=FileNotFoundError):
        # Should not raise FileNotFoundError, it should pass gracefully
        validate_audio_file(str(temp_file))

@patch("mlx_whisper.transcribe")
@patch("mlx.core.metal.clear_cache")
@patch("utils.transcriber.validate_audio_file")
def test_transcribe_audio_with_progress_callback(mock_validate, mock_clear_cache, mock_transcribe, tmp_path):
    temp_file = tmp_path / "test.mp3"
    temp_file.write_text("dummy")
    
    import sys
    import tqdm
    transcribe_module = sys.modules.get('mlx_whisper.transcribe')
    transcribe_module.tqdm = tqdm
    
    callback_calls = []
    def dummy_callback(current, total):
        callback_calls.append((current, total))
        
    mock_transcribe.return_value = {"text": "Hello", "segments": []}
    
    def mock_transcribe_impl(*args, **kwargs):
        assert hasattr(transcribe_module, "tqdm")
        with transcribe_module.tqdm.tqdm(range(100), unit="frames") as pbar:
            pbar.update(20)
            pbar.update(30)
        return {"text": "Hello", "segments": []}
        
    mock_transcribe.side_effect = mock_transcribe_impl
    
    try:
        result = transcribe_audio(str(temp_file), "mlx-community/whisper-base-4bit", progress_callback=dummy_callback)
    finally:
        if hasattr(transcribe_module, "tqdm"):
            delattr(transcribe_module, "tqdm")
            
    assert result == {"text": "Hello", "segments": []}
    assert len(callback_calls) == 3
    assert callback_calls == [(0, 100), (20, 100), (50, 100)]


def test_detect_backend():
    with patch("utils.transcriber.is_cuda_available", return_value=True):
        assert detect_backend() == "cuda"
    with patch("utils.transcriber.is_cuda_available", return_value=False), \
         patch("utils.transcriber.is_mlx_available", return_value=True):
        assert detect_backend() == "mlx"
    with patch("utils.transcriber.is_cuda_available", return_value=False), \
         patch("utils.transcriber.is_mlx_available", return_value=False):
        assert detect_backend() == "cpu"


def test_get_hardware_status_cuda():
    with patch("utils.transcriber.detect_backend", return_value="cuda"):
        status = get_hardware_status()
        assert status["backend"] == "cuda"
        assert status["compute_type"] == "float16"
        assert "RTX" in status["device_name"] or "CUDA" in status["description"]
        assert status["vram_gb"] >= 6


def test_get_available_models():
    cuda_models = get_available_models(backend="cuda")
    assert "large-v3" in cuda_models
    assert "large-v3-turbo" in cuda_models
    assert "medium" in cuda_models
    assert "tiny" in cuda_models

    mlx_models = get_available_models(backend="mlx")
    assert any("mlx-community" in m for m in mlx_models)

    cpu_models = get_available_models(backend="cpu")
    assert "base" in cpu_models


@patch("faster_whisper.WhisperModel")
@patch("utils.transcriber.validate_audio_file")
def test_transcribe_audio_faster_whisper_cuda(mock_validate, mock_model_cls, tmp_path):
    get_whisper_model.clear()
    temp_file = tmp_path / "test.mp3"
    temp_file.write_text("dummy")

    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model

    mock_segment = MagicMock()
    mock_segment.id = 0
    mock_segment.start = 0.0
    mock_segment.end = 2.5
    mock_segment.text = " Xin chào Việt Nam"

    mock_info = MagicMock()
    mock_info.duration = 2.5
    mock_info.language = "vi"

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    result = transcribe_audio(
        str(temp_file),
        model_name="large-v3",
        backend="cuda"
    )

    assert result["text"] == "Xin chào Việt Nam"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == " Xin chào Việt Nam"
    assert result["language"] == "vi"
    assert result["duration"] == 2.5
    mock_validate.assert_called_once_with(str(temp_file))
    mock_model.transcribe.assert_called_once_with(
        str(temp_file),
        language="vi",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
        log_prob_threshold=-1.5,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=2.0
    )


@patch("faster_whisper.WhisperModel")
@patch("utils.transcriber.validate_audio_file")
def test_transcribe_audio_faster_whisper_progress_callback(mock_validate, mock_model_cls, tmp_path):
    get_whisper_model.clear()
    temp_file = tmp_path / "test.mp3"
    temp_file.write_text("dummy")

    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model

    seg1 = MagicMock(id=0, start=0.0, end=1.0, text=" Một")
    seg2 = MagicMock(id=1, start=1.0, end=2.0, text=" Hai")
    mock_info = MagicMock(duration=2.0, language="vi")
    mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

    callback_calls = []
    def progress_cb(current, total):
        callback_calls.append((current, total))

    result = transcribe_audio(
        str(temp_file),
        model_name="large-v3",
        backend="cuda",
        progress_callback=progress_cb
    )

    assert result["text"] == "Một Hai"
    assert (0, 2) in callback_calls
    assert (2, 2) in callback_calls


@patch("faster_whisper.WhisperModel")
@patch("utils.transcriber.validate_audio_file")
def test_transcribe_audio_faster_whisper_space_separated_segments(mock_validate, mock_model_cls, tmp_path):
    get_whisper_model.clear()
    temp_file = tmp_path / "test.mp3"
    temp_file.write_text("dummy")

    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model

    # Segments without leading spaces (should be joined with space)
    seg1 = MagicMock(id=0, start=0.0, end=1.0, text="Xin chào")
    seg2 = MagicMock(id=1, start=1.0, end=2.0, text="Việt Nam")
    mock_info = MagicMock(duration=2.0, language="vi")
    mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

    result = transcribe_audio(
        str(temp_file),
        model_name="large-v3",
        backend="cuda"
    )

    assert result["text"] == "Xin chào Việt Nam"


@patch("faster_whisper.BatchedInferencePipeline")
@patch("faster_whisper.WhisperModel")
def test_get_batched_pipeline(mock_model_cls, mock_pipeline_cls):
    get_whisper_model.clear()
    get_batched_pipeline.clear()
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    pipeline = get_batched_pipeline("large-v3", device="cuda", compute_type="float16", backend="cuda")
    assert pipeline == mock_pipeline
    mock_pipeline_cls.assert_called_once_with(model=mock_model)


@patch("utils.transcriber.get_batched_pipeline")
@patch("utils.transcriber.validate_audio_file")
def test_transcribe_audio_faster_whisper_batched(mock_validate, mock_get_pipeline, tmp_path):
    temp_file = tmp_path / "test_batched.mp3"
    temp_file.write_text("dummy")

    mock_pipeline = MagicMock()
    mock_get_pipeline.return_value = mock_pipeline

    seg1 = MagicMock(id=0, start=0.0, end=1.5, text=" Xin chào")
    seg2 = MagicMock(id=1, start=1.5, end=3.0, text=" thế giới")
    mock_info = MagicMock(duration=3.0, language="vi")
    mock_pipeline.transcribe.return_value = ([seg1, seg2], mock_info)

    result = transcribe_audio(
        str(temp_file),
        model_name="large-v3",
        backend="cuda",
        batch_size=8
    )

    assert result["text"] == "Xin chào thế giới"
    assert len(result["segments"]) == 2
    assert result["segments"][0]["text"] == " Xin chào"
    assert result["segments"][1]["text"] == " thế giới"
    assert result["duration"] == 3.0
    assert result["language"] == "vi"
    mock_validate.assert_called_once_with(str(temp_file))
    mock_pipeline.transcribe.assert_called_once_with(
        str(temp_file),
        batch_size=8,
        language="vi",
        beam_size=5,
        without_timestamps=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
        log_prob_threshold=-1.5,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=2.0
    )


@patch("utils.transcriber.get_batched_pipeline")
@patch("utils.transcriber.validate_audio_file")
def test_transcribe_audio_faster_whisper_batched_progress_callback(mock_validate, mock_get_pipeline, tmp_path):
    temp_file = tmp_path / "test_batched_prog.mp3"
    temp_file.write_text("dummy")

    mock_pipeline = MagicMock()
    mock_get_pipeline.return_value = mock_pipeline

    seg1 = MagicMock(id=0, start=0.0, end=1.5, text=" Đoạn 1")
    seg2 = MagicMock(id=1, start=1.5, end=3.0, text=" Đoạn 2")
    mock_info = MagicMock(duration=3.0, language="vi")
    mock_pipeline.transcribe.return_value = ([seg1, seg2], mock_info)

    callbacks = []
    def on_progress(cur, total):
        callbacks.append((cur, total))

    result = transcribe_audio(
        str(temp_file),
        model_name="large-v3",
        backend="cuda",
        batch_size=16,
        progress_callback=on_progress
    )

    assert result["text"] == "Đoạn 1 Đoạn 2"
    assert (0, 3) in callbacks
    assert (1, 3) in callbacks or (3, 3) in callbacks
    assert callbacks[-1] == (3, 3)


@patch("utils.transcriber.get_batched_pipeline")
@patch("utils.transcriber.get_whisper_model")
def test_force_clear_gpu_cache_clears_batched_pipeline(mock_get_model, mock_get_batched):
    force_clear_gpu_cache()
    mock_get_batched.clear.assert_called_once()



