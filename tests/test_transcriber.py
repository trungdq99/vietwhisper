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


from utils.transcriber import validate_audio_file, transcribe_audio, force_clear_gpu_cache, get_whisper_model

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
        with transcribe_module.tqdm.tqdm(total=100, unit="frames") as pbar:
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

