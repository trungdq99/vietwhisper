import sys
from unittest.mock import MagicMock, patch

import types

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

import pytest
from streamlit.testing.v1 import AppTest

def test_app_renders():
    import os
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.markdown) > 0
    assert any("VietWhisper" in m.value for m in at.markdown)


def test_session_state_initialization():
    import os
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception
    assert at.session_state.queue == []
    assert at.session_state.results == {}
    assert at.session_state.processing is False
    assert at.session_state.last_completed_count == 0
    assert at.session_state.export_cache == {}
    assert at.session_state.zip_cache == {}


@patch("utils.transcriber.force_clear_gpu_cache")
def test_clear_gpu_cache(mock_clear_gpu):
    import os
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    
    # Find and click the clear GPU button
    clear_btn = at.button(key="clear_gpu_btn")
    clear_btn.click().run()
    
    assert mock_clear_gpu.called
    assert len(at.success) > 0
    assert "Đã giải phóng bộ nhớ cache" in at.success[0].value

@patch("utils.transcriber.transcribe_audio")
def test_app_transcription_success(mock_transcribe):
    import os
    
    # Mock return value of transcribe_audio
    mock_transcribe.return_value = {
        "text": "Xin chào thế giới.",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Xin chào thế giới."}
        ]
    }
    
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    
    # Upload file
    uploader = at.file_uploader[0]
    uploader.upload("test.mp3", b"dummy mp3 data")
    at.run()
    
    # Find start button and click
    start_btn = [b for b in at.button if "Bắt đầu" in b.label][0]
    start_btn.click().run()
    
    # Verify transcribe_audio was called
    assert mock_transcribe.called
    
    # Verify preview is shown
    markdown_texts = [m.value for m in at.markdown]
    assert any("Xin chào thế giới." in text for text in markdown_texts)

@patch("utils.transcriber.transcribe_audio")
def test_app_transcription_failure(mock_transcribe):
    import os
    
    # Mock transcribe_audio to raise an error
    mock_transcribe.side_effect = ValueError("Transcribe failed")
    
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    
    # Upload file
    uploader = at.file_uploader[0]
    uploader.upload("test.mp3", b"dummy mp3 data")
    at.run()
    
    # Find start button and click
    start_btn = [b for b in at.button if "Bắt đầu" in b.label][0]
    start_btn.click().run()
    
    # Verify transcribe_audio was called
    assert mock_transcribe.called
    
    # Verify error is shown in UI
    assert len(at.error) > 0
    assert "Lỗi khi xử lý file test.mp3: Transcribe failed" in at.error[0].value
