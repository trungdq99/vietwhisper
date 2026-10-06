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


import pytest
from streamlit.testing.v1 import AppTest

@pytest.fixture(autouse=True)
def reset_singleton():
    from utils.queue_manager import QueueManager
    QueueManager._instance = None
    yield

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
    from utils.queue_manager import QueueManager
    
    # Mock return value of transcribe_audio
    mock_transcribe.return_value = {
        "text": "Xin chào thế giới.",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": " Xin chào thế giới."}
        ]
    }
    
    def mock_start_worker(model_name, batch_size=None, *args, **kwargs):
        qm = QueueManager()
        for name, task in list(qm.tasks.items()):
            if task["status"] == "Chờ xử lý":
                qm.tasks[name]["status"] = "Đang xử lý"
                try:
                    result = mock_transcribe(task["temp_path"], model_name=model_name)
                    qm.results[name] = result
                    qm.tasks[name]["status"] = "Hoàn thành"
                except Exception as e:
                    qm.tasks[name]["status"] = "Lỗi"
                    qm.results[name] = {"error": str(e)}
        qm.current_task_id = None
        
    with patch("utils.queue_manager.QueueManager.start_worker", side_effect=mock_start_worker):
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
    from utils.queue_manager import QueueManager
    
    # Mock transcribe_audio to raise an error
    mock_transcribe.side_effect = ValueError("Transcribe failed")
    
    def mock_start_worker(model_name, batch_size=None, *args, **kwargs):
        qm = QueueManager()
        for name, task in list(qm.tasks.items()):
            if task["status"] == "Chờ xử lý":
                qm.tasks[name]["status"] = "Đang xử lý"
                try:
                    mock_transcribe(task["temp_path"], model_name=model_name)
                    qm.tasks[name]["status"] = "Hoàn thành"
                except Exception as e:
                    qm.tasks[name]["status"] = "Lỗi"
                    qm.results[name] = {"error": str(e)}
        qm.current_task_id = None
        
    with patch("utils.queue_manager.QueueManager.start_worker", side_effect=mock_start_worker):
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

@patch("utils.queue_manager.transcribe_audio")
def test_app_transcription_callback_and_context(mock_transcribe, tmp_path):
    import os
    from utils.queue_manager import QueueManager
    
    # Create the temp file so os.path.exists returns True
    temp_file = tmp_path / "test.mp3"
    temp_file.write_text("dummy")
    
    qm = QueueManager()
    qm.add_task("test.mp3", str(temp_file), "base")
    
    def mock_transcribe_impl(path, model_name, progress_callback):
        progress_callback(50, 100)
        return {"text": "Hello", "segments": []}
        
    mock_transcribe.side_effect = mock_transcribe_impl
    
    qm.start_worker("base")
    qm.task_queue.join()
    
    status = qm.get_status()
    assert len(status) == 1
    assert status[0]["name"] == "test.mp3"
    assert status[0]["status"] == "Hoàn thành"
    assert qm.get_results()["test.mp3"] == {"text": "Hello", "segments": []}

@patch("utils.transcriber.transcribe_audio")
def test_app_partial_failures_and_download_zip(mock_transcribe):
    import os
    from utils.queue_manager import QueueManager
    
    # Mock first file success, second file failure
    def mock_transcribe_impl(path, model_name, **kwargs):
        if "test1.mp3" in path or "test1" in path:
            return {"text": "Success file 1", "segments": []}
        raise ValueError("Failed file 2")
        
    mock_transcribe.side_effect = mock_transcribe_impl
    
    def mock_start_worker(model_name, batch_size=None, *args, **kwargs):
        qm = QueueManager()
        for name, task in list(qm.tasks.items()):
            if task["status"] == "Chờ xử lý":
                qm.tasks[name]["status"] = "Đang xử lý"
                try:
                    result = mock_transcribe(task["temp_path"], model_name=model_name)
                    qm.results[name] = result
                    qm.tasks[name]["status"] = "Hoàn thành"
                except Exception as e:
                    qm.tasks[name]["status"] = "Lỗi"
                    qm.results[name] = {"error": str(e)}
        qm.current_task_id = None
        
    with patch("utils.queue_manager.QueueManager.start_worker", side_effect=mock_start_worker):
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        
        uploader = at.file_uploader[0]
        # Upload two files
        uploader.upload("test1.mp3", b"dummy 1")
        uploader.upload("test2.mp3", b"dummy 2")
        at.run()
        
        start_btn = [b for b in at.button if "Bắt đầu" in b.label][0]
        start_btn.click().run()
        
        # There should be only 1 successful file, so no ZIP button should be displayed
        download_buttons = at.get("download_button")
        zip_buttons = [b for b in download_buttons if "download_zip" in getattr(b.proto, "id", "")]
        assert len(zip_buttons) == 0

@patch("utils.transcriber.transcribe_audio")
def test_app_auto_select_and_zip_button(mock_transcribe):
    import os
    from utils.queue_manager import QueueManager
    
    mock_transcribe.return_value = {"text": "Success", "segments": []}
    
    def mock_start_worker(model_name, batch_size=None, *args, **kwargs):
        qm = QueueManager()
        for name, task in list(qm.tasks.items()):
            if task["status"] == "Chờ xử lý":
                qm.tasks[name]["status"] = "Đang xử lý"
                try:
                    result = mock_transcribe(task["temp_path"], model_name=model_name)
                    qm.results[name] = result
                    qm.tasks[name]["status"] = "Hoàn thành"
                except Exception as e:
                    qm.tasks[name]["status"] = "Lỗi"
                    qm.results[name] = {"error": str(e)}
        qm.current_task_id = None
        
    with patch("utils.queue_manager.QueueManager.start_worker", side_effect=mock_start_worker):
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        
        uploader = at.file_uploader[0]
        uploader.upload("test1.mp3", b"dummy 1")
        uploader.upload("test2.mp3", b"dummy 2")
        at.run()
        
        start_btn = [b for b in at.button if "Bắt đầu" in b.label][0]
        start_btn.click().run()
        
        # Verify selectbox value is set to the last completed file
        selectbox = next(sb for sb in at.selectbox if sb.key == "selected_file_name")
        assert selectbox.value == "test2.mp3"
        
        # ZIP button should be present since 2 files succeeded
        download_buttons = at.get("download_button")
        zip_buttons = [b for b in download_buttons if "download_zip" in getattr(b.proto, "id", "")]
        assert len(zip_buttons) == 1

def test_app_uploader_deletion_sync():
    import os
    from utils.queue_manager import QueueManager
    
    qm = QueueManager()
    
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    
    # 1. Upload two files
    uploader = at.file_uploader[0]
    uploader.upload("test1.mp3", b"dummy 1")
    uploader.upload("test2.mp3", b"dummy 2")
    at.run()
    
    print("Uploader type:", type(uploader))
    print("Uploader attributes:", dir(uploader))
    print("Uploader value:", uploader.value)
    
    # Verify both are registered as pending in qm
    assert len(qm.get_status()) == 2
    assert any(x["name"] == "test1.mp3" for x in qm.get_status())
    assert any(x["name"] == "test2.mp3" for x in qm.get_status())
    
    # 2. Simulate deletion: remove test2.mp3 from the uploader session state list
    at.session_state[uploader.id] = [uploader.value[0]]
    at.run()
    
    # Verify test2.mp3 is removed from qm pending tasks, but test1.mp3 remains
    assert len(qm.get_status()) == 1
    assert qm.get_status()[0]["name"] == "test1.mp3"


@patch("utils.transcriber.detect_backend", return_value="cuda")
@patch("utils.transcriber.is_cuda_available", return_value=True)
def test_app_batch_size_propagation_to_worker(mock_cuda_avail, mock_backend):
    import os
    from utils.queue_manager import QueueManager

    start_worker_calls = []
    def spy_start_worker(model_name, batch_size=None):
        start_worker_calls.append((model_name, batch_size))

    with patch.object(QueueManager, "start_worker", side_effect=spy_start_worker):
        app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app.py"))
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        # Check that batch size selectbox is present for CUDA
        batch_boxes = [sb for sb in at.selectbox if "Batch" in sb.label]
        assert len(batch_boxes) == 1
        assert batch_boxes[0].value == 8

        # Upload a file
        uploader = at.file_uploader[0]
        uploader.upload("cuda_test.mp3", b"dummy cuda data")
        at.run()

        # Click start
        start_btn = [b for b in at.button if "Bắt đầu" in b.label][0]
        start_btn.click().run()

        assert len(start_worker_calls) == 1
        assert start_worker_calls[0][1] == 8


