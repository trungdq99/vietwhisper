import os
import queue
import threading
import time
import pytest
from unittest.mock import patch, MagicMock
from utils.queue_manager import QueueManager

@pytest.fixture(autouse=True)
def reset_singleton():
    QueueManager._instance = None
    yield

def test_singleton_behavior():
    qm1 = QueueManager()
    qm2 = QueueManager()
    assert qm1 is qm2

def test_add_task_updates_states():
    qm = QueueManager()
    assert qm.get_status() == []
    
    qm.add_task("test_file.mp3", "/tmp/test_file.mp3", "base")
    status = qm.get_status()
    assert len(status) == 1
    assert status[0]["name"] == "test_file.mp3"
    assert status[0]["status"] == "Chờ xử lý"
    assert status[0]["progress"] == 0.0
    assert status[0]["elapsed"] == 0.0
    assert status[0]["eta"] is None

@patch("utils.queue_manager.transcribe_audio")
@patch("os.path.exists")
@patch("os.remove")
def test_worker_loop_transitions(mock_remove, mock_exists, mock_transcribe):
    mock_exists.return_value = True
    
    in_progress = threading.Event()
    resume = threading.Event()
    
    def mock_transcribe_impl(path, model_name, progress_callback):
        # Trigger progress callback to test calculation
        progress_callback(50, 100)
        in_progress.set()
        resume.wait(timeout=2)
        return {"text": "Hello world", "segments": []}
        
    mock_transcribe.side_effect = mock_transcribe_impl
    
    qm = QueueManager()
    qm.add_task("test_file.mp3", "/tmp/test_file.mp3", "base")
    
    # Initially task is pending
    assert qm.get_status()[0]["status"] == "Chờ xử lý"
    
    # Start worker
    qm.start_worker("base")
    
    # Wait for mock_transcribe to be called
    assert in_progress.wait(timeout=2)
    
    # Status should be "Đang xử lý"
    status_during = qm.get_status()
    assert status_during[0]["status"] == "Đang xử lý"
    assert status_during[0]["progress"] == 0.5
    
    # Resume and finish
    resume.set()
    
    # Wait for the queue to be empty/processed
    qm.task_queue.join()
    
    # Status should now be "Hoàn thành"
    status_after = qm.get_status()
    assert status_after[0]["status"] == "Hoàn thành"
    assert qm.get_results()["test_file.mp3"] == {"text": "Hello world", "segments": []}

@patch("os.path.exists")
def test_worker_loop_file_not_found(mock_exists):
    mock_exists.return_value = False
    
    qm = QueueManager()
    qm.add_task("missing_file.mp3", "/tmp/missing_file.mp3", "base")
    
    qm.start_worker("base")
    qm.task_queue.join()
    
    status = qm.get_status()
    assert status[0]["status"] == "Lỗi"
    results = qm.get_results()
    assert "error" in results["missing_file.mp3"]

def test_getters_return_copies():
    qm = QueueManager()
    qm.add_task("test.mp3", "/tmp/test.mp3", "base")
    
    # test get_status returns a list copy
    status = qm.get_status()
    status.append({"name": "extra.mp3", "status": "Chờ xử lý"})
    # Internal tasks should not change
    assert len(qm.get_status()) == 1
    
    # test modifying dict in get_status doesn't mutate internal state
    status_2 = qm.get_status()
    status_2[0]["status"] = "Mutated"
    assert qm.get_status()[0]["status"] == "Chờ xử lý"
    
    # test get_results returns a copy
    qm.results["test.mp3"] = {"text": "Original", "segments": [{"text": "segment1"}]}
    res = qm.get_results()
    res["test.mp3"]["text"] = "Mutated"
    res["test.mp3"]["segments"][0]["text"] = "Mutated"
    assert qm.get_results()["test.mp3"]["text"] == "Original"
    assert qm.get_results()["test.mp3"]["segments"][0]["text"] == "segment1"

def test_duplicate_task_ignored():
    qm = QueueManager()
    
    # 1. Test ignoring when status is "Chờ xử lý"
    qm.add_task("test.mp3", "/tmp/test.mp3", "base")
    assert len(qm.get_status()) == 1
    # Adding it again shouldn't add or raise
    qm.add_task("test.mp3", "/tmp/test.mp3", "base")
    assert len(qm.get_status()) == 1
    
    # 2. Test ignoring when status is "Đang xử lý"
    qm.tasks["test.mp3"]["status"] = "Đang xử lý"
    qm.add_task("test.mp3", "/tmp/test.mp3", "base")
    assert qm.tasks["test.mp3"]["status"] == "Đang xử lý"
    
    # 3. Test ignoring when status is "Hoàn thành"
    qm.tasks["test.mp3"]["status"] = "Hoàn thành"
    qm.add_task("test.mp3", "/tmp/test.mp3", "base")
    assert qm.tasks["test.mp3"]["status"] == "Hoàn thành"
