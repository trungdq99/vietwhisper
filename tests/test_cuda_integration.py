import os
import time
import tempfile
import numpy as np
from scipy.io import wavfile
import pytest
from utils.transcriber import transcribe_audio, force_clear_gpu_cache, is_cuda_available, detect_backend
from utils.queue_manager import QueueManager

@pytest.fixture(autouse=True)
def reset_singleton():
    QueueManager._instance = None
    yield

def test_force_clear_gpu_cache_runs_without_error():
    force_clear_gpu_cache()

@pytest.mark.skipif(not is_cuda_available(), reason="Requires NVIDIA CUDA GPU")
def test_cuda_real_audio_transcription(tmp_path):
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    temp_wav = tmp_path / "test_sine.wav"
    wavfile.write(str(temp_wav), sample_rate, (audio * 32767).astype(np.int16))

    callbacks = []
    def on_progress(cur, total):
        callbacks.append((cur, total))

    result = transcribe_audio(
        str(temp_wav),
        model_name="tiny",
        progress_callback=on_progress,
        backend="cuda"
    )

    assert "text" in result
    assert "segments" in result
    assert "duration" in result
    assert result["duration"] > 0
    assert len(callbacks) >= 1
    assert callbacks[-1][0] == callbacks[-1][1]

@pytest.mark.skipif(not is_cuda_available(), reason="Requires NVIDIA CUDA GPU")
def test_cuda_queue_manager_end_to_end(tmp_path):
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    temp_wav = tmp_path / "test_queue.wav"
    wavfile.write(str(temp_wav), sample_rate, (audio * 32767).astype(np.int16))

    qm = QueueManager()
    qm.add_task("test_queue.wav", str(temp_wav), model_name="tiny")
    assert len(qm.get_status()) == 1

    qm.start_worker("tiny")
    qm.task_queue.join()
    time.sleep(0.5)

    status = qm.get_status()
    assert len(status) == 1
    assert status[0]["status"] == "Hoàn thành"
    results = qm.get_results()
    assert "test_queue.wav" in results
    assert "error" not in results["test_queue.wav"]


@pytest.mark.skipif(not is_cuda_available(), reason="Requires NVIDIA CUDA GPU")
def test_cuda_batched_real_audio_transcription(tmp_path):
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    temp_wav = tmp_path / "test_batched_sine.wav"
    wavfile.write(str(temp_wav), sample_rate, (audio * 32767).astype(np.int16))

    callbacks = []
    def on_progress(cur, total):
        callbacks.append((cur, total))

    result = transcribe_audio(
        str(temp_wav),
        model_name="tiny",
        progress_callback=on_progress,
        backend="cuda",
        batch_size=8
    )

    assert "text" in result
    assert "segments" in result
    assert "duration" in result
    assert result["duration"] > 0
    assert len(callbacks) >= 1
    assert callbacks[-1][0] == callbacks[-1][1]


@pytest.mark.skipif(not is_cuda_available(), reason="Requires NVIDIA CUDA GPU")
def test_cuda_batched_real_audio_transcription_batch_16(tmp_path):
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    temp_wav = tmp_path / "test_batched_sine_16.wav"
    wavfile.write(str(temp_wav), sample_rate, (audio * 32767).astype(np.int16))

    result = transcribe_audio(
        str(temp_wav),
        model_name="tiny",
        backend="cuda",
        batch_size=16
    )

    assert "text" in result
    assert "segments" in result
    assert "duration" in result
    assert result["duration"] > 0


@pytest.mark.skipif(not is_cuda_available(), reason="Requires NVIDIA CUDA GPU")
def test_cuda_batched_queue_manager_end_to_end(tmp_path):
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    temp_wav = tmp_path / "test_batched_queue.wav"
    wavfile.write(str(temp_wav), sample_rate, (audio * 32767).astype(np.int16))

    qm = QueueManager()
    qm.add_task("test_batched_queue.wav", str(temp_wav), model_name="tiny", batch_size=8)
    assert len(qm.get_status()) == 1

    qm.start_worker("tiny")
    qm.task_queue.join()
    time.sleep(0.5)

    status = qm.get_status()
    assert len(status) == 1
    assert status[0]["status"] == "Hoàn thành"
    results = qm.get_results()
    assert "test_batched_queue.wav" in results
    assert "error" not in results["test_batched_queue.wav"]


@pytest.mark.skipif(not is_cuda_available(), reason="Requires NVIDIA CUDA GPU")
def test_cuda_batched_sentence_level_timestamps(tmp_path):
    # Tests that batched inference produces granular sentence-level timestamps
    sample_rate = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440Hz tone with a 1-second pause in the middle
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio[int(sample_rate * 1.5):int(sample_rate * 2.5)] = 0.0

    temp_wav = tmp_path / "test_pause.wav"
    wavfile.write(str(temp_wav), sample_rate, (audio * 32767).astype(np.int16))

    result = transcribe_audio(
        str(temp_wav),
        model_name="tiny",
        backend="cuda",
        batch_size=8
    )

    assert "segments" in result
    assert "duration" in result
    assert result["duration"] == pytest.approx(4.0, abs=0.1)

