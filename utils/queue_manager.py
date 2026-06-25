import threading
import queue
import time
import os
import logging
import copy
from typing import Dict, Any, List, Optional
from utils.transcriber import transcribe_audio

logger = logging.getLogger(__name__)

class QueueManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(QueueManager, cls).__new__(cls, *args, **kwargs)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self):
        self.task_queue = queue.Queue()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.worker_thread: Optional[threading.Thread] = None
        self.current_task_id: Optional[str] = None
        self.progress_states: Dict[str, Dict[str, Any]] = {}

    def add_task(self, file_name: str, temp_path: str, model_name: str):
        with self.lock:
            if file_name in self.tasks and self.tasks[file_name]["status"] in ["Hoàn thành", "Chờ xử lý", "Đang xử lý"]:
                return
            
            self.tasks[file_name] = {
                "name": file_name,
                "status": "Chờ xử lý",
                "temp_path": temp_path,
                "added_at": time.time()
            }
            self.progress_states[file_name] = {
                "progress": 0.0,
                "elapsed": 0.0,
                "eta": None
            }
            
            self.task_queue.put({
                "name": file_name,
                "temp_path": temp_path,
                "model_name": model_name
            })

    def start_worker(self):
        with self.lock:
            if self.worker_thread and self.worker_thread.is_alive():
                return
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()

    def _worker_loop(self):
        while True:
            try:
                task = self.task_queue.get(timeout=3)
                file_name = task["name"]
                temp_path = task["temp_path"]
                model_name = task["model_name"]
                
                with self.lock:
                    self.current_task_id = file_name
                    self.tasks[file_name]["status"] = "Đang xử lý"
                
                start_time = time.time()
                processing_start_time = [None]
                
                def progress_callback(current, total):
                    if total <= 0:
                        return
                    if processing_start_time[0] is None:
                        processing_start_time[0] = time.time()
                        
                    progress = current / total
                    elapsed = time.time() - start_time
                    elapsed_processing = time.time() - processing_start_time[0]
                    
                    if progress >= 0.02 and current > 0:
                        remaining = max(0, total - current)
                        fps = current / elapsed_processing if elapsed_processing > 0 else 1.0
                        eta = remaining / fps
                    else:
                        eta = None
                        
                    with self.lock:
                        self.progress_states[file_name] = {
                            "progress": progress,
                            "elapsed": elapsed,
                            "eta": eta
                        }

                try:
                    if os.path.exists(temp_path):
                        result = transcribe_audio(temp_path, model_name=model_name, progress_callback=progress_callback)
                        with self.lock:
                            self.results[file_name] = result
                            self.tasks[file_name]["status"] = "Hoàn thành"
                    else:
                        raise FileNotFoundError(f"Không tìm thấy file tạm: {temp_path}")
                except Exception as e:
                    logger.error(f"Lỗi xử lý file {file_name}: {str(e)}", exc_info=True)
                    with self.lock:
                        self.tasks[file_name]["status"] = "Lỗi"
                        self.results[file_name] = {"error": str(e)}
                finally:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                    self.task_queue.task_done()
                    
            except queue.Empty:
                with self.lock:
                    if self.task_queue.empty():
                        self.current_task_id = None
                        break

    def get_status(self) -> List[Dict[str, Any]]:
        with self.lock:
            status_list = []
            for name, task in self.tasks.items():
                prog = self.progress_states.get(name, {})
                status_list.append({
                    "name": name,
                    "status": task["status"],
                    "progress": prog.get("progress", 0.0),
                    "elapsed": prog.get("elapsed", 0.0),
                    "eta": prog.get("eta", None)
                })
            return status_list

    def get_results(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.results)
            
    def is_processing(self) -> bool:
        with self.lock:
            return self.current_task_id is not None
