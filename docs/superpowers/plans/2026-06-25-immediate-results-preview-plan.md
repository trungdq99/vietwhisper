# Xem và Tải kết quả Nhận diện Tức thời Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho phép người dùng xem kết quả nhận diện tiếng Việt và tải xuống các tệp tin kết quả ngay khi từng tệp âm thanh hoàn thành xử lý trong hàng đợi mà không cần chờ toàn bộ hàng đợi chạy xong, đồng thời đảm bảo hiệu năng và an toàn luồng tuyệt đối.

**Architecture:** 
1. **QueueManager Singleton (`utils/queue_manager.py`):** Lớp quản lý hàng đợi xử lý tuần tự (FIFO) sử dụng `queue.Queue` chuẩn của Python và một background worker thread (daemon), đảm bảo tại một thời điểm chỉ có tối đa 1 file thực hiện nhận dạng trên GPU Metal, tránh quá tải VRAM.
2. **Thread-safe State Coordinator:** Toàn bộ trạng thái hàng đợi, tiến độ và kết quả được bảo vệ bằng `threading.Lock` trong Manager. Khi giao diện đọc trạng thái, Coordinator trả về bản sao (copy) để tránh lỗi sập giao diện `RuntimeError: dictionary changed size during iteration`.
3. **Early Disk Persistence:** Khi người dùng tải file lên, luồng chính Streamlit lập tức ghi dữ liệu bytes đồng bộ xuống đĩa cứng thành file tạm và chỉ lưu đường dẫn `temp_path` vào hàng đợi. Loại bỏ hoàn toàn lỗi đóng file pointer khi rerun (`ValueError: I/O operation on closed file`).
4. **Lazy Export Caching:** Sinh định dạng xuất (.docx, .md) khi người dùng chọn xem file đó và lưu vào cache trong `st.session_state` để tránh nghẽn CPU (GIL lock) khi fragment tự động quét mỗi giây.
5. **Thanh tiến độ kép & UI đồng dạng:** Hiển thị tiến độ tổng thể của hàng đợi và tiến độ chi tiết của file đang chạy, giữ nguyên cấu trúc visual khi kết thúc để tránh giật giao diện (Layout Shifts).

**Tech Stack:** Python 3.10, Streamlit 1.58.0, mlx-whisper.

## Global Constraints
- Không chặn luồng chính của Streamlit khi xử lý nhận dạng âm thanh.
- Tự động hiển thị tiến độ và kết quả tức thời khi từng tệp tin chạy xong qua @st.fragment tĩnh ở module-level.
- Không tự động thay đổi dropdown kết quả khi người dùng đang đọc tệp khác; thay vào đó gửi thông báo qua `st.toast()`.
- Sử dụng cache xuất tệp và lazy generation để tối ưu hóa CPU.
- Đảm bảo tất cả 21 unit test hiện có tiếp tục vượt qua thành công và bổ sung các kịch bản kiểm thử biên quan trọng.

---

### Task 1: Khởi tạo biến trạng thái Session State & Module mới

**Files:**
- Modify: [app.py](file:///Users/trungshin/Downloads/mmm/app.py#L125-L132)
- Create [NEW]: [utils/queue_manager.py](file:///Users/trungshin/Downloads/mmm/utils/queue_manager.py)

**Interfaces:**
- Consumes: None
- Produces: Khởi tạo `st.session_state.results`, `st.session_state.last_completed_count`, `st.session_state.export_cache`, `st.session_state.zip_cache` trong `app.py`.

- [ ] **Step 1: Khởi tạo các biến Session State mới trong app.py**
  Sửa đổi trong [app.py](file:///Users/trungshin/Downloads/mmm/app.py):
  ```python
  # 3. Session State Initialization
  if "queue" not in st.session_state:
      st.session_state.queue = []
  if "results" not in st.session_state:
      st.session_state.results = {}
  if "processing" not in st.session_state:
      st.session_state.processing = False
  if "last_completed_count" not in st.session_state:
      st.session_state.last_completed_count = 0
  if "export_cache" not in st.session_state:
      st.session_state.export_cache = {}
  if "zip_cache" not in st.session_state:
      st.session_state.zip_cache = {}
  ```

- [ ] **Step 2: Tạo cấu trúc khung của module QueueManager**
  Tạo file mới [utils/queue_manager.py](file:///Users/trungshin/Downloads/mmm/utils/queue_manager.py) chứa khai báo các thư viện cần thiết.

- [ ] **Step 3: Chạy thử pytest để đảm bảo cấu trúc trang không bị vỡ**
  Run: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest tests/test_app.py::test_app_renders`
  Expected: PASS

- [ ] **Step 4: Commit các thay đổi ban đầu**
  ```bash
  git add app.py
  git commit -m "feat: initialize session state variables and create queue manager module structure"
  ```

---

### Task 2: Triển khai Lớp QueueManager Singleton & Daemon Thread

**Files:**
- Modify: [utils/queue_manager.py](file:///Users/trungshin/Downloads/mmm/utils/queue_manager.py)

**Interfaces:**
- Consumes: `utils.transcriber.transcribe_audio`
- Produces: Lớp `QueueManager` Singleton thread-safe để quản lý và xử lý hàng đợi chạy nền.

- [ ] **Step 1: Định nghĩa lớp QueueManager và cơ chế Singleton**
  Viết logic `__new__` và `_init_manager` trong [utils/queue_manager.py](file:///Users/trungshin/Downloads/mmm/utils/queue_manager.py):
  ```python
  import threading
  import queue
  import time
  import os
  import logging
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
  ```

- [ ] **Step 2: Triển khai các phương thức add_task, start_worker và _worker_loop**
  Bổ sung logic nạp dữ liệu file từ đĩa, xử lý tuần tự qua callback và giải phóng file tạm:
  ```python
      def add_task(self, file_name: str, temp_path: str, model_name: str):
          with self.lock:
              if file_name in self.tasks and self.tasks[file_name]["status"] == "Hoàn thành":
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
  ```

- [ ] **Step 3: Triển khai các getter method trả về bản sao an toàn luồng**
  ```python
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
              return self.results.copy()
              
      def is_processing(self) -> bool:
          with self.lock:
              return self.current_task_id is not None
  ```

- [ ] **Step 4: Commit module QueueManager**
  ```bash
  git add utils/queue_manager.py
  git commit -m "feat: implement QueueManager Singleton class and worker thread loop"
  ```

---

### Task 3: Tích hợp Streamlit Fragment tĩnh và Ghi file tạm đồng bộ

**Files:**
- Modify: [app.py](file:///Users/trungshin/Downloads/mmm/app.py#L191-L211)

**Interfaces:**
- Consumes: `utils/queue_manager.py`
- Produces: Hàm `@st.fragment` tĩnh ở module-level cập nhật UI tiến độ và ghi file tạm đồng bộ trên luồng chính.

- [ ] **Step 1: Khai báo Fragment render_progress_panel tĩnh ở Module-level**
  Thêm đoạn code sau vào phần trên của [app.py](file:///Users/trungshin/Downloads/mmm/app.py) (không lồng vào khối điều kiện `if`):
  ```python
  from utils.queue_manager import QueueManager
  
  qm = QueueManager()

  @st.fragment(run_every="1s")
  def render_progress_panel():
      statuses = qm.get_status()
      results = qm.get_results()
      
      st.markdown("#### Hàng đợi xử lý:")
      for item in statuses:
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
          
      st.markdown("---")
      
      if qm.is_processing():
          total = len(statuses)
          completed = sum(1 for x in statuses if x["status"] == "Hoàn thành")
          
          active_task = next((x for x in statuses if x["status"] == "Đang xử lý"), None)
          if active_task:
              # 1. Thanh tiến độ tổng thể
              st.markdown(f"**Tiến độ tổng thể ({completed}/{total} tệp):**")
              file_prog = active_task["progress"]
              overall_prog = (completed + file_prog) / total if total > 0 else 0.0
              st.progress(min(1.0, overall_prog))
              
              # 2. Tiến độ tệp hiện tại
              st.markdown(f"🎙️ **Tệp hiện tại:** `{active_task['name']}`")
              st.progress(min(1.0, file_prog))
              
              eta = active_task["eta"]
              elapsed = active_task["elapsed"]
              eta_str = f"{int(eta)} giây" if (eta is not None and eta >= 1) else "đang tính..."
              elapsed_str = f"{int(elapsed)} giây"
              
              st.markdown(f"⏳ Đã chạy: **{elapsed_str}** | 🏁 Còn lại: **{eta_str}**")
      else:
          any_completed = any(x["status"] == "Hoàn thành" for x in statuses)
          any_error = any(x["status"] == "Lỗi" for x in statuses)
          if any_completed:
              st.markdown("**Tiến độ tổng thể (Hoàn thành):**")
              st.progress(1.0)
              if any_error:
                  st.warning("⚠️ **Đã xử lý xong hàng đợi nhưng phát hiện một số tệp bị lỗi.**")
              else:
                  st.success("🎉 **Đã hoàn thành nhận dạng tất cả các tệp âm thanh thành công.**")
      
      # Đồng bộ kết quả an toàn luồng và rerun giao diện khi có file mới dịch xong
      last_completed = st.session_state.get("last_completed_count", 0)
      current_completed = len(results)
      if current_completed > last_completed:
          st.session_state.results = results
          st.session_state.last_completed_count = current_completed
          st.rerun()
          
      if not qm.is_processing() and st.session_state.processing:
          st.session_state.processing = False
          st.rerun()
  ```

- [ ] **Step 2: Triển khai Early Disk Persistence trên Luồng chính Streamlit**
  Cập nhật logic khi phát hiện có tệp tin tải lên: ghi đồng bộ ra file tạm và đẩy đường dẫn vào `QueueManager`:
  ```python
  if uploaded_files:
      # Sát nhập danh sách file upload hiện tại vào QueueManager
      for uf in uploaded_files:
          # Kiểm tra xem file đã được nạp vào task list của QueueManager chưa
          statuses = qm.get_status()
          exists = any(x["name"] == uf.name for x in statuses)
          
          if not exists:
              # Ghi ra file tạm đồng bộ ngay lập tức
              suffix = os.path.splitext(uf.name)[1]
              with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                  CHUNK_SIZE = 1024 * 1024
                  uf.seek(0)
                  while chunk := uf.read(CHUNK_SIZE):
                      tmp_file.write(chunk)
                  temp_path = tmp_file.name
              
              qm.add_task(uf.name, temp_path, model_name)
      
      # Vẽ progress panel
      render_progress_panel()
      
      # Nút Bắt đầu nhận diện
      start_btn = st.button("🚀 Bắt đầu nhận diện", disabled=qm.is_processing() or st.session_state.processing)
      if start_btn:
          st.session_state.processing = True
          st.session_state.last_completed_count = len(st.session_state.results)
          qm.start_worker()
          st.rerun()
  ```

- [ ] **Step 3: Xóa bỏ các định nghĩa render_queue() và logic chạy Thread cũ**
  Lược bỏ hoàn toàn logic cũ không tương thích để tránh xung đột mã nguồn.

- [ ] **Step 4: Commit thay đổi UI và logic nạp file**
  ```bash
  git add app.py
  git commit -m "feat: integrate module-level st.fragment UI and early disk persistence for file uploading"
  ```

---

### Task 4: Cập nhật Cột Phải & Tải xuống với Lazy Export Caching

**Files:**
- Modify: [app.py](file:///Users/trungshin/Downloads/mmm/app.py#L300-L315)

**Interfaces:**
- Consumes: `st.session_state.results`
- Produces: Render kết quả cột phải, lazy generation DOCX/MD/ZIP và lưu cache.

- [ ] **Step 1: Cập nhật Dropdown hiển thị kết quả và st.toast**
  Sửa đổi cột phải hiển thị kết quả để giữ nguyên lựa chọn của người dùng và thông báo qua toast:
  ```python
  with col_right:
      st.markdown("### 📝 Kết quả & Tải về")
      
      if st.session_state.results:
          options = list(st.session_state.results.keys())
          
          # Tự động chọn file mới nhất nếu chưa có file nào được chọn
          if "selected_file_name" not in st.session_state or st.session_state.selected_file_name not in options:
              st.session_state.selected_file_name = options[-1] if options else None
          else:
              # Hiển thị toast thông báo file mới dịch xong
              if "prev_completed_files" not in st.session_state:
                  st.session_state.prev_completed_files = []
              
              new_files = [f for f in options if f not in st.session_state.prev_completed_files]
              if new_files:
                  for new_f in new_files:
                      st.toast(f"🎉 Đã dịch xong: **{new_f}**", icon="✅")
              
              st.session_state.prev_completed_files = options.copy()
              
          selected_file_name = st.selectbox(
              "Chọn file để xem kết quả:",
              options=options,
              key="selected_file_name"
          )
  ```

- [ ] **Step 2: Triển khai Caching & Lazy Export Generation**
  Thay thế logic sinh DOCX/MD/ZIP hàng loạt mỗi rerun bằng lazy caching:
  ```python
          if selected_file_name:
              res = st.session_state.results[selected_file_name]
              # (Vẽ preview nội dung text như cũ...)
              
              single_file_name = os.path.splitext(selected_file_name)[0] + export_format
              cache_key = (selected_file_name, export_format, include_timestamps)
              
              # Lazy generate và cache file đơn lẻ
              if cache_key not in st.session_state.export_cache:
                  segs = res.get("segments", [])
                  if export_format == ".docx":
                      st.session_state.export_cache[cache_key] = export_to_docx(segs, include_timestamps)
                  else:
                      st.session_state.export_cache[cache_key] = export_to_markdown(segs, include_timestamps)
                      
              single_file_data = st.session_state.export_cache[cache_key]
              
              st.download_button(
                  label=f"📥 Tải xuống tệp {single_file_name}",
                  data=single_file_data,
                  file_name=single_file_name,
                  mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document" if export_format == ".docx" else "text/markdown",
                  key="download_single_btn"
              )
              
              # Lazy generate và cache file nén ZIP tổng hợp khi có >=2 file thành công
              if len(st.session_state.results) > 1:
                  completed_keys = tuple(sorted(st.session_state.results.keys()))
                  zip_cache_key = (completed_keys, export_format, include_timestamps)
                  
                  if zip_cache_key not in st.session_state.zip_cache:
                      zip_export_files = {}
                      for name, r in st.session_state.results.items():
                          file_cache_key = (name, export_format, include_timestamps)
                          if file_cache_key not in st.session_state.export_cache:
                              segs = r.get("segments", [])
                              if export_format == ".docx":
                                  st.session_state.export_cache[file_cache_key] = export_to_docx(segs, include_timestamps)
                              else:
                                  st.session_state.export_cache[file_cache_key] = export_to_markdown(segs, include_timestamps)
                          zip_export_files[os.path.splitext(name)[0] + export_format] = st.session_state.export_cache[file_cache_key]
                      
                      st.session_state.zip_cache[zip_cache_key] = export_to_zip(zip_export_files)
                      
                  zip_bio = st.session_state.zip_cache[zip_cache_key]
                  st.download_button(
                      label="🗜️ Tải xuống toàn bộ tệp (.zip)",
                      data=zip_bio,
                      file_name="vietwhisper_transcripts.zip",
                      mime="application/zip",
                      key="download_zip_btn"
                  )
  ```

- [ ] **Step 3: Commit thay đổi**
  ```bash
  git add app.py
  git commit -m "feat: implement lazy export caching and zip compression optimization"
  ```

---

### Task 5: Nâng cấp Unit Tests cho Khả năng Kiểm thử Bất đồng bộ & Biên

**Files:**
- Modify: [tests/test_app.py](file:///Users/trungshin/Downloads/mmm/tests/test_app.py)

**Interfaces:**
- Consumes: `app.py`, `utils/queue_manager.py`
- Produces: Cập nhật và bổ sung 3 test cases hoàn toàn tương thích và đáng tin cậy.

- [ ] **Step 1: Sửa đổi test_app_transcription_success và test_app_transcription_failure**
  Patch `QueueManager.start_worker` để chạy `_worker_loop` một cách đồng bộ trong quá trình test. Khôi phục assert `st.error` cho trường hợp lỗi.

- [ ] **Step 2: Triển khai ca kiểm thử Callback tiến độ & Đăng ký Context**
  Thêm hàm `test_app_transcription_callback_and_context` sử dụng patch `QueueManager` và verify dữ liệu callback.

- [ ] **Step 3: Triển khai các ca kiểm thử biên (Edge Cases)**
  Thêm các hàm `test_app_partial_failures_and_download_zip` (Lỗi nửa chừng, xác minh không có nút ZIP nếu chỉ có 1 file thành công) và `test_app_auto_select_and_zip_button` (Dropdown auto-select và sự hiển thị của nút ZIP thông qua `UnknownElement` và `.proto.label`).

- [ ] **Step 4: Chạy toàn bộ các ca kiểm thử để xác minh độ chính xác**
  Run: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest`
  Expected: All tests pass (>= 23 passed)

- [ ] **Step 5: Commit thay đổi cuối cùng**
  ```bash
  git add tests/test_app.py
  git commit -m "test: upgrade and add edge case unit tests compatible with QueueManager"
  ```

---

## Verification Plan

### Automated Tests
- Chạy bộ công cụ kiểm thử: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest`

### Manual Verification
1. Tải lên cùng lúc 3 tệp âm thanh ngắn.
2. Nhấn nút "Bắt đầu nhận diện".
3. Xác minh trong hàng đợi: 
   - Tệp đang chạy hiển thị badge vàng và cập nhật thanh tiến độ chi tiết.
   - Các tệp kế tiếp hiển thị badge xám "Chờ xử lý".
4. Khi tệp 1 chạy xong:
   - dropdown cột phải tự chọn tệp 1, các nút DOCX/MD của tệp 1 xuất hiện để tải về tức thì.
   - Nút nén ZIP KHÔNG xuất hiện.
5. Khi tệp 2 chạy xong:
   - Dropdown không tự nhảy nếu người dùng đang xem kết quả tệp 1, hiển thị Toast góc màn hình thông báo tệp 2 đã hoàn thành.
   - Nút nén ZIP xuất hiện ở cột phải.
6. F5 reload lại trang và kiểm tra xem không có thread mồ côi chạy Whisper dưới nền và không có lỗi crash UI.
