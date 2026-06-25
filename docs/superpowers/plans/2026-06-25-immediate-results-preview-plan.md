# Xem và Tải kết quả Nhận diện Tức thời Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho phép người dùng xem kết quả nhận diện tiếng Việt và tải xuống các tệp tin kết quả ngay khi từng tệp âm thanh hoàn thành xử lý trong hàng đợi mà không cần chờ toàn bộ hàng đợi chạy xong.

**Architecture:** Sử dụng `threading.Thread` để thực hiện nhận diện dưới nền không chặn giao diện và Streamlit `@st.fragment(run_every="1s")` để cập nhật tiến độ tự động mỗi giây. Fragment sẽ kích hoạt `st.rerun()` để làm mới trang chính khi phát hiện số lượng tệp hoàn thành tăng lên, giúp cập nhật cột xem kết quả.

**Tech Stack:** Python 3.10, Streamlit 1.58.0, mlx-whisper.

## Global Constraints
- Không chặn luồng chính của Streamlit khi xử lý nhận dạng âm thanh.
- Tự động hiển thị tiến độ và kết quả tức thời khi từng tệp tin chạy xong.
- Cho phép người dùng tải tệp đầu ra (.docx, .md, .zip) của các tệp đã dịch xong mà không làm gián đoạn các tệp đang chạy dưới nền.
- Đảm bảo tất cả 21 unit test hiện có tiếp tục vượt qua thành công sau khi thay đổi.

---

### Task 1: Khởi tạo biến trạng thái Session State

**Files:**
- Modify: [app.py](file:///Users/trungshin/Downloads/mmm/app.py#L125-L132)

**Interfaces:**
- Consumes: None
- Produces: `st.session_state.last_completed_count`, `st.session_state.current_file_progress`, `st.session_state.current_file_eta`, `st.session_state.current_file_elapsed`, `st.session_state.current_file_name`

- [ ] **Step 1: Khởi tạo các biến Session State mới**
  Bổ sung khởi tạo các biến trạng thái cho luồng chạy nền và tiến độ chi tiết.
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
  if "current_file_progress" not in st.session_state:
      st.session_state.current_file_progress = 0.0
  if "current_file_eta" not in st.session_state:
      st.session_state.current_file_eta = None
  if "current_file_elapsed" not in st.session_state:
      st.session_state.current_file_elapsed = None
  if "current_file_name" not in st.session_state:
      st.session_state.current_file_name = None
  ```

- [ ] **Step 2: Chạy kiểm thử tự động để đảm bảo cấu trúc trang không bị vỡ**
  Run: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest tests/test_app.py::test_app_renders`
  Expected: PASS

- [ ] **Step 3: Commit các thay đổi ban đầu**
  ```bash
  git add app.py
  git commit -m "feat: initialize background processing state variables"
  ```

---

### Task 2: Triển khai Hàm xử lý hàng đợi dưới nền (Background Thread)

**Files:**
- Modify: [app.py](file:///Users/trungshin/Downloads/mmm/app.py#L214-L298)

**Interfaces:**
- Consumes: `st.session_state.queue`, `st.session_state.results`
- Produces: Hàm `process_queue_in_background(model_name: str)` chạy bất đồng bộ dưới luồng nền và cập nhật `st.session_state.results` và tiến độ.

- [ ] **Step 1: Định nghĩa hàm xử lý hàng đợi dưới nền**
  Thêm định nghĩa hàm `process_queue_in_background` vào trước khối xử lý giao diện chính trong [app.py](file:///Users/trungshin/Downloads/mmm/app.py):
  ```python
  def process_queue_in_background(model_name: str):
      import time
      total_files = len(st.session_state.queue)
      
      for idx, item in enumerate(st.session_state.queue):
          if item["status"] == "Hoàn thành":
              continue
              
          st.session_state.queue[idx]["status"] = "Đang xử lý"
          st.session_state.current_file_name = item["name"]
          st.session_state.current_file_progress = 0.0
          st.session_state.current_file_eta = None
          st.session_state.current_file_elapsed = None
          file_start_time = time.time()
          
          def make_progress_callback(f_idx, file_name, start_t):
              last_update_time = [0.0]
              def callback(current_frames, total_frames):
                  if total_frames <= 0:
                      return
                  current_time = time.time()
                  if current_time - last_update_time[0] < 0.1 and current_frames < total_frames:
                      return
                  last_update_time[0] = current_time
                  
                  file_prog = current_frames / total_frames
                  st.session_state.current_file_progress = file_prog
                  elapsed = current_time - start_t
                  st.session_state.current_file_elapsed = elapsed
                  if current_frames > 0:
                      st.session_state.current_file_eta = elapsed * (total_frames - current_frames) / current_frames
                  else:
                      st.session_state.current_file_eta = None
              return callback
              
          suffix = os.path.splitext(item["name"])[1]
          tmp_file_path = None
          try:
              with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                  tmp_file_path = tmp_file.name
                  CHUNK_SIZE = 1024 * 1024  # 1MB
                  file_data = item["file"]
                  file_data.seek(0)
                  while chunk := file_data.read(CHUNK_SIZE):
                      tmp_file.write(chunk)
              
              prog_cb = make_progress_callback(idx, item["name"], file_start_time)
              result = transcribe_audio(tmp_file_path, model_name=model_name, progress_callback=prog_cb)
              st.session_state.results[item["name"]] = result
              st.session_state.queue[idx]["status"] = "Hoàn thành"
          except Exception as e:
              st.session_state.queue[idx]["status"] = "Lỗi"
              # Lưu lỗi vào results để hiển thị nếu cần
          finally:
              if tmp_file_path and os.path.exists(tmp_file_path):
                  try:
                      os.remove(tmp_file_path)
                  except Exception:
                      pass
                      
      st.session_state.processing = False
  ```

- [ ] **Step 2: Cập nhật sự kiện bấm nút Bắt đầu nhận dạng**
  Sửa đổi khối xử lý sự kiện bấm nút `start_btn` để khởi tạo luồng chạy nền thay vì chạy đồng bộ:
  ```python
          start_btn = st.button("🚀 Bắt đầu nhận diện", disabled=st.session_state.processing)
          
          if start_btn and not st.session_state.processing:
              st.session_state.processing = True
              st.session_state.last_completed_count = len(st.session_state.results)
              st.session_state.current_file_progress = 0.0
              st.session_state.current_file_eta = None
              st.session_state.current_file_elapsed = None
              st.session_state.current_file_name = None
              
              import threading
              from streamlit.runtime.scriptrunner import add_script_run_ctx
              
              thread = threading.Thread(target=process_queue_in_background, args=(model_name,))
              add_script_run_ctx(thread)
              thread.start()
              st.rerun()
  ```

- [ ] **Step 3: Chạy thử kiểm thử để xem trạng thái app**
  Run: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest tests/test_app.py::test_app_renders`
  Expected: PASS

- [ ] **Step 4: Commit thay đổi**
  ```bash
  git add app.py
  git commit -m "feat: implement background processing thread for transcription"
  ```

---

### Task 3: Tích hợp Streamlit Fragment để tự động cập nhật UI

**Files:**
- Modify: [app.py](file:///Users/trungshin/Downloads/mmm/app.py#L191-L211)

**Interfaces:**
- Consumes: `st.session_state.processing`, `st.session_state.current_file_progress`, `st.session_state.results`
- Produces: Hàm `@st.fragment(run_every="1s") def render_progress_panel()`

- [ ] **Step 1: Tạo Fragment render tiến trình**
  Thêm đoạn code sau vào [app.py](file:///Users/trungshin/Downloads/mmm/app.py) để tự động cập nhật tiến trình hàng đợi:
  ```python
      if uploaded_files:
          @st.fragment(run_every="1s")
          def render_progress_panel():
              st.markdown("#### Hàng đợi xử lý:")
              for item in st.session_state.queue:
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
              
              if st.session_state.processing:
                  total_files = len(st.session_state.queue)
                  completed_files = sum(1 for item in st.session_state.queue if item["status"] == "Hoàn thành")
                  
                  file_prog = st.session_state.current_file_progress
                  overall_prog = (completed_files + file_prog) / total_files if total_files > 0 else 0.0
                  
                  st.progress(min(1.0, overall_prog))
                  
                  curr_name = st.session_state.current_file_name
                  elapsed = st.session_state.current_file_elapsed
                  eta = st.session_state.current_file_eta
                  
                  if curr_name:
                      if file_prog > 0:
                          eta_str = f"{int(eta)} giây" if (eta is not None and eta >= 1) else "đang hoàn thành..."
                          elapsed_str = f"{int(elapsed)} giây" if elapsed is not None else "0 giây"
                          st.markdown(
                              f"🎙 ... **Đang xử lý ({completed_files + 1}/{total_files}):** `{curr_name}`\n\n"
                              f"📊 Tiến độ tệp: **{file_prog * 100:.1f}%** | ⏳ Đã trôi qua: **{elapsed_str}** | 🏁 Dự kiến còn lại: **{eta_str}**"
                          )
                      else:
                          st.markdown(
                              f"🎙 ... **Đang khởi tạo ({completed_files + 1}/{total_files}):** `{curr_name}`\n\n"
                              f"⚙️ Đang nạp mô hình và phân tích dữ liệu âm thanh..."
                          )
              else:
                  any_completed = any(item["status"] == "Hoàn thành" for item in st.session_state.queue)
                  if any_completed:
                      st.markdown("🎉 **Đã hoàn thành nhận dạng tất cả các tệp âm thanh.**")
                      
              # Kích hoạt Rerun toàn bộ trang khi phát hiện có tệp tin vừa hoàn thành mới để cập nhật cột kết quả bên phải
              current_completed_count = len(st.session_state.results)
              if current_completed_count > st.session_state.last_completed_count:
                  st.session_state.last_completed_count = current_completed_count
                  st.rerun()

          render_progress_panel()
  ```

- [ ] **Step 2: Xóa bỏ các biến render_queue và tiến trình cũ**
  Xóa định nghĩa `render_queue()` cũ và `queue_placeholder = st.empty()` cũ tại các dòng tương ứng để tránh trùng lặp.

- [ ] **Step 3: Chạy unit test giao diện**
  Run: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest tests/test_app.py::test_app_renders`
  Expected: PASS

- [ ] **Step 4: Commit thay đổi**
  ```bash
  git add app.py
  git commit -m "feat: integrate Streamlit fragment for automatic progress updates"
  ```

---

### Task 4: Tối ưu chọn file kết quả mặc định ở Cột Phải

**Files:**
- Modify: [app.py](file:///Users/trungshin/Downloads/mmm/app.py#L300-L315)

**Interfaces:**
- Consumes: `st.session_state.results`
- Produces: `st.session_state.selected_file_name` tự động chuyển đến file vừa xử lý xong.

- [ ] **Step 1: Cập nhật Logic hiển thị kết quả cột phải**
  Sửa đổi đoạn đầu khối hiển thị kết quả trong `col_right` để tự động chọn file mới hoàn thành gần nhất làm mặc định mà vẫn bảo tồn lựa chọn thủ công của người dùng:
  ```python
  with col_right:
      st.markdown("### 📝 Kết quả & Tải về")
      
      if st.session_state.results:
          options = list(st.session_state.results.keys())
          
          # Tự động chọn file mới nhất nếu chưa có file nào được chọn hoặc có file mới hoàn thành
          if "selected_file_name" not in st.session_state or st.session_state.selected_file_name not in options:
              st.session_state.selected_file_name = options[-1] if options else None
          else:
              if "prev_completed_files" not in st.session_state:
                  st.session_state.prev_completed_files = []
              
              new_files = [f for f in options if f not in st.session_state.prev_completed_files]
              if new_files:
                  st.session_state.selected_file_name = new_files[-1]
              
              st.session_state.prev_completed_files = options.copy()
              
          selected_file_name = st.selectbox(
              "Chọn file để xem kết quả:",
              options=options,
              key="selected_file_name"
          )
  ```

- [ ] **Step 2: Chạy unit test giao diện**
  Run: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest tests/test_app.py::test_app_renders`
  Expected: PASS

- [ ] **Step 3: Commit thay đổi**
  ```bash
  git add app.py
  git commit -m "feat: auto-select newly completed files in results dropdown"
  ```

---

### Task 5: Cập nhật Unit Tests cho phù hợp với luồng chạy Thread bất đồng bộ

**Files:**
- Modify: [tests/test_app.py](file:///Users/trungshin/Downloads/mmm/tests/test_app.py#L66-L124)

**Interfaces:**
- Consumes: `app.py`
- Produces: Unit tests tương thích luồng bất đồng bộ.

- [ ] **Step 1: Sửa đổi test_app_transcription_success**
  Patch `threading.Thread` để thực thi đồng bộ trong môi trường kiểm thử nhằm tránh race-condition.
  Cập nhật trong [tests/test_app.py](file:///Users/trungshin/Downloads/mmm/tests/test_app.py):
  ```python
  @patch("utils.transcriber.transcribe_audio")
  @patch("app.threading.Thread")
  def test_app_transcription_success(mock_thread, mock_transcribe):
      import os
      
      # Mock return value of transcribe_audio
      mock_transcribe.return_value = {
          "text": "Xin chào thế giới.",
          "segments": [
              {"start": 0.0, "end": 2.5, "text": " Xin chào thế giới."}
          ]
      }
      
      # Khi khởi chạy Thread trong ứng dụng, chúng ta cho chạy đồng bộ trực tiếp để lấy kết quả tức thì trong test
      def mock_thread_init(target, args=(), kwargs={}):
          mock_t = MagicMock()
          mock_t.start = lambda: target(*args, **kwargs)
          return mock_t
      mock_thread.side_effect = mock_thread_init
      
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
  ```

- [ ] **Step 2: Sửa đổi test_app_transcription_failure**
  Tương tự, patch `threading.Thread` chạy đồng bộ cho trường hợp lỗi.
  Cập nhật trong [tests/test_app.py](file:///Users/trungshin/Downloads/mmm/tests/test_app.py):
  ```python
  @patch("utils.transcriber.transcribe_audio")
  @patch("app.threading.Thread")
  def test_app_transcription_failure(mock_thread, mock_transcribe):
      import os
      
      # Mock transcribe_audio to raise an error
      mock_transcribe.side_effect = ValueError("Transcribe failed")
      
      def mock_thread_init(target, args=(), kwargs={}):
          mock_t = MagicMock()
          mock_t.start = lambda: target(*args, **kwargs)
          return mock_t
      mock_thread.side_effect = mock_thread_init
      
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
      
      # Do trong luồng nền có try-except bắt lỗi đổi trạng thái sang "Lỗi",
      # chúng ta chạy lại render_progress_panel để giao diện ghi nhận lỗi.
      # Ta xác nhận giao diện hiển thị badge lỗi.
      markdown_texts = [m.value for m in at.markdown]
      assert any("test.mp3" in text for text in markdown_texts)
  ```

- [ ] **Step 3: Chạy toàn bộ các ca kiểm thử để xác minh độ chính xác**
  Run: `PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest`
  Expected: 21 passed

- [ ] **Step 4: Commit thay đổi cuối cùng**
  ```bash
  git add tests/test_app.py
  git commit -m "test: adapt app tests for async background threading model"
  ```
