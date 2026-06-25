# Đặc tả Thiết kế: Xem và Tải kết quả Nhận diện Tức thời (Immediate Results Preview & Download)

Tài liệu này đặc tả thiết kế chi tiết cho tính năng hiển thị kết quả ngay lập tức và cho phép tải trước file output của từng file âm thanh sau khi hoàn thành xử lý, thay vì phải đợi toàn bộ hàng đợi chạy xong trong ứng dụng **VietWhisper**.

---

## 1. Mục tiêu Thiết kế (Design Goal)
*   **Không chặn giao diện (Non-blocking UI)**: Người dùng có thể xem kết quả, sao chép văn bản nhận diện và tải xuống các tệp tin đầu ra (`.docx`, `.md`, `.zip`) của các tệp âm thanh đã xử lý xong **ngay trong khi** các tệp âm thanh khác trong hàng đợi vẫn đang được xử lý.
*   **Tiến độ trực quan theo thời gian thực (Real-time Progress)**: Hiển thị tiến trình phần trăm giải mã chi tiết của tệp tin đang xử lý và trạng thái hàng đợi được cập nhật tự động mỗi giây.
*   **An toàn và ổn định**: Đảm bảo việc tương tác với giao diện (tải file, copy, chuyển tab) không làm gián đoạn tiến trình giải mã GPU Metal dưới nền.

---

## 2. Giải pháp Kiến trúc (Architectural Solution)

Giải pháp sử dụng kết hợp **Luồng nền (Background Thread)** của Python và **Khung cập nhật cục bộ (Streamlit Fragments)** của Streamlit 1.58.

### 2.1. Quản lý Luồng nhận dạng nền (Background Threading)
*   Khi người dùng nhấn nút "🚀 Bắt đầu nhận diện", thay vì thực thi vòng lặp đồng bộ chặn luồng chính, ứng dụng sẽ tạo một luồng chạy nền mới:
    ```python
    import threading
    from streamlit.runtime.scriptrunner import add_script_run_ctx

    thread = threading.Thread(target=process_queue_in_background)
    add_script_run_ctx(thread)  # Liên kết luồng với Streamlit Session State của client hiện tại
    thread.start()
    ```
*   Luồng nền sẽ xử lý tuần tự từng tệp tin trong `st.session_state.queue` để tránh quá tải bộ nhớ GPU. Dữ liệu tiến độ giải mã chi tiết sẽ được ghi trực tiếp vào `st.session_state` thông qua tiến trình `progress_callback`.

### 2.2. Đồng bộ & Cập nhật UI tự động (Streamlit Fragments)
*   Sử dụng tính năng `@st.fragment` của Streamlit để bọc phần hiển thị Hàng đợi và Tiến trình nhận diện ở cột bên trái.
*   Cấu hình `@st.fragment(run_every="1s")` để phần giao diện này tự động chạy lại (rerun) mỗi 1 giây để đọc tiến độ mới nhất từ `st.session_state` mà không làm tải lại toàn bộ trang web.
*   Trong Fragment, kiểm tra số lượng kết quả đã hoàn thành trong `st.session_state.results`. Nếu số lượng kết quả tăng lên so với lần render trước, Fragment sẽ gọi `st.rerun()` để làm mới toàn bộ trang. Việc này giúp cập nhật hộp lựa chọn (selectbox) và các nút tải xuống ở cột bên phải ngay khi có tệp tin vừa xử lý xong.

---

## 3. Chi tiết Thay đổi Mã nguồn (Proposed Changes)

### 3.1. [app.py](file:///Users/trungshin/Downloads/mmm/app.py)
*   **Khởi tạo State**: Bổ sung các biến trạng thái vào `st.session_state`:
    *   `last_completed_count`: Số lượng tệp đã hoàn thành ở lượt render trước (mặc định là 0).
    *   `current_file_progress`: Tiến độ hiện tại của file đang xử lý (0.0 đến 1.0).
    *   `current_file_eta`: Thời gian dự kiến còn lại của file đang xử lý.
    *   `current_file_elapsed`: Thời gian đã trôi qua của file đang xử lý.
*   **Hàm Xử lý Hàng đợi trong Luồng nền (`process_queue_in_background`)**:
    *   Thực hiện tuần tự các bước: ghi tệp tạm, gọi `transcribe_audio` với callback cập nhật tiến độ chi tiết vào session state, lưu kết quả nhận dạng vào `st.session_state.results`, đổi trạng thái tệp thành "Hoàn thành" và dọn dẹp tệp tạm.
    *   Bảo vệ bằng khối `try-finally` để đảm bảo `st.session_state.processing` được đặt về `False` khi kết thúc hoặc xảy ra lỗi.
*   **Tích hợp Streamlit Fragment**:
    *   Tạo hàm `@st.fragment(run_every="1s") def render_progress_panel():` hiển thị danh sách hàng đợi và thanh tiến trình.
    *   Trong Fragment này, so sánh số lượng key của `st.session_state.results` với `st.session_state.last_completed_count`. Nếu lớn hơn, cập nhật `last_completed_count` và gọi `st.rerun()`.
*   **Cập nhật logic cột phải (`col_right`)**:
    *   Cột phải sẽ luôn đọc từ `st.session_state.results` hiện tại để tạo selectbox và các nút tải xuống tương ứng.
    *   Nếu danh sách file hoàn thành thay đổi, tự động chọn file vừa hoàn thành gần nhất làm mặc định để người dùng xem được kết quả ngay lập tức.

### 3.2. [tests/test_app.py](file:///Users/trungshin/Downloads/mmm/tests/test_app.py)
*   Cập nhật các ca kiểm thử `test_app_transcription_success` và `test_app_transcription_failure` để phù hợp với kiến trúc chạy nền bằng Thread.
*   Vì `AppTest` chạy đồng bộ, chúng ta có thể mock hoặc đợi luồng nền chạy xong trong môi trường kiểm thử (sử dụng `.join()` trên luồng nền nếu cần, hoặc mock trực tiếp hành vi chạy thread thành chạy đồng bộ trong môi trường test).

---

## 4. Kế hoạch Kiểm thử & Xác minh (Verification Plan)

### 4.1. Kiểm thử Tự động (Automated Verification)
*   Chạy toàn bộ unit test hiện có để đảm bảo không bị hồi quy lỗi (regression):
    ```bash
    PYTHONPATH=. /Users/trungshin/miniconda3/envs/whisper-mac/bin/pytest
    ```

### 4.2. Kiểm thử Thủ công (Manual Verification)
1.  **Kiểm tra tính tương tác**: Tải lên 3 file âm thanh. Bấm "Bắt đầu nhận diện".
2.  **Xác minh kết quả tức thời**: Sau khi file thứ 1 hoàn thành, xác minh cột kết quả bên phải tự động hiển thị bản dịch của file 1 và xuất hiện nút "Tải xuống" tương ứng, trong khi file thứ 2 vẫn đang được xử lý tiếp tục.
3.  **Xác minh tải xuống khi đang chạy**: Nhấp vào nút "Tải xuống" của file 1 khi file 2 đang nhận diện. Xác minh tệp tải về thành công và luồng nhận diện file 2 không bị lỗi hay dừng lại.
4.  **Xác minh giải phóng tài nguyên**: Theo dõi log hệ thống và đảm bảo bộ nhớ Metal GPU được giải phóng đúng cách sau khi toàn bộ hàng đợi hoàn thành.
