import streamlit as st
import tempfile
import os
import io
from utils.transcriber import transcribe_audio, force_clear_gpu_cache
from utils.exporter import export_to_markdown, export_to_docx, export_to_zip, format_timestamp
from utils.queue_manager import QueueManager

qm = QueueManager()

# 1. Page Configuration
st.set_page_config(
    page_title="VietWhisper - Chuyển đổi Âm thanh tiếng Việt sang Văn bản",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styles (Glassmorphism & Aesthetics)
glass_css = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* Reset and custom font */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0F0F23;
        background-image: radial-gradient(circle at 10% 20%, rgba(255, 122, 0, 0.08) 0%, transparent 40%),
                          radial-gradient(circle at 90% 80%, rgba(255, 0, 122, 0.08) 0%, transparent 40%);
        background-attachment: fixed;
        color: #FFFFFF;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Overwrite Streamlit container default styling with Glassmorphism */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.07) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF7A00 0%, #FF007A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 122, 0, 0.2);
        margin-bottom: 8px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 30px;
    }
    
    /* Button Custom styling */
    div.stButton > button {
        background: linear-gradient(135deg, #FF7A00 0%, #FF007A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(255, 122, 0, 0.3) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 122, 0, 0.5) !important;
    }
    
    /* Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .status-pending { background: rgba(255, 193, 7, 0.15); color: #FFC107; border: 1px solid rgba(255, 193, 7, 0.3); }
    .status-processing { background: rgba(0, 123, 255, 0.15); color: #007BFF; border: 1px solid rgba(0, 123, 255, 0.3); }
    .status-done { background: rgba(40, 167, 69, 0.15); color: #28A745; border: 1px solid rgba(40, 167, 69, 0.3); }
    .status-error { background: rgba(220, 53, 69, 0.15); color: #DC3545; border: 1px solid rgba(220, 53, 69, 0.3); }
    
    .timestamp-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #FF7A00;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-right: 10px;
    }
    
    .transcript-line {
        margin-bottom: 12px;
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.02);
    }
    
    .scroll-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    .scroll-container::-webkit-scrollbar {
        width: 6px;
    }
    .scroll-container::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)

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
if "prev_completed_files" not in st.session_state:
    st.session_state.prev_completed_files = []

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


# 4. Header Section
st.markdown('<h1 class="main-title">🎙️ VietWhisper</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ứng dụng chuyển đổi âm thanh tiếng Việt sang văn bản chất lượng cao, tối ưu Apple Silicon GPU</p>', unsafe_allow_html=True)

# 5. UI Layout - Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình hệ thống")
    
    model_name = st.selectbox(
        "Mô hình Whisper (MLX):",
        options=[
            "mlx-community/whisper-large-v3-4bit",
            "mlx-community/whisper-large-v3-turbo-4bit",
            "mlx-community/whisper-base-4bit",
            "mlx-community/whisper-large-v3"
        ],
        index=0
    )
    
    export_format = st.selectbox(
        "Định dạng xuất:",
        options=[".docx", ".md"],
        index=0
    )
    
    include_timestamps = st.toggle("Bật mốc thời gian (Timestamps)", value=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Giải phóng GPU Metal Cache", key="clear_gpu_btn"):
        force_clear_gpu_cache()
        st.success("Đã giải phóng bộ nhớ cache của mô hình và Metal GPU.")

# 6. Main UI Content - File Upload & Processing
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📤 Tải lên tệp âm thanh")
    
    uploaded_files = st.file_uploader(
        "Kéo thả các file âm thanh vào đây (Hỗ trợ: mp3, m4a, wav, flac, ogg - Tối đa 1GB):",
        type=["mp3", "m4a", "wav", "flac", "ogg"],
        accept_multiple_files=True
    )
    
    # Đồng bộ hóa danh sách tệp chờ xử lý với QueueManager
    uploaded_file_names = {uf.name for uf in uploaded_files} if uploaded_files else set()
    for status_item in qm.get_status():
        if status_item["status"] == "Chờ xử lý" and status_item["name"] not in uploaded_file_names:
            qm.remove_task(status_item["name"])
            
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
            qm.start_worker(model_name)
            st.rerun()
with col_right:
    st.markdown("### 📝 Kết quả & Tải về")
    
    if st.session_state.results:
        options = list(st.session_state.results.keys())
        
        # Tự động chọn file mới nhất nếu chưa có file nào được chọn
        if "selected_file_name" not in st.session_state or st.session_state.selected_file_name not in options:
            st.session_state.selected_file_name = options[-1] if options else None
        else:
            # Hiển thị toast thông báo file mới dịch xong
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
        
        if selected_file_name:
            res = st.session_state.results[selected_file_name]
            if "error" in res:
                st.error(f"Lỗi khi xử lý file {selected_file_name}: {res['error']}")
            else:
                segments = res.get("segments", [])
                
                # Formatted Preview
                st.markdown("#### Xem trước đoạn hội thoại:")
                st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
                
                html_transcript = []
                plain_text_lines = []
                for seg in segments:
                    ts = format_timestamp(seg.get("start", 0.0))
                    text = seg.get("text", "").strip()
                    
                    if include_timestamps:
                        html_line = f'<div class="transcript-line"><span class="timestamp-badge">{ts}</span>{text}</div>'
                        plain_line = f"{ts} {text}"
                    else:
                        html_line = f'<div class="transcript-line">{text}</div>'
                        plain_line = text
                    
                    html_transcript.append(html_line)
                    plain_text_lines.append(plain_line)
                    
                st.markdown("\n".join(html_transcript), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clipboard Copy native component
                full_plain_text = "\n".join(plain_text_lines)
                st.markdown("#### Bản thô để sao chép nhanh (Nhấp nút copy ở góc trên bên phải):")
                st.code(full_plain_text, language="text")
                
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
                if hasattr(single_file_data, "seek"):
                    single_file_data.seek(0)
                
                st.download_button(
                    label=f"📥 Tải xuống tệp {single_file_name}",
                    data=single_file_data,
                    file_name=single_file_name,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document" if export_format == ".docx" else "text/markdown",
                    key="download_single_btn"
                )
            
            # Lazy generate và cache file nén ZIP tổng hợp khi có >=2 file thành công
            success_results = {k: v for k, v in st.session_state.results.items() if "error" not in v}
            if len(success_results) > 1:
                completed_keys = tuple(sorted(success_results.keys()))
                zip_cache_key = (completed_keys, export_format, include_timestamps)
                
                if zip_cache_key not in st.session_state.zip_cache:
                    zip_export_files = {}
                    for name, r in success_results.items():
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
                if hasattr(zip_bio, "seek"):
                    zip_bio.seek(0)
                st.download_button(
                    label="🗜️ Tải xuống toàn bộ tệp (.zip)",
                    data=zip_bio,
                    file_name="vietwhisper_transcripts.zip",
                    mime="application/zip",
                    key="download_zip_btn"
                )
    else:
        st.info("Chưa có kết quả. Vui lòng tải file lên và ấn nút Bắt đầu nhận diện.")
