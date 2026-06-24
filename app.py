import streamlit as st
import tempfile
import os
import io
from utils.transcriber import transcribe_audio, force_clear_gpu_cache
from utils.exporter import export_to_markdown, export_to_docx, export_to_zip, format_timestamp

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
    
    # Synchronize uploaded files and session queue to prevent progress wipeouts on rerun
    uploaded_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    current_queue_names = [item["name"] for item in st.session_state.queue]
    
    if uploaded_file_names != current_queue_names:
        st.session_state.queue = []
        if uploaded_files:
            for uf in uploaded_files:
                status = "Chờ xử lý"
                if uf.name in st.session_state.results:
                    status = "Hoàn thành"
                st.session_state.queue.append({"name": uf.name, "file": uf, "status": status})
                
    if uploaded_files:
        queue_placeholder = st.empty()
        
        def render_queue():
            with queue_placeholder.container():
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
        
        render_queue()
        start_btn = st.button("🚀 Bắt đầu nhận diện")
        
        if start_btn and not st.session_state.processing:
            st.session_state.processing = True
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                for idx, item in enumerate(st.session_state.queue):
                    if item["status"] == "Hoàn thành":
                        continue
                    
                    # Update status and render UI live at the spot
                    st.session_state.queue[idx]["status"] = "Đang xử lý"
                    render_queue()
                    status_text.text(f"Đang xử lý: {item['name']}...")
                    
                    suffix = os.path.splitext(item["name"])[1]
                    tmp_file_path = None
                    try:
                        # Write uploaded file in chunks to prevent OOM
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file_path = tmp_file.name
                            CHUNK_SIZE = 1024 * 1024  # 1MB
                            file_data = item["file"]
                            file_data.seek(0)
                            while chunk := file_data.read(CHUNK_SIZE):
                                tmp_file.write(chunk)
                        
                        result = transcribe_audio(tmp_file_path, model_name=model_name)
                        st.session_state.results[item["name"]] = result
                        st.session_state.queue[idx]["status"] = "Hoàn thành"
                    except Exception as e:
                        st.session_state.queue[idx]["status"] = "Lỗi"
                        st.error(f"Lỗi khi xử lý file {item['name']}: {str(e)}")
                    finally:
                        if tmp_file_path and os.path.exists(tmp_file_path):
                            os.remove(tmp_file_path)
                    
                    render_queue()
                    progress = int((idx + 1) / len(st.session_state.queue) * 100)
                    progress_bar.progress(progress)
                
                status_text.text("Đã hoàn thành nhận dạng tất cả các tệp âm thanh.")
            finally:
                st.session_state.processing = False
            
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()


with col_right:
    st.markdown("### 📝 Kết quả & Tải về")
    
    if st.session_state.results:
        selected_file_name = st.selectbox(
            "Chọn file để xem kết quả:",
            options=list(st.session_state.results.keys())
        )
        
        if selected_file_name:
            res = st.session_state.results[selected_file_name]
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
            
            # Lazy export formatting
            export_files = {}
            for name, r in st.session_state.results.items():
                segs = r.get("segments", [])
                if export_format == ".docx":
                    out_bio = export_to_docx(segs, include_timestamps)
                    export_files[os.path.splitext(name)[0] + ".docx"] = out_bio
                else:
                    out_bio = export_to_markdown(segs, include_timestamps)
                    export_files[os.path.splitext(name)[0] + ".md"] = out_bio
            
            single_file_name = os.path.splitext(selected_file_name)[0] + export_format
            single_file_data = export_files[single_file_name]
            
            st.download_button(
                label=f"📥 Tải xuống tệp {single_file_name}",
                data=single_file_data,
                file_name=single_file_name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document" if export_format == ".docx" else "text/markdown"
            )
            
            if len(st.session_state.results) > 1:
                zip_bio = export_to_zip(export_files)
                st.download_button(
                    label="🗜️ Tải xuống toàn bộ tệp (.zip)",
                    data=zip_bio,
                    file_name="vietwhisper_transcripts.zip",
                    mime="application/zip"
                )
    else:
        st.info("Chưa có kết quả. Vui lòng tải file lên và ấn nút Bắt đầu nhận diện.")
