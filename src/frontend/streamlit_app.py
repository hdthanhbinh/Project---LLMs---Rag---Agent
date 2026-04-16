# src/frontend/streamlit_app.py
# Người 3 – Frontend: Giao diện Streamlit
# Gọi vào RAGChain (Người 1) thông qua app.py

import sys
import os
import tempfile
from pathlib import Path

import streamlit as st

# Trỏ về root project để import được app.py và src/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import RAGChain, DocumentFilter  # Người 1 — DocumentFilter định nghĩa trong app.py


# ------------------------------------------------------------------ #
#  Cấu hình trang                                                      #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ #
#  CSS tuỳ chỉnh                                                       #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #2E2C33;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #FFFFFF !important;
}

/* Chat bubble người dùng */
.user-bubble {
    background: #007BFF;
    color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0;
    max-width: 80%;
    margin-left: auto;
    word-wrap: break-word;
}

/* Chat bubble AI */
.ai-bubble {
    background: #F1F3F5;
    color: #212529;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 80%;
    word-wrap: break-word;
}

/* Source citation box */
.source-box {
    background: #FFF8E1;
    border-left: 3px solid #FFC107;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 0.85em;
    margin-top: 6px;
    color: #212529;
}

/* History entry box */
.history-box {
    background: #3A3840;
    border-left: 3px solid #007BFF;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.82em;
    margin-bottom: 6px;
    color: #E0E0E0;
}

/* Header */
.main-header {
    color: #007BFF;
    font-size: 2em;
    font-weight: 700;
    margin-bottom: 0;
}
.sub-header {
    color: #6C757D;
    font-size: 1em;
    margin-top: 0;
    margin-bottom: 24px;
}
[data-testid="stSidebar"] .stButton button {
    background-color: #444950 !important;
    color: #FFFFFF !important;
    border: 1px solid #666 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #555C63 !important;
    border: 1px solid #888 !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Khởi tạo session state                                              #
# ------------------------------------------------------------------ #
def init_session():
    if "rag" not in st.session_state:
        st.session_state.rag = RAGChain()
        # Thử load FAISS index từ disk nếu đã có (khởi động lại app)
        if st.session_state.rag.load_from_disk_and_build():
            st.session_state.index_ready = True
        else:
            st.session_state.index_ready = False

    if "chat_history" not in st.session_state:
        # Mỗi item: { "role": "user"|"ai", "content": str, "sources": list }
        st.session_state.chat_history = []

    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False

    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None

    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 0

    # ── MỚI: lưu nguồn file đang được lọc (dùng với DocumentFilter) ──
    if "filter_source" not in st.session_state:
        st.session_state.filter_source = None   # None = không lọc (tìm toàn bộ)


init_session()


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 📄 SmartDoc AI")
    st.markdown("---")

    st.markdown("### 📖 Hướng dẫn")
    st.markdown("""
1. Upload file **PDF** hoặc **DOCX**
2. Chờ hệ thống xử lý tài liệu
3. Đặt câu hỏi bất kỳ về nội dung
4. Xem câu trả lời kèm nguồn trích dẫn
""")

    st.markdown("---")
    st.markdown("### ⚙️ Cấu hình")
    st.markdown(f"**Model:** `qwen2.5:7b`")
    st.markdown(f"**Embedding:** `MPNet 768-dim`")
    st.markdown(f"**Retriever:** Hybrid (FAISS + BM25)")

    if st.session_state.index_ready:
        st.success(f"✅ Đã index **{st.session_state.num_chunks}** chunks")
        if st.session_state.uploaded_filename:
            st.markdown(f"📁 `{st.session_state.uploaded_filename}`")

    # ------------------------------------------------------------------ #
    #  MỚI: DocumentFilter — lọc tìm kiếm theo tên file                   #
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.markdown("### 🔍 Lọc tài liệu")

    # Chỉ cho phép lọc khi đã có file được index
    if st.session_state.index_ready and st.session_state.uploaded_filename:
        use_filter = st.checkbox(
            "Chỉ tìm trong file hiện tại",
            value=st.session_state.filter_source is not None,
            help="Bật để giới hạn tìm kiếm trong file đang được nạp.",
        )
        if use_filter:
            st.session_state.filter_source = st.session_state.uploaded_filename
            st.caption(f"🎯 Đang lọc: `{st.session_state.filter_source}`")
        else:
            st.session_state.filter_source = None
            st.caption("🌐 Tìm kiếm toàn bộ tài liệu")
    else:
        st.caption("_Upload tài liệu để kích hoạt bộ lọc._")

    st.markdown("---")
    st.markdown("### 🗑️ Xoá dữ liệu")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Xoá chat", use_container_width=True):
            # ── MỚI: xoá cả chat_history.json qua clear_history() ──
            st.session_state.rag.clear_history()
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Reset index", use_container_width=True):
            # ── MỚI: xoá luôn history khi reset toàn bộ ──
            st.session_state.rag.clear_history()
            st.session_state.rag = RAGChain()
            st.session_state.index_ready = False
            st.session_state.uploaded_filename = None
            st.session_state.num_chunks = 0
            st.session_state.chat_history = []
            st.session_state.filter_source = None
            st.rerun()

    st.markdown("---")

    # ------------------------------------------------------------------ #
    #  MỚI: Lịch sử hội thoại lấy từ get_history() — persistent JSON      #
    # ------------------------------------------------------------------ #
    st.markdown("### 💬 Lịch sử hội thoại")

    try:
        history_entries = st.session_state.rag.get_history()
    except Exception:
        history_entries = []

    if history_entries:
        # Hiển thị 10 câu gần nhất, mới nhất ở trên
        for entry in reversed(history_entries[-10:]):
            q_preview = entry.get("question", "")[:60]
            if len(entry.get("question", "")) > 60:
                q_preview += "..."
            ts = entry.get("timestamp", "")[:16].replace("T", " ")  # "2026-04-16 07:23"
            latency = entry.get("meta", {}).get("latency", "")
            latency_str = f" · {latency}s" if latency else ""
            st.markdown(
                f'<div class="history-box">'
                f'<b>#{entry.get("id", "?")}</b> {q_preview}<br>'
                f'<span style="font-size:0.78em;color:#AAA;">{ts}{latency_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("_Chưa có lịch sử hội thoại._")


# ------------------------------------------------------------------ #
#  Main area                                                           #
# ------------------------------------------------------------------ #
st.markdown('<p class="main-header">📄 SmartDoc AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Document Q&A System</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Upload file                                                         #
# ------------------------------------------------------------------ #
st.markdown("### 📂 Upload tài liệu")

uploaded_file = st.file_uploader(
    label="Chọn file PDF hoặc DOCX",
    type=["pdf", "docx"],
    help="Tối đa 10MB. Hỗ trợ tiếng Việt và tiếng Anh.",
)

if uploaded_file is not None:
    # Chỉ xử lý khi file mới khác file cũ
    if uploaded_file.name != st.session_state.uploaded_filename:
        with st.spinner(f"⏳ Đang xử lý **{uploaded_file.name}**..."):
            try:
                # Lưu file tạm để xử lý
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                # Gọi RAGChain (Người 1)
                num_chunks = st.session_state.rag.load_and_build(tmp_path)

                # Cập nhật session state
                st.session_state.index_ready = True
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.num_chunks = num_chunks
                st.session_state.chat_history = []      # reset chat khi đổi file
                st.session_state.filter_source = None   # reset filter khi đổi file

                # Dọn file tạm
                os.unlink(tmp_path)

                st.success(f"✅ Đã xử lý **{uploaded_file.name}** — {num_chunks} chunks")

            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý file: {e}")
    else:
        st.info(f"✅ **{uploaded_file.name}** đã được nạp ({st.session_state.num_chunks} chunks)")

st.markdown("---")

# ------------------------------------------------------------------ #
#  Hiển thị lịch sử chat                                              #
# ------------------------------------------------------------------ #
st.markdown("### 💬 Hỏi đáp")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ai-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Hiển thị sources nếu có
            if msg.get("sources"):
                with st.expander("📎 Nguồn trích dẫn", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<b>[{src["index"]}]</b> 📄 {src["source"]} '
                            f'— Trang {src["page"]}<br>'
                            f'<i>{src["content"][:200]}{"..." if len(src["content"]) > 200 else ""}</i>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# ------------------------------------------------------------------ #
#  Input câu hỏi                                                       #
# ------------------------------------------------------------------ #
question = st.chat_input(
    placeholder="Nhập câu hỏi về tài liệu... (Enter để gửi)",
    disabled=not st.session_state.index_ready,
)

if question:
    if not st.session_state.index_ready:
        st.warning("⚠️ Hãy upload tài liệu trước khi đặt câu hỏi.")
    else:
        # Lưu câu hỏi vào session state
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "sources": [],
        })

        # ── MỚI: xây dựng DocumentFilter từ filter_source (nếu có) ──
        doc_filter = None
        if st.session_state.filter_source:
            doc_filter = DocumentFilter(sources=[st.session_state.filter_source])

        # Gọi RAGChain — history được lưu tự động bên trong ask_with_sources()
        with st.spinner("🤔 Đang tìm kiếm và tạo câu trả lời..."):
            try:
                result = st.session_state.rag.ask_with_sources(
                    question,
                    filter=doc_filter,   # ── MỚI: truyền filter vào ──
                )
                answer  = result.get("answer", "Không có câu trả lời.")
                sources = result.get("sources", [])
                meta    = result.get("meta", {})
            except Exception as e:
                answer  = f"❌ Lỗi: {e}"
                sources = []
                meta    = {}

        # Lưu câu trả lời vào session state để hiển thị trên UI
        st.session_state.chat_history.append({
            "role": "ai",
            "content": answer,
            "sources": sources,
        })

        # Hiện latency nhỏ ở góc
        if meta.get("latency"):
            st.caption(f"⏱ {meta['latency']}s · model: {meta.get('model', 'unknown')}")

        st.rerun()