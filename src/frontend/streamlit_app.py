# src/frontend/streamlit_app.py

import sys
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import RAGChain, DocumentFilter


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
#  CSS                                                                 #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #2E2C33; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stButton button {
    background-color: #444950 !important;
    color: #FFFFFF !important;
    border: 1px solid #666 !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #555C63 !important;
    border-color: #888 !important;
}
.user-bubble {
    background: #007BFF; color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0; max-width: 80%; margin-left: auto; word-wrap: break-word;
}
.ai-bubble {
    background: #F1F3F5; color: #212529;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0; max-width: 80%; word-wrap: break-word;
}
.source-box {
    background: #FFF8E1; border-left: 3px solid #FFC107;
    padding: 8px 12px; border-radius: 4px;
    font-size: 0.85em; margin-top: 6px; color: #212529;
}
.history-box {
    background: #3A3840; border-left: 3px solid #007BFF;
    padding: 6px 10px; border-radius: 4px;
    font-size: 0.82em; margin-bottom: 6px; color: #E0E0E0;
}
.file-tag {
    display: inline-block;
    background: #007BFF22; color: #007BFF;
    border: 1px solid #007BFF55;
    border-radius: 12px; padding: 2px 10px;
    font-size: 0.82em; margin: 2px 4px 2px 0;
}
.main-header { color: #007BFF; font-size: 2em; font-weight: 700; margin-bottom: 0; }
.sub-header  { color: #6C757D; font-size: 1em; margin-top: 0; margin-bottom: 24px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Session state                                                       #
# ------------------------------------------------------------------ #
def init_session():
    if "rag" not in st.session_state:
        st.session_state.rag = RAGChain()
        st.session_state.index_ready = st.session_state.rag.load_from_disk_and_build()

    if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
    if "index_ready"    not in st.session_state: st.session_state.index_ready    = False
    if "filter_enabled" not in st.session_state: st.session_state.filter_enabled = False
    if "filter_source"  not in st.session_state: st.session_state.filter_source  = None

init_session()


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## 📄 SmartDoc AI")
    st.markdown("---")

    # Hướng dẫn
    st.markdown("### 📖 Hướng dẫn")
    st.markdown("""
1. Upload file **PDF** hoặc **DOCX** (có thể nhiều file)
2. Chờ hệ thống xử lý tài liệu
3. Đặt câu hỏi về nội dung
4. Xem câu trả lời kèm nguồn trích dẫn
""")

    st.markdown("---")

    # Cấu hình
    st.markdown("### ⚙️ Cấu hình")
    st.markdown("**Model:** `qwen2.5:1.5b`")
    st.markdown("**Embedding:** `MPNet 768-dim`")
    st.markdown("**Retriever:** Hybrid (FAISS + BM25)")

    # Danh sách file đã nạp
    loaded = st.session_state.rag.get_loaded_files()
    if loaded:
        st.success(f"✅ Đã index **{len(loaded)}** file")
        for fname in loaded:
            st.markdown(f'<span class="file-tag">📄 {fname}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------------- #
    #  Bộ lọc tài liệu — FIX lỗi phải click 2 lần                     #
    #  Dùng key cố định + on_change callback thay vì đọc value trực tiếp#
    # ---------------------------------------------------------------- #
    st.markdown("### 🔍 Lọc tài liệu")

    if loaded and st.session_state.index_ready:
        # Selectbox chọn file để lọc — None = tìm toàn bộ
        options = ["🌐 Tất cả tài liệu"] + loaded
        current_idx = 0
        if st.session_state.filter_source in loaded:
            current_idx = loaded.index(st.session_state.filter_source) + 1

        selected = st.selectbox(
            "Tìm trong:",
            options=options,
            index=current_idx,
            key="filter_selectbox",
        )

        if selected == "🌐 Tất cả tài liệu":
            st.session_state.filter_source  = None
            st.session_state.filter_enabled = False
            st.caption("🌐 Đang tìm kiếm toàn bộ tài liệu")
        else:
            st.session_state.filter_source  = selected
            st.session_state.filter_enabled = True
            st.caption(f"🎯 Đang lọc: `{selected}`")
    else:
        st.caption("_Upload tài liệu để kích hoạt bộ lọc._")

    st.markdown("---")

    # Xoá dữ liệu
    st.markdown("### 🗑️ Xoá dữ liệu")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Xoá chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag.clear_history()
            st.rerun()
    with col2:
        if st.button("Reset index", use_container_width=True):
            st.session_state.rag.clear_history()
            st.session_state.rag          = RAGChain()
            st.session_state.index_ready  = False
            st.session_state.chat_history = []
            st.session_state.filter_source  = None
            st.session_state.filter_enabled = False
            st.rerun()

    st.markdown("---")

    # Lịch sử hội thoại
    st.markdown("### 💬 Lịch sử hội thoại")
    try:
        history_entries = st.session_state.rag.get_history()
    except Exception:
        history_entries = []

    if history_entries:
        for entry in reversed(history_entries[-10:]):
            q   = entry.get("question", "")
            q_p = q[:60] + ("..." if len(q) > 60 else "")
            ts  = entry.get("timestamp", "")[:16].replace("T", " ")
            lat = entry.get("meta", {}).get("latency", "")
            lat_str = f" · {lat}s" if lat else ""
            st.markdown(
                f'<div class="history-box">'
                f'<b>#{entry.get("id","?")}</b> {q_p}<br>'
                f'<span style="font-size:0.78em;color:#AAA;">{ts}{lat_str}</span>'
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
#  Upload file — hỗ trợ nhiều file, MERGE vào index                   #
# ------------------------------------------------------------------ #
st.markdown("### 📂 Upload tài liệu")

uploaded_files = st.file_uploader(
    label="Chọn file PDF hoặc DOCX (có thể chọn nhiều file)",
    type=["pdf", "docx"],
    accept_multiple_files=True,   # ← cho phép chọn nhiều file
    help="Tối đa 50MB/file. Hỗ trợ tiếng Việt và tiếng Anh.",
)

if uploaded_files:
    already_loaded = st.session_state.rag.get_loaded_files()
    new_files = [f for f in uploaded_files if f.name not in already_loaded]

    if new_files:
        for uf in new_files:
            with st.spinner(f"⏳ Đang xử lý **{uf.name}**..."):
                try:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.getbuffer())
                        tmp_path = tmp.name

                    # Truyền tên GỐC để metadata source = tên gốc
                    n = st.session_state.rag.add_document(tmp_path, uf.name)
                    st.session_state.index_ready = True
                    os.unlink(tmp_path)
                    st.success(f"✅ Đã thêm **{uf.name}** — {n} chunks")

                except Exception as e:
                    st.error(f"❌ Lỗi khi xử lý **{uf.name}**: {e}")
        st.rerun()
    else:
        loaded = st.session_state.rag.get_loaded_files()
        st.info(f"✅ Tất cả file đã được nạp ({len(loaded)} file trong index)")

st.markdown("---")

# ------------------------------------------------------------------ #
#  Chat                                                                #
# ------------------------------------------------------------------ #
st.markdown("### 💬 Hỏi đáp")

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

question = st.chat_input(
    placeholder="Nhập câu hỏi về tài liệu... (Enter để gửi)",
    disabled=not st.session_state.index_ready,
)

if question:
    st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})

    # Xây dựng filter từ session state
    doc_filter = None
    if st.session_state.filter_enabled and st.session_state.filter_source:
        doc_filter = DocumentFilter(sources=[st.session_state.filter_source])

    with st.spinner("🤔 Đang tìm kiếm và tạo câu trả lời..."):
        try:
            result  = st.session_state.rag.ask_with_sources(question, filter=doc_filter)
            answer  = result.get("answer", "Không có câu trả lời.")
            sources = result.get("sources", [])
            meta    = result.get("meta", {})
        except Exception as e:
            answer, sources, meta = f"❌ Lỗi: {e}", [], {}

    st.session_state.chat_history.append({"role": "ai", "content": answer, "sources": sources})

    if meta.get("latency"):
        st.caption(f"⏱ {meta['latency']}s · model: {meta.get('model','unknown')}")

    st.rerun()