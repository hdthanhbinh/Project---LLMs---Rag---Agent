# src/frontend/streamlit_app.py

import sys
import os
import time
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import RAGChain, DocumentFilter
from src.evaluate_chunk_strategy import (
    CHUNK_OVERLAPS,
    CHUNK_SIZES,
    build_report,
    summarize_chunks,
)


# ------------------------------------------------------------------ #
#  Cấu hình trang                                                      #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="SmartDoc AI — RAG vs CoRAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  CSS                                                                 #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #2E2C33;
}
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
    background-color: #555C63 !important; border-color: #888 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details summary {
    background-color: #3A3840 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary:focus,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary:active {
    background-color: #555C63 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary:hover *,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary:focus *,
[data-testid="stSidebar"] [data-testid="stExpander"] details summary:active * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background-color: #2E2C33 !important;
    color: #FFFFFF !important;
}

/* Khung RAG */
.rag-box {
    border: 2px solid #007BFF;
    border-radius: 12px;
    padding: 16px;
    background: #F0F7FF;
    color: #1a1a1a;
    min-height: 120px;
    margin-bottom: 8px;
}
/* Khung CoRAG */
.corag-box {
    border: 2px solid #28A745;
    border-radius: 12px;
    padding: 16px;
    background: #F0FFF4;
    color: #1a1a1a;
    min-height: 120px;
    margin-bottom: 8px;
}
/* Header của mỗi khung */
.rag-header {
    color: #007BFF; font-weight: 700; font-size: 1.05em; margin-bottom: 8px;
}
.corag-header {
    color: #28A745; font-weight: 700; font-size: 1.05em; margin-bottom: 8px;
}
/* Sub-question chip */
.subq-chip {
    display: inline-block;
    background: #D4EDDA; color: #155724;
    border: 1px solid #C3E6CB;
    border-radius: 10px; padding: 2px 10px;
    font-size: 0.8em; margin: 2px 4px 2px 0;
}
/* Source box */
.source-box {
    background: #FFF8E1; border-left: 3px solid #FFC107;
    padding: 6px 10px; border-radius: 4px;
    font-size: 0.82em; margin-top: 4px; color: #212529;
}
/* Metric badge */
.badge-rag   { background:#007BFF22; color:#007BFF; border:1px solid #007BFF55;
               border-radius:8px; padding:2px 10px; font-size:0.82em; }
.badge-corag { background:#28A74522; color:#28A745; border:1px solid #28A74555;
               border-radius:8px; padding:2px 10px; font-size:0.82em; }
/* Chat bubbles */
.user-bubble {
    background:#007BFF; color:white; padding:10px 16px;
    border-radius:18px 18px 4px 18px; margin:6px 0;
    max-width:80%; margin-left:auto; word-wrap:break-word;
}
/* History box */
.history-box {
    background:#3A3840; border-left:3px solid #007BFF;
    padding:6px 10px; border-radius:4px;
    font-size:0.82em; margin-bottom:6px; color:#E0E0E0;
}
.file-tag {
    display:inline-block; background:#007BFF22; color:#007BFF;
    border:1px solid #007BFF55; border-radius:12px;
    padding:2px 10px; font-size:0.82em; margin:2px 4px 2px 0;
}
.main-header { color:#007BFF; font-size:2em; font-weight:700; margin-bottom:0; }
.sub-header  { color:#6C757D; font-size:1em; margin-top:0; margin-bottom:16px; }
.chunk-summary {
    background: #213F34;
    border-left: 3px solid #28A745;
    border-radius: 6px;
    color: #FFFFFF;
    font-size: 0.82rem;
    margin-top: 8px;
    padding: 8px 10px;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    min-height: 72px !important;
    padding: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section > div {
    padding: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    min-height: 72px !important;
    padding: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div {
    padding: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    padding: 6px 8px !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    font-size: 0.72rem !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background-color: #444950 !important;
    color: #FFFFFF !important;
    border: 1px solid #666 !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    background-color: #555C63 !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button *,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] {
    border: 1px solid #555C63;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] div[role="toolbar"] {
    background: #2E2C33 !important;
    border-radius: 8px !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] div[role="toolbar"] button {
    background: #444950 !important;
    color: #FFFFFF !important;
    border: 1px solid #666 !important;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] div[role="toolbar"] button * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] [data-testid="StyledFullScreenButton"],
[data-testid="stSidebar"] [data-testid="stDataFrame"] [data-testid="StyledFullScreenButton"] button {
    background: #444950 !important;
    color: #FFFFFF !important;
    border: 1px solid #666 !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] svg {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Session state                                                       #
# ------------------------------------------------------------------ #
def init_session():
    if "rag" not in st.session_state:
        st.session_state.rag = RAGChain()
        st.session_state.index_ready = st.session_state.rag.load_from_disk_and_build()

    defaults = {
        "chat_history":   [],    # list of { question, rag, corag }
        "index_ready":    False,
        "filter_source":  None,
        "filter_enabled": False,
        "upload_widget_key": 0,
        "chunk_size": 1000,
        "chunk_overlap": 100,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


def run_chunk_strategy_evaluation(uploaded_eval_files) -> tuple[list[dict], str]:
    temp_paths = []
    display_names = []

    try:
        for uf in uploaded_eval_files:
            suffix = Path(uf.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getbuffer())
                temp_paths.append(tmp.name)
                display_names.append(uf.name)

        rows = []
        for chunk_size in CHUNK_SIZES:
            for chunk_overlap in CHUNK_OVERLAPS:
                if chunk_overlap < chunk_size:
                    rows.append(
                        summarize_chunks(temp_paths, chunk_size, chunk_overlap)
                    )

        report = build_report(rows, display_names)
        return rows, report
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def render_chunk_eval_table(rows: list[dict]) -> str:
    best = min(rows, key=lambda row: row.get("seconds", 0)) if rows else None
    body = []
    for row in rows:
        marker = "Nhanh nhất" if row is best else ""
        body.append(
            "<tr>"
            f"<td>{marker}</td>"
            f"<td>{row['chunk_size']}</td>"
            f"<td>{row['chunk_overlap']}</td>"
            f"<td>{row['chunk_count']}</td>"
            f"<td>{row['avg_chars']}</td>"
            f"<td>{row['seconds']}</td>"
            "</tr>"
        )

    return (
        '<div class="chunk-table-wrap">'
        '<table class="chunk-table">'
        "<thead><tr>"
        "<th>Ghi chú</th>"
        "<th>Size</th>"
        "<th>Overlap</th>"
        "<th>Chunks</th>"
        "<th>TB ký tự</th>"
        "<th>Giây</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


@st.dialog("Xác nhận xoá toàn bộ lịch sử")
def confirm_clear_history_dialog():
    st.write("Bạn có chắc muốn xoá toàn bộ lịch sử hỏi đáp không?")
    st.caption("Thao tác này sẽ xoá dữ liệu trong lịch sử chat đã lưu.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Xác nhận xoá", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag.clear_history()
            st.rerun()
    with c2:
        if st.button("Huỷ", use_container_width=True):
            st.rerun()


@st.dialog("Xác nhận xoá tài liệu đã upload")
def confirm_clear_documents_dialog():
    st.write("Bạn có chắc muốn xoá toàn bộ tài liệu đã upload không?")
    st.caption(
        "Thao tác này sẽ xoá vector store trên disk và các file trong data/uploads nếu có."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Xác nhận xoá", use_container_width=True):
            st.session_state.rag.clear_index(delete_uploaded_files=True)
            st.session_state.rag = RAGChain()
            st.session_state.index_ready = False
            st.session_state.filter_source = None
            st.session_state.filter_enabled = False
            st.session_state.upload_widget_key += 1
            st.rerun()
    with c2:
        if st.button("Huỷ", use_container_width=True):
            st.rerun()


# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## SmartDoc AI")
    st.markdown("---")

    st.markdown("### Hướng dẫn")
    st.markdown("""
1. Upload **PDF / DOCX** (nhiều file)
2. Đặt câu hỏi → hệ thống chạy **RAG** và **CoRAG** tuần tự
3. So sánh kết quả, thời gian xử lý
""")
    st.markdown("---")

    st.markdown("### Cấu hình")
    st.markdown("**Model:** `qwen2.5:1.7b`")
    st.markdown("**Embedding:** `MPNet 768-dim`")
    st.markdown("**Retriever:** Hybrid (FAISS + BM25)")

    st.session_state.chunk_size = st.selectbox(
        "Chunk size:",
        options=[500, 1000, 1500, 2000],
        index=[500, 1000, 1500, 2000].index(st.session_state.chunk_size),
        help="Số ký tự tối đa trong mỗi chunk khi upload tài liệu.",
    )
    valid_overlaps = [
        overlap for overlap in [50, 100, 200]
        if overlap < st.session_state.chunk_size
    ]
    if st.session_state.chunk_overlap not in valid_overlaps:
        st.session_state.chunk_overlap = valid_overlaps[0]
    st.session_state.chunk_overlap = st.selectbox(
        "Chunk overlap:",
        options=valid_overlaps,
        index=valid_overlaps.index(st.session_state.chunk_overlap),
        help="Số ký tự lặp lại giữa hai chunk liền kề.",
    )

    st.markdown("### Đánh giá Chunk Strategy")
    eval_files = st.file_uploader(
        "Chọn file PDF/DOCX để so sánh chunk",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="chunk_strategy_eval_files",
        help="Phần này chỉ tạo bảng so sánh, không upload vào index chính.",
    )
    if st.button(
        "Chạy đánh giá chunk",
        use_container_width=True,
        disabled=not eval_files,
    ):
        with st.spinner("Đang đánh giá các cấu hình chunk..."):
            try:
                rows, report = run_chunk_strategy_evaluation(eval_files)
                st.session_state.chunk_eval_rows = rows
                st.session_state.chunk_eval_report = report
            except Exception as exc:
                st.error(f"Không thể đánh giá chunk: {exc}")

    if st.session_state.get("chunk_eval_rows"):
        rows = st.session_state.chunk_eval_rows
        fastest = min(rows, key=lambda row: row.get("seconds", 0))
        fewest_chunks = min(rows, key=lambda row: row.get("chunk_count", 0))
        st.markdown(
            '<div class="chunk-summary">'
            f"Nhanh nhất: size {fastest['chunk_size']}, "
            f"overlap {fastest['chunk_overlap']} ({fastest['seconds']}s)<br>"
            f"Ít chunks nhất: size {fewest_chunks['chunk_size']}, "
            f"overlap {fewest_chunks['chunk_overlap']} "
            f"({fewest_chunks['chunk_count']} chunks)"
            "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
            height=300,
        )
        st.download_button(
            "Tải báo cáo Markdown",
            data=st.session_state.chunk_eval_report,
            file_name="chunk_strategy_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    loaded = st.session_state.rag.get_loaded_files()
    if loaded:
        st.success(f"Đã index **{len(loaded)}** file")
        for fname in loaded:
            st.markdown(f'<span class="file-tag">{fname}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Bộ lọc
    st.markdown("### Lọc tài liệu")
    if loaded and st.session_state.index_ready:
        options     = ["Tất cả tài liệu"] + loaded
        current_idx = 0
        if st.session_state.filter_source in loaded:
            current_idx = loaded.index(st.session_state.filter_source) + 1

        selected = st.selectbox("Tìm trong:", options=options, index=current_idx,
                                key="filter_selectbox")
        if selected == "Tất cả tài liệu":
            st.session_state.filter_source  = None
            st.session_state.filter_enabled = False
            st.caption("Tìm kiếm toàn bộ tài liệu")
        else:
            st.session_state.filter_source  = selected
            st.session_state.filter_enabled = True
            st.caption(f"Đang lọc: `{selected}`")
    else:
        st.caption("_Upload tài liệu để kích hoạt bộ lọc._")

    st.markdown("---")

    # Xoá
    st.markdown("### Xoá dữ liệu")
    if st.button("Xoá Toàn Bộ Lịch Sử", use_container_width=True):
        confirm_clear_history_dialog()
    if st.button("Xoá Tài Liệu Đã Upload", use_container_width=True):
        confirm_clear_documents_dialog()

    st.markdown("---")

    # Lịch sử
    st.markdown("### Lịch sử")
    try:
        hist = st.session_state.rag.get_history()
    except Exception:
        hist = []

    if hist:
        for entry in reversed(hist[-8:]):
            q   = entry.get("question", "")
            qp  = q[:55] + ("..." if len(q) > 55 else "")
            answer = entry.get("answer", "")
            answer_preview = answer[:90] + ("..." if len(answer) > 90 else "")
            ts  = entry.get("timestamp", "")[:16].replace("T", " ")
            lat = entry.get("meta", {}).get("latency", "")
            lats = f" · {lat}s" if lat else ""
            method = entry.get("meta", {}).get("method", "")
            badge = f'<span class="badge-corag">CoRAG</span>' if method == "corag" \
                    else f'<span class="badge-rag">RAG</span>'
            st.markdown(
                f'<div class="history-box">{badge} {qp}<br>'
                f'<span style="font-size:0.78em;color:#DDD;">{answer_preview}</span><br>'
                f'<span style="font-size:0.78em;color:#AAA;">{ts}{lats}</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Xem câu hỏi và câu trả lời", expanded=False):
                st.markdown("**Câu hỏi**")
                st.write(q)
                st.markdown("**Câu trả lời**")
                st.write(answer or "_Chưa có câu trả lời được lưu._")

                sources = entry.get("sources") or []
                if sources:
                    st.markdown("**Nguồn**")
                    for src in sources[:3]:
                        page = src.get("page", "N/A")
                        st.caption(f"[{src.get('index')}] {src.get('source')} - Trang {page}")
    else:
        st.caption("_Chưa có lịch sử._")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
st.markdown('<p class="main-header">SmartDoc AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">RAG vs CoRAG — So sánh tuần tự</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Upload                                                              #
# ------------------------------------------------------------------ #
st.markdown("### Upload tài liệu")

uploaded_files = st.file_uploader(
    "Chọn file PDF hoặc DOCX (có thể chọn nhiều file)",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key=f"document_uploader_{st.session_state.upload_widget_key}",
    help="Có thể chọn nhiều file.",
)

if uploaded_files:
    already = st.session_state.rag.get_loaded_files()
    new_files = [f for f in uploaded_files if f.name not in already]
    if new_files:
        for uf in new_files:
            with st.spinner(f"Đang xử lý **{uf.name}**..."):
                try:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.getbuffer())
                        tmp_path = tmp.name
                    n = st.session_state.rag.add_document(
                        tmp_path,
                        uf.name,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                    )
                    st.session_state.index_ready = True
                    os.unlink(tmp_path)
                    st.caption(
                        f"Chunk size: {st.session_state.chunk_size}; "
                        f"overlap: {st.session_state.chunk_overlap}"
                    )
                    st.success(f"**{uf.name}** — {n} chunks")
                except Exception as e:
                    st.error(f"{uf.name}: {e}")
        st.rerun()
    else:
        st.info(f"Tất cả file đã được nạp ({len(already)} file)")

st.markdown("---")

# ------------------------------------------------------------------ #
#  Hiển thị lịch sử chat (2 cột)                                      #
# ------------------------------------------------------------------ #
st.markdown("### Hỏi đáp — RAG vs CoRAG")

for turn in st.session_state.chat_history:
    # Câu hỏi
    st.markdown(
        f'<div class="user-bubble">{turn["question"]}</div>',
        unsafe_allow_html=True,
    )

    col_rag, col_corag = st.columns(2)

    # ── Cột RAG ──
    with col_rag:
        r = turn.get("rag", {})
        lat = r.get("meta", {}).get("latency", "—")
        st.markdown(
            f'<div class="rag-box">'
            f'<div class="rag-header">RAG &nbsp;'
            f'<span style="font-weight:400;font-size:0.85em;">({lat}s)</span></div>'
            f'{r.get("answer","—")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if r.get("sources"):
            with st.expander("📎 Nguồn (RAG)", expanded=False):
                for src in r["sources"]:
                    st.markdown(
                        f'<div class="source-box"><b>[{src["index"]}]</b> '
                        f'{src["source"]} — Trang {src["page"]}<br>'
                        f'<i>{src["content"][:180]}{"..." if len(src["content"])>180 else ""}</i>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Cột CoRAG ──
    with col_corag:
        c = turn.get("corag", {})
        lat2 = c.get("meta", {}).get("latency", "—")
        sub_qs = c.get("sub_questions", [])

        # Hiển thị sub-questions
        sub_html = ""
        if sub_qs:
            chips = "".join(f'<span class="subq-chip">{q}</span>' for q in sub_qs)
            sub_html = f'<div style="margin-bottom:8px;">{chips}</div>'

        st.markdown(
            f'<div class="corag-box">'
            f'<div class="corag-header">CoRAG &nbsp;'
            f'<span style="font-weight:400;font-size:0.85em;">({lat2}s)</span></div>'
            f'{sub_html}'
            f'{c.get("answer","—")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if c.get("sources"):
            with st.expander("📎 Nguồn (CoRAG)", expanded=False):
                for src in c["sources"]:
                    st.markdown(
                        f'<div class="source-box"><b>[{src["index"]}]</b> '
                        f'{src["source"]} — Trang {src["page"]}<br>'
                        f'<i>{src["content"][:180]}{"..." if len(src["content"])>180 else ""}</i>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # So sánh latency nhanh
    try:
        lat_rag   = float(turn["rag"]["meta"].get("latency", 0))
        lat_corag = float(turn["corag"]["meta"].get("latency", 0))
        faster = "RAG" if lat_rag <= lat_corag else "CoRAG"
        diff   = abs(lat_rag - lat_corag)
        st.caption(f"⏱ RAG: {lat_rag}s &nbsp;|&nbsp; CoRAG: {lat_corag}s &nbsp;→&nbsp; **{faster}** nhanh hơn {diff:.2f}s")
    except Exception:
        pass

    st.markdown("---")


# ------------------------------------------------------------------ #
#  Input câu hỏi                                                       #
# ------------------------------------------------------------------ #
question = st.chat_input(
    placeholder="Nhập câu hỏi... (RAG và CoRAG sẽ chạy tuần tự)",
    disabled=not st.session_state.index_ready,
)

if question:
    # Build filter
    doc_filter = None
    if st.session_state.filter_enabled and st.session_state.filter_source:
        doc_filter = DocumentFilter(sources=[st.session_state.filter_source])

    rag_chain = st.session_state.rag

    # Chạy TUẦN TỰ: RAG trước, CoRAG sau — đo thời gian chính xác từng bước
    with st.spinner("Đang chạy RAG..."):
        result_rag = rag_chain.ask_rag(question, doc_filter, save_history=True)
 
    with st.spinner("Đang chạy CoRAG (decompose → retrieve → synthesize)..."):
        result_corag = rag_chain.ask_corag(question, doc_filter, save_history=True)
 
    st.session_state.chat_history.append({
        "question": question,
        "rag":      result_rag,
        "corag":    result_corag,
    })

    st.rerun()
