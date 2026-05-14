# Trang thai yeu cau phat trien project

Tai lieu nay tong hop doi chieu giua muc **8. YEU CAU PHAT TRIEN PROJECT** trong `gutenberg.pdf` va code hien tai cua project.

Ngay lap: 2026-05-14

## Tong quan nhanh

| STT | Yeu cau | Trang thai | Ghi chu ngan |
| --- | --- | --- | --- |
| 1 | Ho tro file DOCX | Hoan thanh co ban | Da upload/xu ly DOCX, giu thu tu paragraph/table, smoke test OK |
| 2 | Luu tru lich su hoi thoai | Hoan thanh co ban | Da luu JSON gom question + answer + sources/meta va sidebar xem duoc chi tiet |
| 3 | Nut xoa lich su/vector store | Hoan thanh co ban | Da co xac nhan truoc khi xoa va reset index xoa that FAISS tren disk |
| 4 | Cai thien chunk strategy | Gan hoan thanh | Da co UI/API tuy chinh chunk params va script tao bao cao so sanh |
| 5 | Citation/source tracking | Mot phan | Co source/page/content, thieu click xem context goc va highlight |
| 6 | Conversational RAG | Chua hoan thanh | Co history/prompt condense, nhung chua dung memory trong pipeline |
| 7 | Hybrid search | Mot phan/gan hoan thanh | Da co FAISS + BM25 + RRF, thieu benchmark voi vector search |
| 8 | Multi-document RAG + metadata filtering | Mot phan/gan hoan thanh | Da upload nhieu file va filter source, thieu filter UI theo type/date |
| 9 | Re-ranking voi Cross-Encoder | Chua co | Chua implement reranker |
| 10 | Self-RAG | Chua co | Chua co self-evaluation, query rewriting, confidence scoring |

## Chi tiet theo tung yeu cau

### 1. Them ho tro file DOCX

**Da co**

- UI Streamlit cho upload `pdf` va `docx`: `src/frontend/streamlit_app.py`.
- API upload chap nhan `.pdf`, `.docx`: `src/api/main.py`.
- Xu ly DOCX bang `python-docx`: `src/processor.py`.
- `python-docx` da co trong `requirements.txt`.
- DOCX extraction giu thu tu paragraph/table trong file thay vi gom rieng tung loai.
- Smoke test DOCX da pass: tao file co paragraph + table, load lai va kiem tra thu tu text.

**Con thieu / can kiem tra**

- Chua co pytest chinh thuc cho DOCX.
- Chua danh gia do chinh xac text extraction voi file DOCX phuc tap.

### 2. Luu tru lich su hoi thoai

**Da co**

- Luu history vao `data/chat_history.json`: `src/backend/history_store.py`.
- API xem history: `GET /history`, `GET /history/{entry_id}` trong `src/api/main.py`.
- Moi entry history luu `question`, `answer`, `sources`, `meta`, `timestamp`.
- Sidebar Streamlit hien thi preview cau tra loi va co expander xem day du cau hoi/cau tra loi/nguon.
- Smoke test history da pass: add/get history giu dung ca question va answer.

**Con thieu / can kiem tra**

- Chua tach lich su theo user/session.
- Trong UI hien tai, moi cau hoi chay ca RAG va CoRAG, nen history luu thanh 2 entry rieng.
- Chua co chuc nang click vao entry history de dua lai vao main chat, hien tai chi xem lai trong sidebar.

### 3. Them nut xoa lich su va vector store

**Da co**

- Nut `Xoa chat` trong sidebar Streamlit.
- Nut `Reset index` trong sidebar Streamlit.
- Co checkbox xac nhan truoc khi xoa chat/reset index.
- `Reset index` tren Streamlit goi `RAGChain.clear_index()` va xoa that `src/faiss_index`.
- Co tuy chon xoa ca file upload trong `data/uploads`.
- API xoa history: `DELETE /history`.
- API xoa tung document: `DELETE /documents/{filename}`.
- API xoa toan bo document/index: `DELETE /documents`.

**Con thieu / can sua**

- Co the cai tien UI thanh modal/dialog that neu muon dep hon, hien tai dang dung expander + checkbox xac nhan.

### 4. Cai thien chunk strategy

**Da co**

- `split_text(documents, chunk_size=1000, chunk_overlap=100)` trong `src/processor.py`.
- `process_multiple_documents(..., chunk_size, chunk_overlap)` da nhan tham so.
- CLI test trong `src/processor.py` co `--chunk-size` va `--chunk-overlap`.
- Streamlit cho chon `chunk_size` va `chunk_overlap` truoc khi upload.
- API `/upload` nhan `chunk_size` va `chunk_overlap` qua form-data.
- Metadata moi chunk co `chunk_size` va `chunk_overlap`.
- Co script `src/evaluate_chunk_strategy.py` tao bao cao Markdown cho cac cau hinh 500/1000/1500/2000 va 50/100/200.
- Streamlit sidebar co muc `Danh gia Chunk Strategy` de chay benchmark va tai bao cao Markdown ngay tren giao dien.

**Con thieu / can lam**

- Chua co benchmark accuracy tu dong voi ground-truth Q&A; hien tai report moi so sanh so chunks, do dai chunk va thoi gian xu ly.
- Neu doi chunk config cho file da index, can reset/xoa tai lieu va upload lai de re-index theo config moi.

### 5. Citation/source tracking

**Da co**

- RAG/CoRAG tra ve `sources` gom `index`, `source`, `page`, `content`.
- UI hien thi nguon trong expander cho RAG va CoRAG.
- Prompt yeu cau cite source/page khi co the.

**Con thieu / can lam**

- Chua co click de mo/xem context goc trong PDF.
- Chua co highlight doan van duoc dung de tra loi.
- Vi tri trong PDF moi co page, chua co toa do/section/offset.
- Citation trong answer phu thuoc LLM, chua co format bat buoc nhu `[1]`, `[2]`.

### 6. Implement Conversational RAG

**Da co**

- Co luu chat history.
- `PromptBuilder` co `get_condense_question_prompt()`.

**Con thieu / can lam**

- Pipeline RAG chua truyen history vao LLM.
- Chua rewrite follow-up question thanh standalone question truoc retrieval.
- Chua co memory object/chat history theo session.
- Chua co UI de hoi tiep dua tren ngu canh hoi thoai cu.

### 7. Them hybrid search

**Da co**

- `SemanticRetriever` dung FAISS.
- `KeywordRetriever` dung BM25.
- `HybridRetriever` ket hop semantic + keyword bang RRF.
- App/API dang build retriever hybrid.

**Con thieu / can lam**

- Chua co option bat/tat hybrid search trong UI.
- Chua co pure vector baseline de so sanh.
- Chua co benchmark performance/latency/relevance.
- Tham so `alpha`, `top_k` dang hardcoded.

### 8. Multi-document RAG voi metadata filtering

**Da co**

- UI upload nhieu file cung luc.
- API upload nhieu file cung luc.
- Metadata moi chunk co `source`, `file_type`, `date_uploaded`.
- Retriever co `match_filter()` ho tro source, file type, date range.
- UI co filter theo source/document.
- Cau tra loi hien thi document nguon trong sources.

**Con thieu / can lam**

- UI chua filter theo `file_type`.
- UI chua filter theo `date_uploaded`.
- Streamlit upload dung temp file, metadata ngay upload trong chunk co the la thoi diem xu ly temp, khong phai metadata quan ly file dai han.
- Can kiem tra dong bo source name giua API upload va Streamlit upload.

### 9. Implement Re-ranking voi Cross-Encoder

**Da co**

- Chua thay implement.
- `sentence-transformers` da co trong dependencies, co the dung de them CrossEncoder.

**Con thieu / can lam**

- Them buoc retrieve nhieu ung vien truoc, sau do rerank bang CrossEncoder.
- Them class/module reranker rieng.
- Tich hop vao RAG/CoRAG pipeline.
- So sanh voi bi-encoder/current approach.
- Toi uu latency, vi CrossEncoder co the cham.

### 10. Advanced RAG voi Self-RAG

**Da co**

- Co `CoRAGService`: decompose question -> retrieve nhieu lan -> synthesize.
- Co mot phan gan voi multi-hop reasoning qua sub-questions.

**Con thieu / can lam**

- Chua co Self-RAG dung nghia: LLM tu danh gia cau tra loi.
- Chua co query rewriting tu dong trong pipeline chinh.
- Chua co confidence scoring.
- Chua co reflection/evaluation step de quyet dinh retrieve lai hay tra loi.
- Chua co schema output cho confidence/evidence quality.

## Cac viec nen uu tien

1. Sua `Reset index` de xoa that FAISS index tren disk va them confirmation dialog.
2. Them UI/API cho `chunk_size` va `chunk_overlap`.
3. Implement Conversational RAG dung history de rewrite follow-up question.
4. Them reranking bang CrossEncoder.
5. Them Self-RAG/co che confidence scoring.
6. Viet test tu dong cho DOCX, history, filter, hybrid retrieval.

## File code lien quan

- `src/processor.py`: load PDF/DOCX, split chunk, metadata.
- `src/backend/retriever.py`: semantic, keyword, hybrid search, metadata filter.
- `src/backend/rag_service.py`: RAG answer, source tracking.
- `src/backend/corag_service.py`: CoRAG/decomposition.
- `src/backend/history_store.py`: luu/xoa/xem history.
- `src/frontend/streamlit_app.py`: UI upload, chat, filter, history.
- `src/api/main.py`: FastAPI upload, ask, history, documents.
- `app.py`: RAGChain orchestration cho Streamlit/CLI.
