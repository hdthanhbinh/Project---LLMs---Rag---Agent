# Prompts cho GitHub Copilot hoan thanh yeu cau project

File nay gom cac prompt co the copy vao GitHub Copilot Chat. Nen chay tung prompt mot, review diff sau moi lan, roi test lai truoc khi lam prompt tiep theo.

## Prompt 1: Sua Reset Index va them confirmation dialog

```text
Ban dang lam trong project Python/Streamlit/FastAPI RAG.

Hay doc cac file:
- src/frontend/streamlit_app.py
- app.py
- src/backend/vector_store.py
- src/api/main.py

Muc tieu:
Hoan thanh yeu cau "Clear History" va "Clear Vector Store" trong muc 8.2.3.

Can lam:
1. Trong Streamlit sidebar, them confirmation dialog hoac co che xac nhan truoc khi xoa chat history.
2. Them confirmation dialog hoac co che xac nhan truoc khi reset/xoa vector store.
3. Khi reset vector store, phai xoa that index FAISS tren disk tai src/faiss_index, khong chi reset session_state.
4. Neu co file upload dai han trong data/uploads thi them tuy chon xoa chung khi reset index, hoac ghi ro reset index chi xoa index.
5. Dam bao sau khi reset, app khong load lai index cu khi refresh.
6. Giu code theo style hien tai, khong refactor lon.

Sau khi sua, hay tom tat:
- File nao da sua
- Chuc nang moi hoat dong nhu the nao
- Cach test thu cong tren Streamlit
```

## Prompt 2: Them UI/API tuy chinh chunk_size va chunk_overlap

```text
Ban dang lam trong project RAG dung Streamlit + FastAPI + LangChain.

Hay doc cac file:
- src/processor.py
- app.py
- src/frontend/streamlit_app.py
- src/api/main.py

Muc tieu:
Hoan thanh yeu cau 8.2.4 "Cai thien chunk strategy".

Can lam:
1. Cho phep user tuy chinh chunk_size va chunk_overlap tren Streamlit khi upload tai lieu.
2. Gia tri goi y:
   - chunk_size: 500, 1000, 1500, 2000
   - chunk_overlap: 50, 100, 200
3. Truyen chunk_size/chunk_overlap tu UI vao RAGChain.add_document().
4. Sua app.py de add_document(file_path, original_name, chunk_size=1000, chunk_overlap=100) va goi split_text dung tham so.
5. Trong FastAPI /upload, cho phep nhan chunk_size va chunk_overlap qua form parameters voi default 1000/100.
6. Dam bao metadata hoac response upload co luu/tra ve chunk settings da dung.
7. Khong lam hong DOCX/PDF upload hien tai.

Sau khi sua, hay them huong dan test:
- Upload cung mot file voi cac chunk size khac nhau
- Kiem tra so chunks thay doi
- Kiem tra hoi dap van hoat dong
```

## Prompt 3: Tao bao cao so sanh chunk strategy

```text
Hay tao mot script nho de danh gia chunk strategy cho project nay.

Doc cac file:
- src/processor.py
- app.py
- src/backend/retriever.py

Muc tieu:
Hoan thanh phan "thu nghiem chunk_size/chunk_overlap va bao cao ket qua" cua yeu cau 8.2.4.

Can lam:
1. Tao script moi trong `src/`, vi du `src/evaluate_chunk_strategy.py`.
2. Script nhan input la duong dan file PDF/DOCX va mot danh sach cau hoi mau.
3. Thu cac cau hinh:
   - chunk_size: 500, 1000, 1500, 2000
   - chunk_overlap: 50, 100, 200
4. Voi moi cau hinh, load document, split chunk, build FAISS, retrieve cho tung cau hoi.
5. Ghi lai metrics co ban:
   - so chunks
   - thoi gian build index
   - thoi gian retrieve trung binh
   - top sources/pages retrieve duoc
6. Xuat ket qua ra Markdown hoac CSV trong documentation/.
7. Khong can goi LLM neu muon tranh cham, chi can retrieval benchmark la du.

Hay giu script de chay duoc tu terminal va co README ngan trong comment dau file.
```

## Prompt 4: Implement Conversational RAG

```text
Ban dang lam project RAG. Hay implement Conversational RAG that su.

Doc cac file:
- app.py
- src/backend/rag_service.py
- src/backend/prompt_builder.py
- src/backend/history_store.py
- src/frontend/streamlit_app.py
- src/api/main.py

Hien tai project co history va PromptBuilder.get_condense_question_prompt(), nhung pipeline chua dung history de xu ly follow-up questions.

Muc tieu:
Hoan thanh yeu cau 8.2.6 "Implement Conversational RAG".

Can lam:
1. Them co che rewrite/condense question:
   - Neu co chat history, LLM rewrite cau hoi moi thanh standalone question.
   - Dung get_condense_question_prompt() hoac tao prompt tuong duong.
2. RAG retrieval phai dung standalone question, khong dung truc tiep follow-up question ngan.
3. Answer van tra loi theo cau hoi goc cua user.
4. Luu vao meta:
   - original_question
   - standalone_question
   - used_conversation_history: true/false
5. Streamlit chat_history nen duoc truyen vao ask_rag/ask_corag hoac RAGChain.
6. API /ask co the nhan optional history hoac dung history luu tru hien co, nhung can thiet ke ro va don gian.
7. Them fallback: neu rewrite fail thi dung cau hoi goc.

Sau khi sua, hay dua example test:
- Hoi: "Tai lieu nay noi ve gi?"
- Hoi tiep: "No co uu diem nao?"
- He thong phai rewrite cau sau dua tren cau truoc.
```

## Prompt 5: Cai thien citation/source tracking

```text
Hay cai thien citation/source tracking cho project RAG nay.

Doc cac file:
- src/backend/rag_service.py
- src/backend/corag_service.py
- src/backend/prompt_builder.py
- src/frontend/streamlit_app.py
- src/processor.py

Muc tieu:
Hoan thanh them yeu cau 8.2.5.

Can lam:
1. Bat buoc format citation trong cau tra loi theo dang [1], [2] dua tren sources tra ve.
2. Context dua vao prompt phai gan nhan source index ro rang, vi du [1] source=... page=...
3. UI Streamlit hien thi sources tuong ung voi citation [1], [2].
4. Cho phep user xem context goc bang expander/click trong UI.
5. Highlight hoac lam noi bat text source/context duoc dung. Neu khong co exact span, highlight toan bo chunk source.
6. Dam bao RAG va CoRAG deu co citation nhat quan.
7. Neu page metadata khong co voi DOCX thi hien thi "N/A" thay vi page 0.

Khong can mo file PDF truc tiep trong browser neu phuc tap; truoc mat chi can click/expander xem context goc va highlight chunk la du.
```

## Prompt 6: Them pure vector baseline va benchmark Hybrid Search

```text
Hay hoan thien yeu cau 8.2.7 "Hybrid search".

Doc cac file:
- src/backend/retriever.py
- app.py
- src/frontend/streamlit_app.py
- src/api/main.py

Hien tai da co SemanticRetriever, KeywordRetriever, HybridRetriever. Can them phan so sanh voi pure vector search.

Can lam:
1. Them tuy chon search_mode:
   - "semantic"
   - "keyword"
   - "hybrid"
2. Streamlit sidebar co selectbox chon search mode.
3. RAGChain build service theo search_mode hoac retriever wrapper co the chon mode.
4. API /ask co optional search_mode.
5. Meta response can co:
   - search_mode
   - retrieval_latency
   - top_k
6. Them script benchmark nho trong scripts/evaluate_retrieval_modes.py de so sanh semantic vs hybrid tren mot danh sach cau hoi.
7. Khong pha vo default hien tai: default van la hybrid.

Sau khi sua, hay ghi cach test tung mode va cach doc metrics.
```

## Prompt 7: Hoan thien Multi-document metadata filtering

```text
Hay hoan thien yeu cau 8.2.8 "Multi-document RAG voi metadata filtering".

Doc cac file:
- src/processor.py
- src/backend/retriever.py
- app.py
- src/frontend/streamlit_app.py
- src/api/main.py

Hien tai metadata co source, file_type, date_uploaded va retriever co match_filter(), nhung UI moi filter theo source.

Can lam:
1. Streamlit sidebar them filter:
   - source/document
   - file_type: all/pdf/docx
   - date_from/date_to
2. Tao DocumentFilter tu cac input tren va truyen vao ask_rag/ask_corag.
3. Dam bao match_filter so sanh date_uploaded an toan, tot nhat parse ISO datetime thay vi so sanh string thuan neu can.
4. UI sources phai hien thi document nao da duoc dung.
5. API /documents nen tra ve metadata dong bo voi chunks/source filter.
6. Neu upload qua Streamlit, metadata source nen la original filename va file_type/date_uploaded phai duoc giu on dinh.

Sau khi sua, hay test:
- Upload 1 PDF va 1 DOCX
- Filter chi PDF
- Filter chi DOCX
- Filter theo date range
```

## Prompt 8: Implement Re-ranking voi Cross-Encoder

```text
Hay implement re-ranking voi Cross-Encoder cho project RAG.

Doc cac file:
- src/backend/retriever.py
- src/backend/rag_service.py
- src/backend/corag_service.py
- app.py
- requirements.txt

Muc tieu:
Hoan thanh yeu cau 8.2.9.

Can lam:
1. Tao module moi, vi du src/backend/reranker.py.
2. Dung sentence_transformers.CrossEncoder voi model nhe, vi du:
   - cross-encoder/ms-marco-MiniLM-L-6-v2
   hoac model phu hop voi latency CPU.
3. Retrieval flow:
   - HybridRetriever lay nhieu ung vien hon, vi du top 10-20.
   - CrossEncoder score lai pairs (query, doc.page_content).
   - Chon top_k cuoi cung.
4. Them option enable_rerank true/false.
5. Meta response can co:
   - rerank_enabled
   - rerank_model
   - rerank_latency
   - candidate_count
6. Streamlit sidebar them toggle "Re-ranking".
7. API /ask co optional enable_rerank.
8. Co fallback neu model load fail: log warning va dung retrieval cu.
9. Toi uu latency: lazy load CrossEncoder, khong load moi request.

Sau khi sua, hay them cach test va so sanh voi khi tat rerank.
```

## Prompt 9: Implement Self-RAG co query rewriting va confidence scoring

```text
Hay implement Self-RAG o muc vua phai cho project nay.

Doc cac file:
- src/backend/rag_service.py
- src/backend/corag_service.py
- src/backend/prompt_builder.py
- src/backend/retriever.py
- app.py
- src/frontend/streamlit_app.py

Muc tieu:
Hoan thanh yeu cau 8.2.10 "Advanced RAG voi Self-RAG".

Can lam:
1. Tao service moi, vi du src/backend/self_rag_service.py.
2. Flow de xuat:
   - Rewrite query neu cau hoi mo hoac follow-up.
   - Retrieve documents.
   - Generate draft answer.
   - LLM tu danh gia:
     a. context co du khong
     b. answer co duoc support boi context khong
     c. confidence score 0-1
   - Neu confidence thap, thu retrieve lai voi rewritten query hoac noi khong du thong tin.
3. Output schema:
   - question
   - rewritten_question
   - answer
   - sources
   - confidence
   - self_eval
   - meta
4. Them RAGChain.ask_self_rag().
5. Streamlit co option chon mode: RAG / CoRAG / Self-RAG, hoac hien them cot Self-RAG neu hop ly.
6. API /ask co optional mode hoac endpoint /ask/self-rag.
7. Giu fallback an toan: neu self-eval loi thi tra answer RAG binh thuong voi confidence null.

Khong can lam qua phuc tap. Uu tien code ro rang, co meta de debug, va khong pha RAG/CoRAG hien tai.
```

## Prompt 10: Them test tu dong cho cac yeu cau chinh

```text
Hay them test tu dong cho project Python nay.

Doc cac file:
- src/processor.py
- src/backend/history_store.py
- src/backend/retriever.py
- app.py
- src/api/main.py
- requirements.txt

Muc tieu:
Tang diem Testing trong muc 8.3.

Can lam:
1. Them pytest vao requirements neu chua co.
2. Tao thu muc tests/.
3. Viet test cho:
   - load DOCX co paragraph va table
   - split_text giu metadata source/file_type/date_uploaded
   - history_store add/get/clear, dung temp file hoac monkeypatch HISTORY_FILE
   - match_filter source/file_type/date range
   - HybridRetriever dedup/key behavior o muc unit neu co the mock docs
4. Neu FastAPI test de lam, them test cho /health va history endpoints bang TestClient.
5. Khong goi Ollama/LLM that trong unit tests.
6. Khong build embedding model nang trong unit tests neu khong can.

Sau khi sua, hay chay pytest va sua loi neu co.
```

## Prompt 11: Don dep README va tai lieu danh gia

```text
Hay cap nhat documentation cho project.

Doc:
- PROJECT_REQUIREMENTS_STATUS.md
- README.md
- documentation/README.md neu co
- src/frontend/streamlit_app.py
- src/api/main.py

Muc tieu:
Hoan thanh phan Documentation trong muc 8.3.

Can lam:
1. Cap nhat README.md voi:
   - Mo ta project
   - Tinh nang da co
   - Cach cai dat
   - Cach chay Streamlit
   - Cach chay FastAPI
   - Cach upload PDF/DOCX
   - Cach dung filter/search mode/rerank neu da implement
2. Them bang mapping yeu cau 8.2 -> tinh nang/code.
3. Them muc Known limitations.
4. Them muc Testing neu co pytest.
5. Dam bao khong viet sai cac feature chua implement.

Giu README ngan gon, ro rang, de giao vien/cham bai doc nhanh.
```

## Prompt tong hop neu muon Copilot lap ke hoach truoc

```text
Hay doc file PROJECT_REQUIREMENTS_STATUS.md va toan bo project. Lap ke hoach implement de hoan thanh cac yeu cau con thieu trong muc 8.2 cua bao cao.

Khong code ngay. Hay tra ve:
1. Thu tu uu tien nen lam
2. File nao can sua cho moi yeu cau
3. Rui ro/cham diem can luu y
4. Cach test sau moi buoc

Sau do dung ke hoach nay de implement tung buoc khi toi yeu cau.
```
