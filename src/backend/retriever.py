# backend/retriever.py
from langchain_community.retrievers import BM25Retriever

def match_filter(doc, filter) -> bool:
    """Kiểm tra doc có thỏa metadata filter không."""
    if filter is None:
        return True

    meta = doc.metadata or {}
    print(f"DEBUG match_filter | source='{meta.get('source')}' | file_type='{meta.get('file_type')}' | filter.sources={filter.sources}")

    if filter.sources:
        if meta.get("source") not in filter.sources:
            return False

    if filter.file_type:
        stored = meta.get("file_type", "").lstrip(".")
        wanted = filter.file_type.lstrip(".")
        if stored != wanted:
            return False

    if filter.date_from:
        if meta.get("date_uploaded", "") < filter.date_from:
            return False

    if filter.date_to:
        if meta.get("date_uploaded", "") > filter.date_to:
            return False

    return True
class SemanticRetriever:
    def __init__(self,vector_store,  k: int = 4):
        self.k = k
        self.vector_store = vector_store

        # semantic retriever
    def get_retriever(self):
            return self.vector_store.as_retriever(search_kwargs={"k": self.k})
    def retrieve(self, query: str) -> list:
            return self.vector_store.similarity_search(query, k=self.k)

class KeywordRetriever:
    def __init__(self,documents, k: int = 3):
        self.k = k
        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = k
    #keyword retriever
    def get_retriever(self):
        return self.retriever

    def retrieve(self, query: str) -> list :
        return self.retriever.invoke(query)

class HybridRetriever:
    RRF_K = 60  # hằng số chuẩn của RRF

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        keyword_retriever: KeywordRetriever,
        alpha: float = 0.5,
        top_k: int = 4,
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha
        self.top_k = top_k

    def _make_key(self, doc) -> str:
        source = doc.metadata.get("source", "")
        page = str(doc.metadata.get("page", ""))
        key = source + page
        return key if key.strip() else doc.page_content[:100]

    def retrieve(self, query: str, filter=None) -> list:
        semantic_docs = self.semantic_retriever.vector_store\
            .similarity_search_with_score(query, k=30)
        keyword_docs = self.keyword_retriever.retriever.invoke(query)

        scores: dict[str, float] = {}
        doc_map: dict[str, object] = {}

        for rank, (doc, _) in enumerate(semantic_docs):
            if not match_filter(doc, filter):   # ✅ filter ở đây
                continue
            key = self._make_key(doc)
            scores[key] = scores.get(key, 0.0) + self.alpha * (1 / (rank + self.RRF_K))
            doc_map[key] = doc

        for rank, doc in enumerate(keyword_docs):
            if not match_filter(doc, filter):   # ✅ filter ở đây
                continue
            key = self._make_key(doc)
            scores[key] = scores.get(key, 0.0) + (1 - self.alpha) * (1 / (rank + self.RRF_K))
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.top_k]]