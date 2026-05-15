# src/backend/retriever.py
from langchain_community.retrievers import BM25Retriever
from datetime import datetime
from src.backend.reranker import get_reranker

# ------------------------------------------------------------------ #
#  Date helpers                                                        #
# ------------------------------------------------------------------ #

def _parse_iso_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.split(".")[0])
        return datetime.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return None


def match_filter(doc, filter) -> bool:
    """Return True if doc's metadata satisfies every active filter field."""
    if filter is None:
        return True

    meta = doc.metadata or {}
    print(
        f"DEBUG match_filter | source='{meta.get('source')}' | "
        f"file_type='{meta.get('file_type')}' | "
        f"date_uploaded='{meta.get('date_uploaded')}' | "
        f"filter.sources={filter.sources} | filter.file_type={filter.file_type}"
    )

    if filter.sources and meta.get("source") not in filter.sources:
        return False

    if filter.file_type:
        stored = meta.get("file_type", "").lstrip(".")
        wanted = filter.file_type.lstrip(".")
        if stored != wanted:
            return False

    if filter.date_from:
        doc_date = _parse_iso_date(meta.get("date_uploaded", ""))
        filter_date = _parse_iso_date(filter.date_from)
        if doc_date and filter_date and doc_date < filter_date:
            return False

    if filter.date_to:
        doc_date = _parse_iso_date(meta.get("date_uploaded", ""))
        filter_date = _parse_iso_date(filter.date_to)
        if doc_date and filter_date and doc_date > filter_date:
            return False

    return True


# ------------------------------------------------------------------ #
#  Retrievers                                                          #
# ------------------------------------------------------------------ #

class SemanticRetriever:
    def __init__(self, vector_store, k: int = 4):
        self.k = k
        self.vector_store = vector_store

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": self.k})

    def retrieve(self, query: str) -> list:
        return self.vector_store.similarity_search(query, k=self.k)


class KeywordRetriever:
    def __init__(self, documents, k: int = 3):
        self.k = k
        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = k

    def get_retriever(self):
        return self.retriever

    def retrieve(self, query: str) -> list:
        return self.retriever.invoke(query)


class HybridRetriever:
    """
    Reciprocal Rank Fusion of semantic + BM25 results,
    with optional CrossEncoder re-ranking.

    When enable_rerank=True:
      - A wider candidate pool is fed to the cross-encoder
        (CANDIDATE_MULTIPLIER × top_k, capped at CANDIDATE_CAP)
      - The cross-encoder re-scores and selects the final top_k

    After every retrieve() call the following instance attributes are set
    so that upstream services can include them in response metadata:

        self.last_rerank_info : dict  (keys defined below)

    rerank_info keys:
        rerank_enabled    bool
        rerank_model      str
        rerank_latency    float  (seconds, 0.0 when disabled)
        candidate_count   int    (docs fed to cross-encoder)
    """

    RRF_K = 60                 # standard RRF constant
    CANDIDATE_MULTIPLIER = 4   # candidate_pool = top_k * CANDIDATE_MULTIPLIER
    CANDIDATE_CAP = 20         # never feed more than this to the cross-encoder

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        keyword_retriever: KeywordRetriever,
        alpha: float = 0.5,
        top_k: int = 4,
        enable_rerank: bool = False,
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha
        self.top_k = top_k
        self.enable_rerank = enable_rerank
        self.reranker = get_reranker() if enable_rerank else None

        # Set after every retrieve() call
        self.last_rerank_info: dict = {
            "rerank_enabled": False,
            "rerank_model": None,
            "rerank_latency": 0.0,
            "candidate_count": 0,
        }

    def _make_key(self, doc) -> str:
        source = doc.metadata.get("source", "")
        page = str(doc.metadata.get("page", ""))
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        content_sig = (doc.page_content or "")[:120]
        key = "|".join([source, page, chunk_id, content_sig])
        return key if key.strip("|") else content_sig

    def retrieve(self, query: str, filter=None) -> list:
        # ── Step 1: fetch raw candidates ──────────────────────────────
        # Always fetch a wide semantic pool so the reranker has good input.
        semantic_docs = self.semantic_retriever.vector_store \
            .similarity_search_with_score(query, k=30)
        keyword_docs = self.keyword_retriever.retriever.invoke(query)

        # ── Step 2: RRF scoring with filter ───────────────────────────
        scores: dict[str, float] = {}
        doc_map: dict[str, object] = {}

        for rank, (doc, _) in enumerate(semantic_docs):
            if not match_filter(doc, filter):
                continue
            key = self._make_key(doc)
            scores[key] = scores.get(key, 0.0) + self.alpha * (1 / (rank + self.RRF_K))
            doc_map[key] = doc

        for rank, doc in enumerate(keyword_docs):
            if not match_filter(doc, filter):
                continue
            key = self._make_key(doc)
            scores[key] = scores.get(key, 0.0) + (1 - self.alpha) * (1 / (rank + self.RRF_K))
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

        # ── Step 3: optional re-ranking ───────────────────────────────
        if self.enable_rerank and self.reranker:
            candidate_k = min(
                len(sorted_keys),
                max(self.top_k * self.CANDIDATE_MULTIPLIER, self.CANDIDATE_CAP),
            )
            candidates = [doc_map[k] for k in sorted_keys[:candidate_k]]
            reranked, rerank_info = self.reranker.rerank(
                query, candidates, top_k=self.top_k
            )
            self.last_rerank_info = rerank_info
            return reranked

        # ── No rerank: return top_k by RRF score ──────────────────────
        self.last_rerank_info = {
            "rerank_enabled": False,
            "rerank_model": None,
            "rerank_latency": 0.0,
            "candidate_count": min(len(sorted_keys), self.top_k),
        }
        return [doc_map[k] for k in sorted_keys[:self.top_k]]