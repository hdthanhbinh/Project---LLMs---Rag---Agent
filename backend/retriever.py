# backend/retriever.py
from langchain_community.retrievers import BM25Retriever
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
    def __init__(self, semantic_retriever: SemanticRetriever, keyword_retriever: KeywordRetriever, alpha: float = 0.5):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha
    def retrieve(self, query: str):
        semantic_docs = self.semantic_retriever.vector_store.similarity_search_with_score(query, k=10)
        keyword_docs = self.keyword_retriever.retriever.invoke(query)

        scores = {}   # key: str identifier
        doc_map = {}  # key: str identifier → Document object

        # Semantic score
        for doc, score in semantic_docs:
            # Dùng source + page nếu có, fallback về page_content
            key = doc.metadata.get("source", "") + str(doc.metadata.get("page", "")) \
                or doc.page_content[:100]
            
            scores[key] = scores.get(key, 0) + self.alpha * (1 - score)
            doc_map[key] = doc  # lưu lại object gốc

        # Keyword score
        for rank, doc in enumerate(keyword_docs):
            key = doc.metadata.get("source", "") + str(doc.metadata.get("page", "")) \
                or doc.page_content[:100]
            
            scores[key] = scores.get(key, 0) + (1 - self.alpha) * (1 / (rank + 1))
            doc_map[key] = doc

        # Sort và trả về Document objects
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

        return [doc_map[k] for k in sorted_keys[:self.semantic_retriever.k]]