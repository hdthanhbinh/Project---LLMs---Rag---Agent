# src/backend/reranker.py
"""
Cross-Encoder based re-ranking module for RAG.

Flow:
  1. HybridRetriever returns top N candidates  (N >> top_k, e.g. 15-20)
  2. CrossEncoder scores each (query, doc) pair
  3. Select top_k based on re-rank scores
  4. Return (reranked_docs, rerank_info) so callers can add meta

Lazy-load pattern: model is only loaded on the first rerank() call,
not on import or class construction.
"""

import time
from typing import Optional
import torch


class CrossEncoderReranker:
    """Lazy-loaded Cross-Encoder for document re-ranking."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None       # populated on first use
        self.load_error: str | None = None

    # ------------------------------------------------------------------ #
    #  Internal: lazy model load                                           #
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> bool:
        """Load CrossEncoder on first call.  Returns True on success."""
        if self.model is not None:
            return True
        if self.load_error:
            return False

        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            print(f"[Reranker] Loading CrossEncoder: {self.model_name} on {self.device}")
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
            print(f"[Reranker] Loaded successfully.")
            return True
        except Exception as exc:
            self.load_error = str(exc)
            print(f"[Reranker] ERROR: could not load model — {exc}")
            return False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        docs: list,
        top_k: int = 4,
        batch_size: int = 32,
    ) -> tuple[list, dict]:
        """
        Re-rank `docs` for `query` using the cross-encoder.

        Returns:
            (reranked_docs, rerank_info)

            rerank_info keys:
              rerank_enabled    bool
              rerank_model      str
              rerank_latency    float   seconds
              candidate_count   int     docs fed to cross-encoder
        """
        base_info = {
            "rerank_enabled": False,
            "rerank_model": self.model_name,
            "rerank_latency": 0.0,
            "candidate_count": len(docs),
        }

        if not docs:
            return [], base_info

        if not self._ensure_loaded():
            print("[Reranker] Fallback: returning original order (model unavailable).")
            return docs[:top_k], base_info

        t0 = time.time()
        try:
            pairs = [[query, doc.page_content] for doc in docs]
            with torch.no_grad():
                scores = self.model.predict(pairs, batch_size=batch_size)

            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            reranked = [doc for doc, _ in scored[:top_k]]

            latency = round(time.time() - t0, 3)
            print(
                f"[Reranker] {len(docs)} candidates → {len(reranked)} results "
                f"in {latency:.3f}s"
            )
            return reranked, {
                "rerank_enabled": True,
                "rerank_model": self.model_name,
                "rerank_latency": latency,
                "candidate_count": len(docs),
            }

        except Exception as exc:
            print(f"[Reranker] ERROR during scoring: {exc}")
            return docs[:top_k], base_info

    def get_model_info(self) -> dict:
        return {
            "enabled": self.model is not None or self.load_error is None,
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.model is not None,
            "error": self.load_error,
        }


# ------------------------------------------------------------------ #
#  Module-level singleton                                              #
# ------------------------------------------------------------------ #
_reranker_instance: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: str = CrossEncoderReranker.DEFAULT_MODEL,
) -> CrossEncoderReranker:
    """Return (or create) the module-level reranker singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker_instance = CrossEncoderReranker(model_name=model_name, device=device)
    return _reranker_instance