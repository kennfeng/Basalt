from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder


class AtlasReRanker:
    def __init__(
        self,
        model: Any = None,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 32,
    ) -> None:
        if model is not None:
            self.model: Any = model
            self.device: str = getattr(model, "device", "cpu")
            self.batch_size: int = batch_size
            return

        print(f"Loading PyTorch Re-ranker model: {model_name}...")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size: int = batch_size
        try:
            self.model: Any = CrossEncoder(model_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}'. "
                "Check network access, Hugging Face cache, and model name."
            ) from exc
        print(f"Model loaded on {self.device}")

    def _score_pairs(self, pairs: Sequence[list[str]]) -> tuple[np.ndarray, list[int]]:
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return scores, ranked_indices

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int = 3
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores, ranked_indices = self._score_pairs(pairs)

        return [
            {"document": documents[i], "score": float(scores[i])}
            for i in ranked_indices[:top_n]
        ]

    def rerank_with_ids(
        self, query: str, candidates: Sequence[tuple[str, str]], top_n: int = 3
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, doc] for _, doc in candidates]
        scores, ranked_indices = self._score_pairs(pairs)

        return [
            {
                "id": candidates[i][0],
                "document": candidates[i][1],
                "score": float(scores[i]),
            }
            for i in ranked_indices[:top_n]
        ]


if __name__ == "__main__":
    ranker = AtlasReRanker()
    test_query = "How do I build a RAG system?"
    test_docs = [
        "To build a RAG system, you need a vector database and an LLM.",
        "Making a sandwich requires bread, cheese, and ham.",
        "Retrieval-Augmented Generation (RAG) combines search with LLM generation for better accuracy.",
        "The weather today is sunny with a chance of rain.",
    ]

    print("\nOriginal Documents Count:", len(test_docs))
    ranked = ranker.rerank(test_query, test_docs)

    print("\nTop Ranked Results:")
    for i, res in enumerate(ranked):
        print(f"{i + 1}. [Score: {res['score']:.4f}] {res['document']}")
