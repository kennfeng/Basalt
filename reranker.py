import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder


def _parse_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


class BasaltReRanker:
    # score query plus doc pairs precisely after fast retrieval
    # hold a cross-encoder model with batch size and device choice
    # cross-encoder = reads query and doc together for relevance
    def __init__(
        self,
        model: Any = None,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        env_model = os.getenv("BASALT_RERANKER_MODEL")
        if model_name is None:
            model_name = env_model or "BAAI/bge-reranker-base"
        env_batch = _parse_env_int("BASALT_RERANKER_BATCH_SIZE", 32)
        if batch_size is None:
            batch_size = env_batch
        env_device = os.getenv("BASALT_RERANKER_DEVICE")
        env_fp16 = os.getenv("BASALT_RERANKER_FP16", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        if model is not None:
            self.model: Any = model
            self.device: str = getattr(model, "device", env_device or "cpu")
            self.batch_size: int = batch_size
            self.fp16: bool = env_fp16
            return

        print(f"Loading PyTorch Re-ranker model: {model_name}...")
        if env_device:
            self.device: str = env_device
        else:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size: int = batch_size
        self.fp16: bool = env_fp16
        try:
            self.model: Any = CrossEncoder(model_name, device=self.device)
            if self.fp16 and hasattr(self.model, "model"):
                try:
                    self.model.model.half()
                except Exception:  # noqa: BLE001, S110
                    pass
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}'. "
                "Check network access, Hugging Face cache, and model name."
            ) from exc
        print(f"Model loaded on {self.device}")

    def _score_pairs(self, pairs: Sequence[list[str]]) -> tuple[np.ndarray, list[int]]:
        # score paired inputs and order them best first
        # predict relevance per pair, return scores plus ranked positions
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return scores, ranked_indices

    def _validate_top_n(self, top_n: int) -> None:
        # reject negative result sizes before model work starts
        if top_n < 0:
            raise ValueError(f"top_n must be >= 0, got {top_n}")

    def rerank(
        self, query: str, documents: Sequence[str], top_n: int = 3
    ) -> list[dict[str, Any]]:
        # order plain texts by relevance to the query
        # pair each doc with query, score, return top docs with scores
        # top_n capped at 20 to bound prompt size and latency
        if not documents:
            return []
        self._validate_top_n(top_n)
        top_n = min(top_n, 20)

        pairs = [[query, doc] for doc in documents]
        scores, ranked_indices = self._score_pairs(pairs)

        return [
            {"document": documents[i], "score": float(scores[i])}
            for i in ranked_indices[:top_n]
        ]

    def rerank_with_ids(
        self, query: str, candidates: Sequence[tuple[str, str]], top_n: int = 3
    ) -> list[dict[str, Any]]:
        # order id plus text pairs
        # same scoring as rerank but return id, document, and score
        # id = stable doc key used for citations and eval matching
        if not candidates:
            return []
        self._validate_top_n(top_n)
        top_n = min(top_n, 20)

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
    ranker = BasaltReRanker()
    test_query = "How do I build a RAG system?"
    test_docs = [
        "To build a RAG system, you need a vector database and an LLM.",
        "Making a sandwich requires bread, cheese, and ham.",
        (
            "Retrieval-Augmented Generation (RAG) combines search "
            "with LLM generation for better accuracy."
        ),
        "The weather today is sunny with a chance of rain.",
    ]

    print("\nOriginal Documents Count:", len(test_docs))
    ranked = ranker.rerank(test_query, test_docs)

    print("\nTop Ranked Results:")
    for i, res in enumerate(ranked):
        print(f"{i + 1}. [Score: {res['score']:.4f}] {res['document']}")
