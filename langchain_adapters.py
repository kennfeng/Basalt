from typing import Any

from langchain_core.language_models import BaseChatModel

from ingest import BasaltIngestor
from reranker import BasaltReRanker


def create_llm(
    provider: str = "ollama",
    model_name: str = "llama3.2:1b",
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    # build a chat model for either local or hosted serving
    # pick ollama by default or openai path
    if base_url is not None:
        kwargs["base_url"] = base_url

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, **kwargs)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for the 'openai' provider. "
                "Install it with: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(model=model_name, **kwargs)

    raise ValueError(f"Unknown provider: {provider!r}. Supported: 'ollama', 'openai'.")


class ChromaRetrieverAdapter:
    # expose vector search in the retriever shape the pipeline expects
    # hold the ingestor and default candidate count
    def __init__(self, ingestor: BasaltIngestor, n_results: int = 10) -> None:
        self.ingestor = ingestor
        self.n_results = n_results

    def get_relevant_documents_with_ids(self, query: str) -> list[tuple[str, str]]:
        # fetch candidate ids plus texts for later fusion and rerank
        # delegate to vector search with the configured result count
        return self.ingestor.search_with_ids(query, n_results=self.n_results)


class CrossEncoderRerankerAdapter:
    # expose precise reranking in the shape the pipeline expects
    # hold the ranker and final top count sent to the llm
    def __init__(self, ranker: BasaltReRanker, top_n: int = 3) -> None:
        self.ranker = ranker
        self.top_n = top_n

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[dict]:
        # order fused candidates and keep ids for citations
        # delegate to id-aware rerank with the configured top count
        return self.ranker.rerank_with_ids(query, candidates, top_n=self.top_n)
