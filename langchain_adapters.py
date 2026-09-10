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
    def __init__(self, ingestor: BasaltIngestor, n_results: int = 10) -> None:
        self.ingestor = ingestor
        self.n_results = n_results

    def get_relevant_documents_with_ids(self, query: str) -> list[tuple[str, str]]:
        return self.ingestor.search_with_ids(query, n_results=self.n_results)


class CrossEncoderRerankerAdapter:
    def __init__(self, ranker: BasaltReRanker, top_n: int = 3) -> None:
        self.ranker = ranker
        self.top_n = top_n

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[dict]:
        return self.ranker.rerank_with_ids(query, candidates, top_n=self.top_n)
