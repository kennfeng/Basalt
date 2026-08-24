import os
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ingest import AtlasIngestor, ensure_seeded
from langchain_adapters import (
    ChromaRetrieverAdapter,
    CrossEncoderRerankerAdapter,
    create_llm,
)
from reranker import AtlasReRanker
from sample_data import SAMPLE_DOCUMENTS

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question based only on the "
    "provided context. If the context does not contain enough information to answer "
    'the question, say "I don\'t have enough information to answer this question." '
    "Do not make up or infer information beyond what is stated in the context. "
    "Cite specific parts of the context when possible."
)

HUMAN_TEMPLATE = "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


class LangChainRAG:
    def __init__(
        self,
        retriever: ChromaRetrieverAdapter,
        reranker: CrossEncoderRerankerAdapter,
        llm: Runnable,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_TEMPLATE),
            ]
        )
        self._chain: Runnable | None = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            self._chain = self.prompt | self.llm | StrOutputParser()
        return self._chain

    @classmethod
    def from_defaults(
        cls,
        db_path: str | None = None,
        model_name: str | None = None,
        llm_model_name: str | None = None,
        provider: str | None = None,
        n_results: int | None = None,
        top_n: int | None = None,
        sample_docs: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ) -> "LangChainRAG":
        db_path = (
            db_path
            if db_path is not None
            else os.environ.get("ATLAS_DB_PATH", "./atlas_db")
        )
        model_name = (
            model_name
            if model_name is not None
            else os.environ.get("ATLAS_RERANKER_MODEL", "BAAI/bge-reranker-base")
        )
        llm_model_name = (
            llm_model_name
            if llm_model_name is not None
            else os.environ.get("ATLAS_LLM_MODEL", "llama3.2:1b")
        )
        provider = (
            provider
            if provider is not None
            else os.environ.get("ATLAS_PROVIDER", "ollama")
        )
        n_results = (
            n_results
            if n_results is not None
            else int(os.environ.get("ATLAS_N_RESULTS", "10"))
        )
        top_n = top_n if top_n is not None else int(os.environ.get("ATLAS_TOP_N", "3"))
        if base_url is None:
            base_url = os.environ.get("ATLAS_OLLAMA_BASE_URL")

        ingestor = AtlasIngestor(db_path=db_path)
        ensure_seeded(
            db_path=db_path,
            texts=sample_docs or [text for _, text in SAMPLE_DOCUMENTS],
            ingestor=ingestor,
        )

        ranker = AtlasReRanker(model_name=model_name)
        retriever = ChromaRetrieverAdapter(ingestor, n_results=n_results)
        reranker = CrossEncoderRerankerAdapter(ranker, top_n=top_n)

        llm_kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            llm_kwargs["num_predict" if provider == "ollama" else "max_tokens"] = (
                max_tokens
            )

        llm = create_llm(
            provider=provider,
            model_name=llm_model_name,
            **llm_kwargs,
            **({"base_url": base_url} if base_url is not None else {}),
        )

        return cls(retriever, reranker, llm)

    def _prepare(self, query: str) -> dict[str, Any]:
        candidates = self.retriever.get_relevant_documents_with_ids(query)
        if not candidates:
            return {"context": "", "source_documents": []}

        ranked_results = self.reranker.rerank(query, candidates)
        context_docs = [res["document"] for res in ranked_results]
        context = "\n\n".join(context_docs)
        return {"context": context, "source_documents": ranked_results}

    def ask(self, query: str) -> dict[str, Any]:
        prepared = self._prepare(query)
        if not prepared["source_documents"]:
            return {
                "answer": "I couldn't find any relevant documents in the database.",
                "source_documents": [],
            }

        answer = self.chain.invoke(
            {
                "context": prepared["context"],
                "query": query,
            }
        )
        return {
            "answer": answer,
            "source_documents": prepared["source_documents"],
        }
