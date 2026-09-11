from typing import Any

import httpx

from rag_pipeline import LangChainRAG


class BasaltRAG:
    def __init__(
        self,
        pipeline: LangChainRAG | None = None,
        model: str | None = None,
        provider: str | None = None,
        sample_docs: list[str] | None = None,
        n_results: int | None = None,
        top_n: int | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ) -> None:
        print("--- Initializing RAG System ---")
        self.pipeline = pipeline or LangChainRAG.from_defaults(
            llm_model_name=model,
            provider=provider,
            sample_docs=sample_docs,
            n_results=n_results,
            top_n=top_n,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )

    def ask(self, query: str) -> dict[str, Any]:
        try:
            return self.pipeline.ask(query[:2000])
        except (ConnectionError, httpx.TransportError):
            return {
                "answer": "ERROR: Could not connect to Ollama. Please check that Ollama is running.",
                "source_documents": [],
            }


if __name__ == "__main__":
    rag = BasaltRAG()
    while True:
        try:
            user_input = input("\nAsk Basalt (or type 'exit'): ")
            if user_input.lower() == "exit":
                break
            if not user_input.strip():
                continue

            result = rag.ask(user_input)
            print(f"\n--- RESPONSE ---\n{result['answer']}")

            if result["source_documents"]:
                print("\n--- SOURCES (Re-ranked) ---")
                for i, src in enumerate(result["source_documents"]):
                    print(f"{i + 1}. [{src['score']:.4f}] {src['document']}")
        except (EOFError, KeyboardInterrupt):
            break
