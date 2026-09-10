import os
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from sample_data import SAMPLE_DOCUMENTS


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must be < chunk_size, got chunk_overlap={chunk_overlap}, "
            f"chunk_size={chunk_size}"
        )
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - chunk_overlap
    return chunks if chunks else [text]


def _parse_env_int(name: str, default: int | None = None) -> int | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _create_client(db_path: str) -> Any:
    host = os.getenv("BASALT_CHROMA_HOST")
    if host:
        port = _parse_env_int("BASALT_CHROMA_PORT", 8000) or 8000
        return chromadb.HttpClient(host=host, port=port)
    return chromadb.PersistentClient(path=db_path)


def _hnsw_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"hnsw:space": "cosine"}
    m = _parse_env_int("BASALT_HNSW_M")
    if m is not None:
        metadata["hnsw:M"] = m
    ef_c = _parse_env_int("BASALT_HNSW_CONSTRUCTION_EF")
    if ef_c is not None:
        metadata["hnsw:construction_ef"] = ef_c
    ef_s = _parse_env_int("BASALT_HNSW_SEARCH_EF")
    if ef_s is not None:
        metadata["hnsw:search_ef"] = ef_s
    return metadata


class BasaltIngestor:
    def __init__(
        self,
        db_path: str = "./basalt_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
    ) -> None:
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.embed_device = os.getenv("BASALT_EMBED_DEVICE", "cpu")
        self.embed_batch_size = _parse_env_int("BASALT_EMBED_BATCH_SIZE", 32) or 32
        self.client = _create_client(db_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn,
            metadata=_hnsw_metadata(),
        )

    def add_documents(
        self,
        text_list: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int = 64,
        batch_size: int | None = None,
    ) -> None:
        if chunk_size is not None and chunk_size > 0:
            if chunk_overlap >= chunk_size:
                raise ValueError(
                    f"chunk_overlap must be < chunk_size, got chunk_overlap={chunk_overlap}, "
                    f"chunk_size={chunk_size}"
                )
            chunked_texts: list[str] = []
            chunked_metadata: list[dict[str, Any]] | None = (
                [] if metadata_list is not None else None
            )
            chunked_ids: list[str] = []

            for i, text in enumerate(text_list):
                meta = metadata_list[i] if metadata_list is not None else None
                doc_id = ids[i] if ids is not None else f"id_{i}"
                chunks = _chunk_text(text, chunk_size, chunk_overlap)
                for j, chunk in enumerate(chunks):
                    chunked_texts.append(chunk)
                    if chunked_metadata is not None:
                        chunked_metadata.append(
                            {**(meta or {}), "_chunk_index": j, "_parent_id": doc_id}
                        )
                    chunked_ids.append(f"{doc_id}_chunk_{j}")

            text_list = chunked_texts
            metadata_list = chunked_metadata
            ids = chunked_ids

        if ids is None:
            ids = [f"id_{i}" for i in range(len(text_list))]

        effective_batch = batch_size
        if effective_batch is None:
            effective_batch = _parse_env_int("BASALT_BATCH_SIZE", 512) or 512
        if effective_batch < 1:
            raise ValueError(f"batch_size must be >= 1, got {effective_batch}")

        for start in range(0, len(text_list), effective_batch):
            end = start + effective_batch
            batch_docs = text_list[start:end]
            batch_ids = ids[start:end]
            batch_metas = (
                metadata_list[start:end] if metadata_list is not None else None
            )
            self.collection.upsert(
                documents=batch_docs, metadatas=batch_metas, ids=batch_ids
            )

    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return results["documents"][0]

    def search_with_ids(
        self,
        query: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, str]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        return list(zip(results["ids"][0], results["documents"][0]))


def ensure_seeded(
    db_path: str,
    texts: list[str],
    ids: list[str] | None = None,
    ingestor: BasaltIngestor | None = None,
) -> bool:
    auto_seed = os.getenv("BASALT_AUTO_SEED", "true").lower()
    if auto_seed in ("false", "0", "no", "off"):
        return False
    if ingestor is None:
        ingestor = BasaltIngestor(db_path=db_path)
    if ingestor.collection.count() == 0:
        ingestor.add_documents(text_list=texts, ids=ids)
        return True
    if ids is not None:
        existing_ids = ingestor.collection.get()["ids"]
        if existing_ids:
            ingestor.collection.delete(ids=existing_ids)
        ingestor.add_documents(text_list=texts, ids=ids)
        return True
    return False


if __name__ == "__main__":
    ensure_seeded(db_path="./basalt_db", texts=[text for _, text in SAMPLE_DOCUMENTS])

    ingestor = BasaltIngestor()

    query = "What is RAG and why use a vector DB?"
    candidates = ingestor.search(query, n_results=3)

    print(f"\nQuery: {query}")
    print("Top Candidate Matches:")
    for i, doc in enumerate(candidates):
        print(f"{i + 1}. {doc}")
