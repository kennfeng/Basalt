import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore

from ingest import BasaltIngestor


def tokenize(text: str) -> list[str]:
    # split text into searchable word tokens for sparse matching
    return re.findall(r"\w+", text.lower())


# merge meaning-match and word-match into one better ranking
# add weight * 1/(k+rank) per appearance
def rrf_fuse(
    dense_ids: list[str],
    sparse_ids: list[str],
    k: int = 60,
    top_n: int = 20,
    weights: tuple[float, float] = (0.5, 0.5),
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + weights[0] * (1.0 / (k + rank))
    for rank, doc_id in enumerate(sparse_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + weights[1] * (1.0 / (k + rank))
    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return ranked[:top_n]


class BM25Index:
    # keep exact-word search ready without re-reading the corpus
    # hold ids, texts, and token lists plus a bm25 scorer
    # bm25 = ranks by rare-word overlap
    def __init__(
        self,
        corpus_ids: list[str],
        corpus_texts: list[str],
        tokenized_corpus: list[list[str]] | None = None,
    ) -> None:
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required: pip install rank-bm25")
        self.corpus_ids = list(corpus_ids)
        self.corpus_texts = list(corpus_texts)
        self.id_to_doc: dict[str, str] = dict(zip(corpus_ids, corpus_texts))
        self.id_to_idx: dict[str, int] = {
            doc_id: i for i, doc_id in enumerate(corpus_ids)
        }
        if tokenized_corpus is None:
            tokenized_corpus = [tokenize(t) for t in corpus_texts]
        self.tokenized_corpus = tokenized_corpus
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, n_results: int = 50) -> list[tuple[str, str]]:
        # find docs with the same words as the query
        # score token overlap, drop zero scores, keep best n_results
        if not query.strip():
            return []
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[str, str]] = []
        for idx in ranked_idx[:n_results]:
            if scores[idx] <= 0:
                continue
            doc_id = self.corpus_ids[idx]
            out.append((doc_id, self.corpus_texts[idx]))
        return out

    def save(self, path: Path) -> None:
        # persist the sparse index next to the vector store
        # write to tmp file first, gate large files, then replace target
        # sharded = split across files, tantivy = alternate sparse backend
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "corpus_ids": self.corpus_ids,
                    "corpus_texts": self.corpus_texts,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
            )
        try:
            if tmp.stat().st_size > 200 * 1024 * 1024:
                sharded = os.getenv("BASALT_BM25_SHARDED", "false").lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )
                tantivy = os.getenv("BASALT_BM25_TANTIVY", "false").lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )
                if not sharded and not tantivy:
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001, S110
                        pass
                    raise RuntimeError(
                        "bm25.json >200MB requires BASALT_BM25_SHARDED=true or "
                        "BASALT_BM25_TANTIVY=true; shard to bm25_shard_*.json + "
                        "bm25_manifest.json"
                    )
                warnings.warn(
                    "bm25.json >200MB ... consider sharded json or tantivy",
                    UserWarning,
                    stacklevel=2,
                )
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001, S110
            pass
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        # rebuild the sparse index from a saved json file
        # read ids, texts, and tokens then construct a new index
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            corpus_ids=data["corpus_ids"],
            corpus_texts=data["corpus_texts"],
            tokenized_corpus=data.get("tokenized_corpus"),
        )

    @classmethod
    def build_from_jsonl(cls, jsonl_path: Path) -> "BM25Index":
        # build a fresh sparse index from a jsonl corpus file
        # read each line id plus abstract or text, skip incomplete rows
        # jsonl = one json doc per line, abstract = paper summary
        ids: list[str] = []
        texts: list[str] = []
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                doc_id = str(obj.get("id") or obj.get("doc_id") or "")
                abstract = (
                    obj.get("abstract")
                    or obj.get("text")
                    or obj.get("document")
                    or obj.get("content")
                    or ""
                )
                if not doc_id or not abstract:
                    continue
                ids.append(doc_id)
                texts.append(str(abstract))
        return cls(corpus_ids=ids, corpus_texts=texts)


def build_from_collection(collection: Any, limit: int = 10000) -> BM25Index | None:
    # rebuild sparse search from stored vectors without a corpus file
    # page through collection ids and docs, then build a new index
    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    offset = 0
    while True:
        try:
            batch = collection.get(
                limit=limit, offset=offset, include=["documents", "ids"]
            )
        except TypeError:
            try:
                fallback = collection.get(include=["documents", "ids"])
                f_ids = fallback.get("ids", [])
                f_docs = fallback.get("documents", [])
                if f_ids and f_docs:
                    corpus_ids = list(f_ids)
                    corpus_texts = list(f_docs)
            except Exception:  # noqa: BLE001, S110
                pass
            break
        except Exception:  # noqa: BLE001, S110
            break
        ids = batch.get("ids", [])
        docs = batch.get("documents", [])
        if not ids or not docs:
            break
        corpus_ids.extend(list(ids))
        corpus_texts.extend(list(docs))
        if len(ids) < limit:
            break
        offset += limit
    if corpus_ids and corpus_texts:
        return BM25Index(corpus_ids=corpus_ids, corpus_texts=corpus_texts)
    return None


def load_or_build_bm25(
    json_path: Path | None = None,
    pickle_path: Path | None = None,
    jsonl_path: Path | None = None,
    corpus_ids: list[str] | None = None,
    corpus_texts: list[str] | None = None,
    db_count: int | None = None,
    collection: Any | None = None,
) -> BM25Index | None:
    # return a usable sparse index with the cheapest safe source
    if pickle_path is not None and json_path is None:
        warnings.warn(
            "pickle_path deprecated use json_path",
            DeprecationWarning,
            stacklevel=2,
        )
        json_path = pickle_path
    if json_path is not None and Path(json_path).exists():
        try:
            return BM25Index.load(Path(json_path))
        except Exception:  # noqa: BLE001, S110
            pass
    if corpus_ids is not None and corpus_texts is not None:
        idx = BM25Index(corpus_ids=corpus_ids, corpus_texts=corpus_texts)
        if json_path is not None:
            try:
                idx.save(Path(json_path))
            except Exception:  # noqa: BLE001, S110
                pass
        return idx
    effective_count: int | None = db_count
    if effective_count is None and collection is not None:
        try:
            effective_count = int(collection.count())
        except Exception:  # noqa: BLE001, S110
            effective_count = None
    if effective_count is not None and effective_count > 1000:
        if collection is None:
            return None
        idx = build_from_collection(collection, limit=10000)
        if idx is None:
            return None
        if json_path is not None:
            try:
                idx.save(Path(json_path))
            except Exception:  # noqa: BLE001, S110
                pass
        return idx
    if jsonl_path is not None and Path(jsonl_path).exists():
        idx = BM25Index.build_from_jsonl(Path(jsonl_path))
        if json_path is not None:
            try:
                idx.save(Path(json_path))
            except Exception:  # noqa: BLE001, S110
                pass
        return idx
    return None


class HybridRetriever:
    # combine vector search and word search for stronger candidates
    # dense = vector meaning search, sparse = bm25 word search
    def __init__(
        self,
        ingestor: BasaltIngestor,
        bm25_index: BM25Index | None = None,
        dense_n: int = 50,
        sparse_n: int = 50,
        rrf_k: int = 60,
        top_n: int = 20,
    ) -> None:
        self.ingestor = ingestor
        self.bm25_index = bm25_index
        self.dense_n = dense_n
        self.sparse_n = sparse_n
        self.rrf_k = rrf_k
        self.top_n = top_n

    def retrieve(self, query: str) -> list[tuple[str, str]]:
        # get the fused top list used before reranking
        # run dense and sparse, fuse ids, return pairs in fused order
        dense_results = self.ingestor.search_with_ids(query, n_results=self.dense_n)
        dense_ids = [doc_id for doc_id, _ in dense_results]
        dense_map = dict(dense_results)
        sparse_results: list[tuple[str, str]] = []
        if self.bm25_index is not None:
            sparse_results = self.bm25_index.search(query, n_results=self.sparse_n)
        sparse_ids = [doc_id for doc_id, _ in sparse_results]
        sparse_map = dict(sparse_results)
        if not sparse_results:
            return dense_results[: self.top_n]
        if not dense_results:
            return sparse_results[: self.top_n]
        fused_ids = rrf_fuse(dense_ids, sparse_ids, k=self.rrf_k, top_n=self.top_n)
        fused: list[tuple[str, str]] = []
        for doc_id in fused_ids:
            doc = dense_map.get(doc_id) or sparse_map.get(doc_id) or ""
            if doc:
                fused.append((doc_id, doc))
        return fused

    def hybrid_search(
        self, query: str, n_results: int | None = None
    ) -> list[tuple[str, str]]:
        # same fusion as retrieve with a per-call result size
        top = n_results if n_results is not None else self.top_n
        dense_n = max(self.dense_n, top)
        sparse_n = max(self.sparse_n, top)
        dense_results = self.ingestor.search_with_ids(query, n_results=dense_n)
        dense_ids = [doc_id for doc_id, _ in dense_results]
        dense_map = dict(dense_results)
        sparse_results: list[tuple[str, str]] = []
        if self.bm25_index is not None:
            sparse_results = self.bm25_index.search(query, n_results=sparse_n)
        sparse_ids = [doc_id for doc_id, _ in sparse_results]
        sparse_map = dict(sparse_results)
        if not sparse_results:
            return dense_results[:top]
        if not dense_results:
            return sparse_results[:top]
        fused_ids = rrf_fuse(dense_ids, sparse_ids, k=self.rrf_k, top_n=top)
        return [
            (doc_id, dense_map.get(doc_id) or sparse_map.get(doc_id, ""))
            for doc_id in fused_ids
        ]
