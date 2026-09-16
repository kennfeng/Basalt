# Basalt

Research copilot over arXiv abstracts. Local two-stage RAG with hybrid retrieval and cross-encoder re-ranking — no data leaves your machine.

Retrieves with dense embeddings + BM25 (RRF), re-ranks with a cross-encoder, generates grounded answers with Ollama or Gemini.

```
Query → [Hybrid: dense (MiniLM) + sparse (BM25) → RRF k=60 → top 20]
      → [Cross-encoder (bge-reranker) → top 3]
      → [LLM (llama3.2:1b) → answer + citations]
```

## Features

- **Hybrid retrieval** — `rank-bm25` + Chroma HNSW fused by reciprocal rank fusion; rescues lexical IDs that dense alone misses.
- **Grounded generation** — context-only system prompt; answers cite re-ranked abstracts.
- **Local by default, swappable** — embedded Chroma, `BASALT_PROVIDER=ollama|openai` via `create_llm`.
- **Evaluated** — 15 queries over 46 docs (15 hard negatives); +57% MRR from re-ranking (see Evaluation).

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (or an OpenAI-compatible endpoint for Gemini)

### Install

```bash
git clone https://github.com/kennfeng/Basalt && cd Basalt
pip install -r requirements.txt -r requirements-dev.txt
ollama pull llama3.2:1b
```

### Run — research copilot over 250 arXiv abstracts

The repo ships `data/sample.jsonl` (250 abstracts, `cs.AI`/`cs.CL`/`cs.IR`/`cs.CV`/`cs.LG`) and a 9-doc in-memory fallback (`sample_data.py`) for quick start.

**Quick start (9 docs):**

```bash
python main.py
# Ask Basalt (or type 'exit'): What is RAG and why use a vector DB?
```

**Full use case (250 abstracts, persistent):**

```bash
# ingest 250 abstracts — idempotent, checkpointed (ctrl-c safe with --resume)
python -m scripts.bulk_ingest --source data/sample.jsonl --batch-size 512

# BM25 index is built automatically on first query (to ./basalt_db/bm25.json)
BASALT_HYBRID=true python main.py
# Try: What is cross-encoder re-ranking for scientific literature?
#      How does hybrid retrieval improve recall?
#      What is the latency tradeoff between vector search and re-ranking?
```

Ingest any JSONL: `python -m scripts.bulk_ingest --source corpus.jsonl --resume --checkpoint corpus.checkpoint --db-path ./basalt_db`

### HTTP API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"What is cross-encoder re-ranking?"}'
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status":"ok"\|"degraded","pipeline_initialized","db_ready","ollama_reachable","db_count"}` — 200 or 503. No model load. |
| `POST` | `/ask` | `{"query":"..."}` (1–2000 chars) → `{"answer","source_documents":[{"id","document","score"}]}`. 401 if `BASALT_API_KEY` set, 503 if busy. |

Health is `ok` when the DB directory exists and Ollama is reachable; the collection is seeded idempotently on first `BasaltRAG` init.

## Configuration

`LangChainRAG.from_defaults()` and `BasaltRAG` resolve `explicit arg > $BASALT_* > default`:

| Variable | Default | Notes |
|---|---|---|
| `BASALT_DB_PATH` | `./basalt_db` | Chroma persistent dir (`/data/basalt_db` in Docker) |
| `BASALT_RERANKER_MODEL` | `BAAI/bge-reranker-base` | Use `bge-reranker-small` for 512 MB hosts |
| `BASALT_LLM_MODEL` | `llama3.2:1b` | Any Ollama tag or Gemini model via `openai` provider |
| `BASALT_PROVIDER` | `ollama` | `ollama` or `openai` |
| `BASALT_BASE_URL` | unset | e.g. `http://localhost:11434` or Gemini `.../v1beta/openai/` |
| `BASALT_N_RESULTS` | `10` | Dense candidates before rerank |
| `BASALT_TOP_N` | `3` | Docs returned to LLM (capped at 20) |
| `BASALT_HYBRID` | `true` | `false` disables BM25+RRF |
| `BASALT_HYBRID_DENSE_N` / `SPARSE_N` / `RRF_K` / `TOP_N` | `50` / `50` / `60` / `20` | Hybrid RRF tuning |
| `BASALT_HNSW_M` / `CONSTRUCTION_EF` / `SEARCH_EF` | `16` / `200` / `10` | HNSW index tuning |
| `BASALT_BATCH_SIZE` | `512` | `upsert` batch size |
| `BASALT_EMBED_DEVICE` / `EMBED_BATCH_SIZE` | `cpu` / `32` | Embedding device + batch |
| `BASALT_RERANKER_DEVICE` / `BATCH_SIZE` / `FP16` | `cpu` / `32` / `false` | Reranker accel |
| `BASALT_MAX_CONCURRENCY` / `ASK_TIMEOUT` | `4` / `30` | API semaphore + timeout |
| `BASALT_AUTO_SEED` | `true` | `false` disables seeding empty DB |
| `BASALT_API_KEY` | unset | If set, requires `X-API-Key` header |
| `BASALT_CHROMA_HOST` / `PORT` | unset | If set, use `HttpClient` instead of embedded |

Example: `BASALT_PROVIDER=openai BASALT_LLM_MODEL=gemini-2.0-flash OPENAI_API_KEY=... uvicorn app:app`

## Evaluation

```bash
python eval/run_eval.py --yes                 # retrieval vs retrieval+rerank
python eval/run_eval.py --yes --keep-db       # keep DB for generation eval
python eval/run_generation_eval.py --db-path eval/eval_db  # requires Ollama
```

Dataset: `eval/eval_dataset.json` — 46 docs (31 ground-truth + 15 hard negatives) × 15 queries. Warm-up query excluded from timing. `eval/results.json` is committed and frozen.

| Metric | Retrieval only | Retrieval + re-rank |
|---|---|---|
| Hit Rate @3 | 100% | 100% |
| MRR @3 | 0.467 | **0.733 (+57%)** |
| Latency (mean, CPU) | ~55 ms | ~1.87 s (p50 1.85 s, p90 2.02 s) |

Re-ranking puts the correct abstract at rank 1 in 7/15 more queries, at ~34× latency cost (batch 32, `bge-reranker-base`). Use `EvalReporter` (`eval/analyzer.py`) and `GenerationReporter` (`eval/generation_analyzer.py`) for per-query, percentile, and `compare()` analysis.

## Testing

```bash
python -m pytest tests/ -q   # 219 tests, mocked torch/chromadb/sentence-transformers — no GPU or downloads
ruff check --fix . && ruff format .
```

## Deployment

```bash
cp .env.example .env
docker compose up --build -d          # app :8000, ollama :11434
docker compose --profile chroma up -d  # optional remote Chroma
```

Image pre-pulls HF models (`all-MiniLM-L6-v2`, `bge-reranker`) and seeds `BASALT_DB_PATH` at boot via `scripts/entrypoint.sh:1`. Models load lazily on first `/ask`. For GPU: `docker build -f Dockerfile.gpu -t basalt-api-gpu .` and uncomment `deploy.resources` in `docker-compose.yml` (`--gpus all`).

## Project Structure

```
.
├── main.py                 # BasaltRAG CLI
├── app.py                  # FastAPI /health, /ask
├── ingest.py               # BasaltIngestor (batched upsert, HNSW, chunking)
├── hybrid.py               # BM25Index + HybridRetriever (RRF)
├── reranker.py             # BasaltReRanker
├── rag_pipeline.py         # LangChainRAG (hybrid → rerank → LLM)
├── langchain_adapters.py   # create_llm + adapters
├── sample_data.py          # 9-doc fallback corpus
├── data/sample.jsonl       # 250 arXiv abstracts (demo corpus)
├── eval/
│   ├── eval_dataset.json   # 46 docs × 15 queries
│   ├── results.json        # frozen retrieval results
│   └── analyzer.py         # EvalReporter
├── scripts/
│   ├── bulk_ingest.py      # JSONL → Chroma (checkpointed)
│   ├── fetch_arxiv.py      # arXiv bulk fetch
│   └── bench_scale.py      # scale harness
└── tests/
```
