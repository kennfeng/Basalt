# Basalt

Research copilot over arXiv abstracts. Local two-stage **Retrieval-Augmented Generation (RAG)** with **Hybrid Retrieval** and **Cross-Encoder Re-ranking** — no data leaves your machine.

Retrieves with dense embeddings + **BM25** (**Reciprocal Rank Fusion**), re-ranks with a **Cross-Encoder**, generates grounded answers with an **LLM** (Ollama or Gemini). Built with **Python**, **FastAPI**, **ChromaDB**, **Sentence-Transformers**, **LangChain**, **Docker**.

```
Query → [Hybrid: dense (MiniLM) + sparse (BM25) → RRF k=60 → top 20]
      → [Cross-encoder (bge-reranker) → top 3]
      → [LLM (llama3.2:1b) → answer + citations]
```

## Visual Demo

Static SPA served at `/ui`

```bash
docker compose up --build -d && open http://localhost:8001/ui/
```

**Screens:** Ask (answer + source cards) · Inspect (retrieval trace) · Corpus (250 abstracts) · Eval (Hit@3 / MRR / latency). Streaming via `POST /ask/stream`.

## Features

- **Hybrid Retrieval — BM25 + Dense + RRF** — sparse + dense fused by **Reciprocal Rank Fusion**.
- **Cross-Encoder Re-ranking** — `BAAI/bge-reranker-base`: **MRR 0.467 → 0.733 (+57%)** at `k=3`.
- **Grounded generation** — answers cite re-ranked abstracts only.
- **Local by default, swappable** — embedded Chroma, `BASALT_PROVIDER=ollama|openai`.
- **FastAPI + Observability** — `GET /health`, `POST /ask`, `POST /ask/stream` (**SSE**), `GET /scale_report`, `GET /ui/`. `X-Trace-Id` on every response. OpenAPI at `/docs`.
- **Evaluated** — 15 queries over 46 docs (see Evaluation).

## Getting Started

### Prerequisites

- **Python** 3.10+
- [Ollama](https://ollama.com/) (or an OpenAI-compatible endpoint for Gemini)
- **Docker** / **Docker Compose** (for one-command demo)

### Install

```bash
git clone https://github.com/kennfeng/Basalt && cd Basalt
pip install -r requirements.txt -r requirements-dev.txt
ollama pull llama3.2:1b
```

### Run — research copilot over 250 arXiv abstracts (Docker)

The repo ships `data/sample.jsonl` (250 abstracts). All runs via Docker on `:8001`.

```bash
docker compose up --build -d
open http://localhost:8001/ui/
```

### HTTP API — Docker (`:8001`)

```bash
docker compose up --build -d
curl http://localhost:8001/health
curl -X POST http://localhost:8001/ask -H "Content-Type: application/json" \
  -d '{"query":"What is cross-encoder re-ranking?"}'
curl -N -X POST http://localhost:8001/ask/stream -H "Content-Type: application/json" \
  -d '{"query":"What is cross-encoder re-ranking?"}'
open http://localhost:8001/ui/
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status, `db_count`, readiness — 200 or 503. |
| `POST` | `/ask` | `{"query":"..."}` (1–2000 chars) → `{"answer","source_documents"}`. |
| `POST` | `/ask/stream` | Streaming answer (**SSE**) + final `source_documents`. |
| `GET` | `/scale_report` | Benchmark results. |
| `GET` | `/ui/` | Static SPA. |
| `GET` | `/` | `302` → `/ui/` |
| `GET` | `/docs` | OpenAPI. |

Health is `ok` when the DB directory exists and Ollama is reachable.

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
| `BASALT_HNSW_M` / `CONSTRUCTION_EF` / `SEARCH_EF` | `16` / `200` / `10` | HNSW tuning |
| `BASALT_BATCH_SIZE` | `512` | Ingest batch size |
| `BASALT_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `BASALT_MAX_CONCURRENCY` / `ASK_TIMEOUT` | `4` / `30` | API concurrency + timeout |
| `BASALT_AUTO_SEED` | `true` | Seed empty DB on first run |
| `BASALT_API_KEY` | unset | If set, requires `X-API-Key` header |
| `BASALT_CHROMA_HOST` / `PORT` | unset | If set, use `HttpClient` instead of embedded |

Example: `BASALT_PROVIDER=openai BASALT_LLM_MODEL=gemini-2.0-flash OPENAI_API_KEY=... docker compose up --build -d`

## Evaluation

```bash
python eval/run_eval.py --yes                 # retrieval vs retrieval+rerank
python eval/run_eval.py --yes --keep-db       # keep DB for generation eval
python eval/run_generation_eval.py --db-path eval/eval_db  # requires Ollama
```

Dataset: `eval/eval_dataset.json` — 46 docs × 15 queries.

| Metric | Retrieval only | Retrieval + re-rank |
|---|---|---|
| Hit Rate @3 | 100% | 100% |
| MRR @3 | 0.467 | **0.733 (+57%)** |
| Latency (mean, CPU) | ~55 ms | ~1.87 s |

## Scale

```bash
python -m scripts.bench_scale --corpus data/synth_10k.jsonl --output eval/scale_report.json --n-values 10000,100000 --real
curl http://localhost:8001/scale_report
```

## Testing & CI/CD

```bash
python -m pytest tests/ -q
ruff check --fix . && ruff format .
```

CI runs `ruff check` / `ruff format --check` / `pytest` on every push.

## Deployment — Docker only

```bash
cp .env.example .env
docker compose up --build -d          # app :8001, ollama :11434
```

Image pre-pulls the models and seeds `BASALT_DB_PATH` at boot. Models load lazily on first `/ask`.

UI is served from the same image (`ui/` → `/ui/`), no Node at runtime.

## Project Structure

```
.
├── main.py                 # CLI
├── app.py                  # FastAPI service
├── ingest.py               # ingestion
├── hybrid.py               # hybrid retrieval
├── reranker.py             # re-ranking
├── rag_pipeline.py         # RAG pipeline
├── langchain_adapters.py   # LLM adapters
├── sample_data.py          # fallback corpus
├── data/sample.jsonl       # 250 arXiv abstracts
├── ui/                     # static SPA
├── eval/                   # evaluation harness
├── scripts/                # ingest / fetch / bench
└── tests/
```
