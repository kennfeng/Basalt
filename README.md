# Basalt

Research copilot over arXiv abstracts. Local two-stage **Retrieval-Augmented Generation (RAG)** with **Hybrid Retrieval** and **Cross-Encoder Re-ranking**

Retrieves with dense embeddings + **BM25** (**Reciprocal Rank Fusion**), re-ranks with a **Cross-Encoder**, generates grounded answers with an **LLM** (Ollama or Gemini). Built with **Python**, **FastAPI**, **ChromaDB**, **Sentence-Transformers**, **LangChain**, **Docker**.

## Visual Demo

Static SPA served at `/ui`

```bash
docker compose up --build -d && open http://localhost:8001/ui/
```

**Screens:** Ask (answer + source cards) · Inspect (retrieval trace) · Corpus (250 abstracts) · Eval (Hit@3 / MRR / latency). Streaming via `POST /ask/stream`.

## Features

- **Hybrid Retrieval - BM25 + Dense + RRF** - sparse + dense fused by **Reciprocal Rank Fusion**.
- **Cross-Encoder Re-ranking** - `BAAI/bge-reranker-base`: **MRR 0.467 → 0.733 (+57%)** at `k=3`.
- **Grounded generation** - answers cite re-ranked abstracts only.
- **Local by default, swappable** - embedded Chroma, `BASALT_PROVIDER=ollama|openai`.
- **FastAPI + Observability** - `GET /health`, `POST /ask`, `POST /ask/stream` (**SSE**), `GET /scale_report`, `GET /ui/`. `X-Trace-Id` on every response. OpenAPI at `/docs`.
- **Evaluated** - 15 queries over 46 docs (see Evaluation).

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

### Run - research copilot over 250 arXiv abstracts (Docker)

The repo ships `data/sample.jsonl` (250 abstracts). All runs via Docker on `:8001`.

```bash
docker compose up --build -d
open http://localhost:8001/ui/
```

### HTTP API - Docker (`:8001`)

```bash
docker compose up --build -d
curl http://localhost:8001/health
curl -X POST http://localhost:8001/ask -H "Content-Type: application/json" \
  -d '{"query":"What is cross-encoder re-ranking?"}'
curl -N -X POST http://localhost:8001/ask/stream -H "Content-Type: application/json" \
  -d '{"query":"What is cross-encoder re-ranking?"}'
open http://localhost:8001/ui/
```

### HTTP API - Local venv alternate (`:8000`, no Docker)

```bash
source .venv/bin/activate && ollama serve &
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"What is cross-encoder re-ranking?"}'
curl -N -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" \
  -d '{"query":"What is cross-encoder re-ranking?"}'
open http://localhost:8000/ui/
```

## Evaluation

```bash
python eval/run_eval.py --yes                 # retrieval vs retrieval+rerank
python eval/run_eval.py --yes --keep-db       # keep DB for generation eval
python eval/run_generation_eval.py --db-path eval/eval_db  # requires Ollama
```

Dataset: `eval/eval_dataset.json` - 46 docs × 15 queries.

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

## Deployment - Docker only

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
