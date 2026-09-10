# Basalt

Basalt is a local, two-stage Retrieval-Augmented Generation (RAG) system.

### How It Works
```
User Query
    │
    ▼
[Stage 1] Bi-encoder Vector Search
    │  Returns top-N candidate documents
    ▼
[Stage 2] Cross-Encoder Re-ranking
    │  Scores each (query, document) pair jointly, selects top-K
    ▼
[Stage 3] LLM Generation
    │  Generates an answer from the re-ranked context
    ▼
Answer
```

### Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- Python 3.10 or higher.

### Installation
1. Clone the repository and navigate to the directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
   Runtime deps include the `langchain-ollama` (default provider) and `langchain-openai` adapters; `requirements-dev.txt` adds pytest, ruff, and numpy for development.
3. Pull the required LLM:
   ```bash
   ollama pull llama3.2:1b
   ```

### Running the Project
```bash
python main.py
```

### HTTP API

Basalt ships a FastAPI service in `app.py`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

- `GET /health` — returns `{"status": "ok" | "degraded", "pipeline_initialized", "db_ready", "ollama_reachable"}` with status 200 when healthy and 503 when degraded. It only checks the database directory and Ollama reachability — it never loads models or initializes the pipeline. `db_ready` is true when the database directory exists at `BASALT_DB_PATH` (the Docker entrypoint creates and seeds it at boot; `python main.py` seeds it at startup), so a fresh deploy reports healthy without any `/ask`. `pipeline_initialized` is informational-only (true after the first `/ask`).
- `POST /ask` — body `{"query": "..."}` returns the same contract as `BasaltRAG.ask` (`{answer, source_documents}`). Ollama connection errors are returned as `"ERROR: Could not connect to Ollama (...)"` strings with status 200, matching the CLI behavior; other errors propagate as 500s.

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "What is RAG?"}'
```

### Configuration

`LangChainRAG.from_defaults()` (used by both `main.py` and `app.py`) resolves its settings from environment variables with explicit arguments taking precedence:

| Variable | Default |
|---|---|
| `BASALT_DB_PATH` | `./basalt_db` |
| `BASALT_RERANKER_MODEL` | `BAAI/bge-reranker-base` |
| `BASALT_LLM_MODEL` | `llama3.2:1b` |
| `BASALT_PROVIDER` | `ollama` |
| `BASALT_BASE_URL` | (unset) |
| `BASALT_N_RESULTS` | `10` |
| `BASALT_TOP_N` | `3` |

Precedence: explicit non-`None` argument > environment variable > default. `python main.py` honors all of them, e.g. `BASALT_BASE_URL=http://localhost:11434 python main.py`.

### LLM Provider Swapping

Basalt uses a `create_llm` factory to configure the LLM provider:

```python
from langchain_adapters import create_llm

llm = create_llm(provider="ollama", model_name="llama3.2:1b")
```

Pass a provider to `BasaltRAG` or `LangChainRAG.from_defaults()`:

```python
rag = BasaltRAG(provider="ollama", model="llama3.2:1b")
```

### Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

The test suite uses mocked ML dependencies (torch, sentence-transformers, chromadb) so tests run fast without GPU or model downloads. Tests cover:

- **Unit tests**: ingest, reranker, adapters, pipeline, eval metrics
- **E2E tests**: real `results.json` loaded through `EvalReporter` for analytics validation
- **Wiring tests**: full `BasaltRAG` ask flow with mocked pipeline

### Evaluation

```bash
python eval/run_eval.py
```

#### CLI Flags

| Flag | Description |
|---|---|
| `--k` | Number of top results to evaluate (default: 3) |
| `--retrieve-n` | Candidates to retrieve before re-ranking (default: 10) |
| `--dataset` | Path to the eval dataset JSON (default: `eval/eval_dataset.json`) |
| `--db-path` | Path to the ChromaDB directory (default: `eval/eval_db`, must be under `eval/`) |
| `--output` | Save results as JSON |
| `--export-csv` | Export summary and per_query CSVs via EvalReporter |
| `--compare` | Path to another results JSON for side-by-side comparison |
| `--keep-db` | Keep the ChromaDB database after evaluation |
| `--yes` | Confirm destructive removal of an existing database at `--db-path` (required if one exists) |

The first query is executed as a warm-up before latency timing begins, so cold-start costs do not pollute the reported latencies.

#### Evaluation Results

| Metric   | Retrieval Only | Retrieval + Re-rank |
| -------- | -------------- | ------------------- |
| Hit Rate | 100%           | 100%                |
| MRR      | 0.467          | 0.733 (+57%)        |
| Latency  | ~55 ms         | ~1.9 s (CPU)        |

The cross-encoder improved Mean Reciprocal Rank (MRR) by 57%, proving it is significantly better at putting the most factual document at the #1 spot for the LLM.

### Generation Evaluation

Evaluate answer faithfulness and relevance against a judge LLM:

```bash
python eval/run_generation_eval.py --db-path eval/eval_db
```

The database at `--db-path` must already contain the 46-document eval corpus from `eval/eval_dataset.json`. The command validates this and refuses to run against a database holding only the sample documents (raising a clear `ValueError`), so it can never silently score answers over the wrong corpus. Seed the corpus database once with `python eval/run_eval.py --keep-db --yes`.

Requires Ollama running. Uses a judge LLM (default `llama3.2:1b`) to score each generated answer: **faithfulness** measures whether every claim in the answer is supported by the retrieved context (1.0 supported / 0.0 unsupported), and **relevance** measures whether the answer addresses the question (1.0 complete / 0.0 no address). Rows where the judge returns unparseable output are flagged `judge_error: true`, excluded from the average scores, and counted in `num_judge_errors`.

| Flag | Description |
|---|---|
| `--dataset` | Eval dataset JSON (default: `eval/eval_dataset.json`) |
| `--output` | Results JSON (default: `eval/generation_results.json`, gitignored) |
| `--db-path` | ChromaDB directory (default: `./basalt_db`; must contain the eval corpus, see above) |
| `--n-results` | Candidates to retrieve (default: 10) |
| `--top-n` | Documents after re-ranking (default: 3) |
| `--provider` | Pipeline LLM provider (default: `ollama`) |
| `--llm-model` | Pipeline LLM model (default: `llama3.2:1b`) |
| `--judge-provider` | Judge LLM provider (default: `ollama`) |
| `--judge-model` | Judge LLM model (default: `llama3.2:1b`) |
| `--base-url` | Ollama base URL override (default: none) |

`GenerationReporter` (`eval/generation_analyzer.py`) mirrors `EvalReporter`'s API for analysis: `per_query_df`, `summary_df`, `worst_queries()`, and `export_csv()`.

### Eval Analytics

The `EvalReporter` class (`eval/analyzer.py`) uses pandas DataFrames for rich evaluation analysis:

```python
from eval.analyzer import EvalReporter

reporter = EvalReporter.from_file("eval/results.json")

# Per-query DataFrame with strategy labels
reporter.per_query_df

# Summary comparison
reporter.summary_df

# Latency percentiles
reporter.latency_percentiles("retrieval_only", quantiles=[0.5, 0.9, 0.99])

# Worst-performing queries
reporter.worst_queries("retrieval_plus_rerank", metric="mrr", n=5)

# Difficulty breakdown by precision buckets
reporter.difficulty_breakdown("retrieval_only")

# Compare two evaluation runs
reporter.compare(other_reporter)

# Export to CSV
reporter.export_csv("exports/", which="all")
```

### Deployment (Docker)

```bash
cp .env.example .env
docker compose up --build -d
```

Boot order: the Ollama container pulls `llama3.2:1b` before starting; the app container pre-pulls the Hugging Face models (`all-MiniLM-L6-v2` + `BAAI/bge-reranker-base`) into the `HF_HOME` volume, creates and seeds the database at `BASALT_DB_PATH` (`python -m scripts.seed_db`, idempotent — a no-op when the collection already has documents), then serves uvicorn. `/health` therefore reports `db_ready: true` and the healthcheck passes on a fresh deploy without any `/ask`; model weights still load lazily on the first `/ask`. For a GPU host, build the CUDA image instead: `docker build -f Dockerfile.gpu -t basalt-api-gpu .`. That image only accelerates if the container is granted the device at run time — `docker run --gpus all` (host needs nvidia-container-toolkit), or uncomment the `deploy.resources.reservations.devices` blocks in `docker-compose.yml` for both `basalt-api` and `ollama`; without this the app silently falls back to CPU.

### File Structure

```
Basalt/
├── main.py
├── app.py
├── ingest.py
├── reranker.py
├── rag_pipeline.py
├── langchain_adapters.py
├── sample_data.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── LICENSE
├── Dockerfile
├── Dockerfile.gpu
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml
├── scripts/
│   ├── __init__.py
│   ├── entrypoint.sh
│   ├── pre_pull.py
│   └── seed_db.py
│
├── eval/
│   ├── run_eval.py
│   ├── analyzer.py
│   ├── generation_eval.py
│   ├── generation_analyzer.py
│   ├── run_generation_eval.py
│   ├── eval_dataset.json
│   └── results.json
│
└── tests/
    ├── conftest.py
    ├── test_ingest.py
    ├── test_reranker.py
    ├── test_langchain_adapters.py
    ├── test_rag_pipeline.py
    ├── test_main.py
    ├── test_app.py
    ├── test_run_eval.py
    ├── test_analyzer.py
    ├── test_env_plumbing.py
    ├── test_generation_eval.py
    ├── test_generation_analyzer.py
    ├── test_run_generation_eval.py
    └── test_pre_pull.py
```
