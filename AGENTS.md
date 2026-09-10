# AGENTS.md

## Project

Basalt: local two-stage RAG — ChromaDB bi-encoder retrieval (`ingest.py`) -> `BAAI/bge-reranker-base` cross-encoder re-rank (`reranker.py`) -> Ollama LLM generation (`rag_pipeline.py`). `main.py` is the `BasaltRAG` CLI wrapper, `app.py` the FastAPI service (`/health`, `/ask`), `langchain_adapters.py` holds the `create_llm` factory and adapter classes, `eval/` holds `run_eval.py` + pandas-backed `EvalReporter` (`analyzer.py`) plus the generation-eval suite (`generation_eval.py`, `generation_analyzer.py`, `run_generation_eval.py`). Ops surface: `scripts/pre_pull.py` + `scripts/seed_db.py` + `scripts/entrypoint.sh`, `Dockerfile`/`Dockerfile.gpu`, `docker-compose.yml`, `.github/workflows/ci.yml`, `pyproject.toml`, `LICENSE`. Code style: fully type-annotated, no comments (commits enforce this), ruff-formatted.

## Commands

- Run app: `python main.py` — needs Ollama running with `ollama pull llama3.2:1b`. On startup `LangChainRAG.from_defaults` seeds `./basalt_db` (gitignored) with sample docs if the collection is empty (rag_pipeline.py:95)
- Serve API: `uvicorn app:app --host 0.0.0.0 --port 8000` — FastAPI wrapper; `/health` checks db + Ollama without loading models, `/ask` delegates to `BasaltRAG.ask`
- Tests: `python -m pytest tests/ -q` from repo root — 182 tests, ~30 s, no GPU/network needed
- Eval: `python eval/run_eval.py` — downloads bge-reranker-base from HF; refuses to reuse an existing `eval/eval_db` without `--yes` (deletes it first). Flags documented in README
- Generation eval: `python eval/run_generation_eval.py` — judge-LLM faithfulness/relevance scoring; needs Ollama; output `eval/generation_results.json` is gitignored. Flags documented in README
- Docker: `docker compose up --build -d` — app + Ollama containers, first-boot model pre-pull + idempotent DB seeding (entrypoint order: pre_pull → seed_db → uvicorn), so the `/health` healthcheck passes without a prior `/ask`; GPU variant via `Dockerfile.gpu`
- Lint/format: `ruff check --fix` + `ruff format`; pre-commit config mirrors both plus trailing-whitespace/eof/check-yaml etc. (`pre-commit run --all-files`)

## Testing quirks

- `tests/conftest.py` pre-injects MagicMocks for `torch`, `sentence_transformers`, `chromadb` (incl. `chromadb.utils`) into `sys.modules` — heavy deps never load. Any new heavy import must be added there or tests will attempt real downloads/model loads
- Tests do need real `langchain-core`, `pandas`, and `numpy` (numpy is declared in requirements-dev.txt; it also comes transitively via pandas/torch)
- `test_run_eval.py` E2E tests assert structural invariants of the committed `eval/results.json` (config k=3, retrieve_n=10, 15 queries, 30 per-query rows, both strategies, latency ordering, mrr improvement) — regenerating results.json with different settings breaks them
- `test_app.py` uses `TestClient(create_app(rag_factory=fake))` — the factory must be a zero-arg callable returning a duck-typed `.ask` object; `/health` must never call it
- `test_env_plumbing.py` patches `rag_pipeline.BasaltIngestor`/`BasaltReRanker`/`create_llm` and clears `BASALT_*` env vars to verify env resolution and precedence
- Generation-eval tests use fake pipelines/judges and synthetic `SAMPLE_GENERATION_RESULTS` dicts — no real LLM calls
- `eval/` is a package (`__init__.py`); tests import `from eval import run_eval` and `from eval import run_generation_eval`

## Gotchas

- `requirements.txt` declares all runtime deps, including `langchain-core` (imported directly by rag_pipeline.py/tests), `fastapi`/`uvicorn` (runtime for `app.py`), and the `langchain-ollama`/`langchain-openai` adapters (lazily imported by `create_llm`). Dev tools (`pytest`, `ruff`) and `numpy` (imported directly by tests) live in `requirements-dev.txt`; install both with `pip install -r requirements.txt -r requirements-dev.txt`
- `BasaltRAG.ask` swallows only connection-class exceptions (`ConnectionError`/`httpx.TransportError`) into the answer string `"ERROR: Could not connect to Ollama (...)"`; other errors propagate (main.py:33-41)
- On Windows, ChromaDB holds file handles in `eval/eval_db`; run_eval deletes it via `del ingestor; gc.collect()` then `rmtree`, catching PermissionError with a warning — a leftover DB is normal, not a bug
- Imports are repo-root-relative (conftest.py and run_eval.py insert repo root into `sys.path`); always run from repo root
- `eval/generation_results.json` is gitignored; `GenerationReporter.from_file` raises `FileNotFoundError` for missing files
- `.env.example` documents the `BASALT_*` variables; `cp .env.example .env` for Docker Compose (real `.env` is gitignored). Precedence is explicit arg > env > default — `from_defaults` uses `is not None` checks so explicit `0`/empty values win
