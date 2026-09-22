import json
import logging
import os
import secrets
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from main import BasaltRAG

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def check_db(db_path: str) -> bool:
    return Path(db_path).is_dir()


def check_ollama(base_url: str | None) -> bool:
    url = (base_url or "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=2.0)
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _parse_int_env(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def _parse_float_env(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = os.getenv("BASALT_API_KEY")
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_rag(request: Request) -> BasaltRAG:
    state = request.app.state
    if getattr(state, "rag", None) is None:
        with state.lock:
            if getattr(state, "rag", None) is None:
                if state.rag_factory is not None:
                    state.rag = state.rag_factory()
                else:
                    state.rag = BasaltRAG()
    return state.rag


def create_app(rag_factory: Any = None) -> FastAPI:
    app = FastAPI()
    app.state.rag = None
    app.state.rag_factory = rag_factory
    app.state.lock = Lock()
    app.state.semaphore = Semaphore(_parse_int_env("BASALT_MAX_CONCURRENCY", 4, 1, 32))
    app.state.ask_timeout = _parse_float_env("BASALT_ASK_TIMEOUT", 30.0, 1.0, 300.0)

    @app.get("/health")
    def health() -> JSONResponse:
        db_path = os.environ.get("BASALT_DB_PATH", "./basalt_db")
        base_url = os.environ.get("BASALT_BASE_URL")
        db_ready = check_db(db_path)
        ollama_reachable = check_ollama(base_url)
        pipeline_initialized = getattr(app.state, "rag", None) is not None
        db_count = 0
        if db_ready:
            try:
                import chromadb

                host = os.getenv("BASALT_CHROMA_HOST")
                if host:
                    port_raw = os.getenv("BASALT_CHROMA_PORT", "8000")
                    try:
                        port = int(port_raw) if port_raw else 8000
                    except ValueError:
                        port = 8000
                    client = chromadb.HttpClient(host=host, port=port)
                else:
                    client = chromadb.PersistentClient(path=db_path)
                try:
                    col = client.get_collection(name="documents")
                except Exception:  # noqa: BLE001
                    col = client.get_or_create_collection(name="documents")
                raw = col.count()
                db_count = int(raw) if isinstance(raw, int) else 0
            except Exception:  # noqa: BLE001
                db_count = 0
        hybrid_enabled = os.getenv("BASALT_HYBRID", "true").lower() not in (
            "false",
            "0",
            "no",
            "off",
        )
        bm25_ready = Path(db_path, "bm25.json").exists()
        dense_n = _parse_int_env("BASALT_HYBRID_DENSE_N", 50, 1, 100)
        sparse_n = _parse_int_env("BASALT_HYBRID_SPARSE_N", 50, 1, 100)
        rrf_k = _parse_int_env("BASALT_HYBRID_RRF_K", 60, 1, 200)
        hybrid_top_n = _parse_int_env("BASALT_HYBRID_TOP_N", 20, 1, 50)
        hnsw_m = _parse_int_env("BASALT_HNSW_M", 16, 1, 64)
        hnsw_con = _parse_int_env("BASALT_HNSW_CONSTRUCTION_EF", 200, 1, 1000)
        hnsw_search = _parse_int_env("BASALT_HNSW_SEARCH_EF", 10, 1, 1000)
        reranker_model = os.getenv("BASALT_RERANKER_MODEL", "BAAI/bge-reranker-base")
        llm_model = os.getenv("BASALT_LLM_MODEL", "llama3.2:1b")
        embed_model = os.getenv("BASALT_EMBED_MODEL", "all-MiniLM-L6-v2")
        status = "ok" if db_ready and ollama_reachable else "degraded"
        content: dict[str, Any] = {
            "status": status,
            "pipeline_initialized": pipeline_initialized,
            "db_ready": db_ready,
            "ollama_reachable": ollama_reachable,
            "db_count": int(db_count),
            "hybrid_enabled": hybrid_enabled,
            "bm25_ready": bm25_ready,
            "rrf_k": rrf_k,
            "hybrid_config": {
                "dense_n": dense_n,
                "sparse_n": sparse_n,
                "rrf_k": rrf_k,
                "top_n": hybrid_top_n,
            },
            "hnsw_config": {
                "m": hnsw_m,
                "construction_ef": hnsw_con,
                "search_ef": hnsw_search,
            },
            "models": {
                "reranker": reranker_model,
                "llm": llm_model,
                "embedding": embed_model,
            },
            "retrieval_pipeline": (
                "Hybrid BM25+dense RRF (k=60) -> Cross-Encoder bge-reranker -> LLM"
            ),
            "vector_db": "ChromaDB HNSW",
        }
        return JSONResponse(
            content=content,
            status_code=200 if status == "ok" else 503,
        )

    @app.post("/ask")
    def ask(
        body: AskRequest,
        request: Request,
        _auth: None = Depends(verify_api_key),
    ) -> Any:
        sem: Semaphore = request.app.state.semaphore
        timeout: float = request.app.state.ask_timeout
        acquired = sem.acquire(timeout=timeout)
        if not acquired:
            return JSONResponse(
                content={"detail": "Server busy, try again later"},
                status_code=503,
            )
        try:
            rag = get_rag(request)
            result = rag.ask(body.query)
            return JSONResponse(content=result)
        finally:
            sem.release()

    @app.post("/ask/stream")
    def ask_stream(
        body: AskRequest,
        request: Request,
        _auth: None = Depends(verify_api_key),
    ) -> Any:
        sem: Semaphore = request.app.state.semaphore
        timeout: float = request.app.state.ask_timeout
        acquired = sem.acquire(timeout=timeout)
        if not acquired:
            return JSONResponse(
                content={"detail": "Server busy, try again later"},
                status_code=503,
            )
        rag: Any = None
        query = body.query[:2000]
        pipeline: Any = None
        prepared: dict[str, Any] | None = None
        try:
            rag = get_rag(request)
            pipeline = getattr(rag, "pipeline", None)
            if pipeline is not None and hasattr(pipeline, "_prepare"):
                try:
                    prepared = pipeline._prepare(query)
                except Exception:  # noqa: BLE001
                    prepared = {"context": "", "source_documents": []}
        finally:
            sem.release()

        def generate():  # type: ignore[no-untyped-def]
            if pipeline is not None and prepared is not None:
                if not prepared.get("source_documents"):
                    payload = json.dumps(
                        {
                            "answer": (
                                "I couldn't find any relevant documents "
                                "in the database."
                            ),
                            "source_documents": [],
                            "done": True,
                        }
                    )
                    yield f"data: {payload}\n\n"
                    return
                context = prepared.get("context", "")
                source_docs = prepared.get("source_documents", [])
                chain = getattr(pipeline, "chain", None)
                if chain is not None:
                    try:
                        for chunk in chain.stream({"context": context, "query": query}):
                            if chunk:
                                yield f"data: {json.dumps({'token': chunk})}\n\n"
                    except (ConnectionError, httpx.TransportError):
                        logger.warning(
                            "ask_stream ollama connection error", exc_info=True
                        )
                        err = json.dumps(
                            {
                                "error": (
                                    "ERROR: Could not connect to Ollama. "
                                    "Please check that Ollama is running."
                                ),
                            }
                        )
                        yield f"data: {err}\n\n"
                        return
                    except Exception:  # noqa: BLE001
                        logger.exception("ask_stream error")
                        err = json.dumps({"error": "Internal server error"})
                        yield f"data: {err}\n\n"
                        return
                    done_payload = json.dumps(
                        {"done": True, "source_documents": source_docs}
                    )
                    yield f"data: {done_payload}\n\n"
                    return
            assert rag is not None
            result = rag.ask(query)
            answer = result.get("answer", "")
            for i in range(0, len(answer), 64):
                yield f"data: {json.dumps({'token': answer[i : i + 64]})}\n\n"
            final_payload = json.dumps(
                {
                    "done": True,
                    "source_documents": result.get("source_documents", []),
                }
            )
            yield f"data: {final_payload}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/scale_report")
    def scale_report() -> JSONResponse:
        report_path = Path("eval/scale_report.json")
        if not report_path.exists():
            return JSONResponse(
                content={
                    "detail": (
                        "Scale report not found. Run bench_scale --real to generate."
                    )
                },
                status_code=404,
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-cache",
                },
            )
        try:
            text = report_path.read_text(encoding="utf-8")
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("scale_report invalid json", exc_info=True)
            return JSONResponse(
                content={"detail": "Invalid scale report"},
                status_code=500,
                headers={
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("scale_report error")
            return JSONResponse(
                content={"detail": "Internal server error"},
                status_code=500,
                headers={
                    "X-Content-Type-Options": "nosniff",
                },
            )
        return JSONResponse(
            content=data,
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    if Path("ui").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

    return app


app = create_app()
