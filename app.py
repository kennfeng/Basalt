import os
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from main import BasaltRAG


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


def check_db(db_path: str) -> bool:
    return Path(db_path).is_dir()


def check_ollama(base_url: str | None) -> bool:
    url = (base_url or "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        response = httpx.get(url, timeout=2.0)
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


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
    app.state.semaphore = Semaphore(int(os.getenv("BASALT_MAX_CONCURRENCY", "4")))
    app.state.ask_timeout = float(os.getenv("BASALT_ASK_TIMEOUT", "30"))

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
                from ingest import BasaltIngestor

                ingestor = BasaltIngestor(db_path=db_path)
                raw = ingestor.collection.count()
                db_count = int(raw) if isinstance(raw, int) else 0
            except Exception:
                db_count = 0
        status = "ok" if db_ready and ollama_reachable else "degraded"
        return JSONResponse(
            content={
                "status": status,
                "pipeline_initialized": pipeline_initialized,
                "db_ready": db_ready,
                "ollama_reachable": ollama_reachable,
                "db_count": int(db_count),
            },
            status_code=200 if status == "ok" else 503,
        )

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> dict:
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
            return rag.ask(body.query)
        finally:
            sem.release()

    return app


app = create_app()
