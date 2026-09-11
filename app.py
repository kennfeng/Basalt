import os
import secrets
from pathlib import Path
from threading import Lock, Semaphore
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from main import BasaltRAG


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
                from ingest import BasaltIngestor

                ingestor = BasaltIngestor(db_path=db_path)
                raw = ingestor.collection.count()
                db_count = int(raw) if isinstance(raw, int) else 0
            except Exception:  # noqa: BLE001
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
    def ask(
        body: AskRequest,
        request: Request,
        _auth: None = Depends(verify_api_key),
    ) -> dict:
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
