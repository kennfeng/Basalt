import json
import logging
from unittest.mock import MagicMock

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import check_db, check_ollama, create_app


class FakeRAG:
    def __init__(self, answer="fake answer", source_documents=None):
        self.answer = answer
        self.source_documents = source_documents or []

    def ask(self, query):
        return {"answer": self.answer, "source_documents": self.source_documents}


def test_health_ok_when_all_ready(monkeypatch):
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    monkeypatch.setattr("app.check_ollama", lambda base_url: True)
    monkeypatch.setattr("app.check_db", lambda db_path: True)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["pipeline_initialized"] is False
    assert body["db_ready"] is True
    assert body["ollama_reachable"] is True


def test_health_degraded_when_ollama_unreachable(monkeypatch):
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    monkeypatch.setattr("app.check_ollama", lambda base_url: False)
    monkeypatch.setattr("app.check_db", lambda db_path: True)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"


def test_health_degraded_when_db_missing(monkeypatch):
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    monkeypatch.setattr("app.check_ollama", lambda base_url: True)
    monkeypatch.setattr("app.check_db", lambda db_path: False)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"


def test_health_never_initializes_pipeline(monkeypatch):
    def factory():
        raise RuntimeError("pipeline should not be initialized during /health")

    client = TestClient(create_app(rag_factory=factory))
    monkeypatch.setattr("app.check_ollama", lambda base_url: True)
    monkeypatch.setattr("app.check_db", lambda db_path: True)
    resp = client.get("/health")
    assert resp.status_code == 200


def test_ask_returns_answer_and_source_documents(monkeypatch):
    rag = FakeRAG(
        answer="the answer",
        source_documents=[{"id": "d1", "document": "doc", "score": 0.9}],
    )
    client = TestClient(create_app(rag_factory=lambda: rag))
    monkeypatch.setattr("app.check_ollama", lambda base_url: True)
    monkeypatch.setattr("app.check_db", lambda db_path: True)
    resp = client.post("/ask", json={"query": "What is RAG?"})
    assert resp.status_code == 200
    assert resp.json() == {
        "answer": "the answer",
        "source_documents": [{"id": "d1", "document": "doc", "score": 0.9}],
    }


def test_ask_422_on_empty_query():
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    resp = client.post("/ask", json={"query": ""})
    assert resp.status_code == 422


def test_ask_422_on_missing_query():
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_error_string_passthrough_with_200():
    rag = FakeRAG(answer="ERROR: Could not connect to Ollama (ConnectionError: down)")
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask", json={"query": "q"})
    assert resp.status_code == 200
    assert resp.json()["answer"].startswith("ERROR: Could not connect to Ollama")


def test_ask_500_on_unexpected_exception():
    class ExplodingRAG:
        def ask(self, query):
            raise RuntimeError("boom")

    client = TestClient(
        create_app(rag_factory=lambda: ExplodingRAG()),
        raise_server_exceptions=False,
    )
    resp = client.post("/ask", json={"query": "q"})
    assert resp.status_code == 500


def test_rag_factory_called_once_across_two_asks():
    calls = []

    def factory():
        calls.append(1)
        return FakeRAG()

    client = TestClient(create_app(rag_factory=factory))
    client.post("/ask", json={"query": "q1"})
    client.post("/ask", json={"query": "q2"})
    assert len(calls) == 1


def test_create_app_returns_fastapi_instance():
    assert isinstance(create_app(), FastAPI)
    assert callable(check_db)
    assert callable(check_ollama)


def test_check_db_true_when_directory_exists(tmp_path):
    db = tmp_path / "basalt_db"
    db.mkdir()
    assert check_db(str(db)) is True


def test_check_db_false_when_directory_missing(tmp_path):
    assert check_db(str(tmp_path / "basalt_db")) is False


def test_check_db_false_for_plain_file(tmp_path):
    file = tmp_path / "basalt_db"
    file.write_text("not a db")
    assert check_db(str(file)) is False


def _collect_sse_payloads(resp) -> list[dict]:
    payloads: list[dict] = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            payloads.append(json.loads(line[6:]))
    return payloads


def _make_streaming_rag(chain_stream_effect=None, chain_stream_value=None):
    chain = MagicMock()
    if chain_stream_effect is not None:
        chain.stream.side_effect = chain_stream_effect
    elif chain_stream_value is not None:
        chain.stream.return_value = chain_stream_value
    else:
        chain.stream.return_value = iter(["hello"])
    pipeline = MagicMock()
    pipeline._prepare.return_value = {
        "context": "ctx",
        "source_documents": [{"id": "d1", "document": "doc", "score": 0.9}],
    }
    pipeline.chain = chain
    rag = MagicMock()
    rag.pipeline = pipeline
    return rag, chain, pipeline


def test_ask_stream_sanitizes_generic_error() -> None:
    rag, _, _ = _make_streaming_rag(
        chain_stream_effect=RuntimeError("boom /etc/passwd sensitive")
    )
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask/stream", json={"query": "hi"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("X-Trace-Id") is None
    payloads = _collect_sse_payloads(resp)
    assert len(payloads) == 1
    err = payloads[0]
    assert err["error"] == "Internal server error"
    assert "trace_id" not in err
    raw = json.dumps(err) + resp.text
    assert "boom" not in raw
    assert "/etc/passwd" not in raw
    assert "sensitive" not in raw
    assert "Traceback" not in raw
    assert "/app/" not in raw


def test_ask_stream_preserves_ollama_connection_error() -> None:
    rag, _, _ = _make_streaming_rag(chain_stream_effect=ConnectionError("down"))
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask/stream", json={"query": "hi"})
    assert resp.status_code == 200
    payloads = _collect_sse_payloads(resp)
    assert len(payloads) == 1
    err = payloads[0]
    assert err["error"] == (
        "ERROR: Could not connect to Ollama. Please check that Ollama is running."
    )
    assert "trace_id" not in err


def test_ask_stream_preserves_ollama_transport_error() -> None:
    rag, _, _ = _make_streaming_rag(
        chain_stream_effect=httpx.TransportError("transport boom")
    )
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask/stream", json={"query": "hi"})
    assert resp.status_code == 200
    payloads = _collect_sse_payloads(resp)
    assert len(payloads) == 1
    err = payloads[0]
    assert err["error"] == (
        "ERROR: Could not connect to Ollama. Please check that Ollama is running."
    )
    assert "trace_id" not in err


def test_ask_stream_logs_raw_error(caplog) -> None:
    caplog.set_level(logging.ERROR, logger="app")
    rag, _, _ = _make_streaming_rag(
        chain_stream_effect=RuntimeError("boom /etc/passwd sensitive")
    )
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask/stream", json={"query": "hi"})
    payloads = _collect_sse_payloads(resp)
    assert len(payloads) == 1
    assert "boom /etc/passwd sensitive" in caplog.text
    assert any(r.levelname == "ERROR" for r in caplog.records)


def test_ask_stream_headers_preserved_on_success() -> None:
    rag, _, _ = _make_streaming_rag(chain_stream_value=["hello", " world"])
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask/stream", json={"query": "hi"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("X-Trace-Id") is None
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("Cache-Control") == "no-cache"
    assert resp.headers.get("X-Accel-Buffering") == "no"
    payloads = _collect_sse_payloads(resp)
    tokens = [p.get("token") for p in payloads if "token" in p]
    assert "".join(tokens) == "hello world"
    assert payloads[-1].get("done") is True
    assert payloads[-1].get("source_documents") == [
        {"id": "d1", "document": "doc", "score": 0.9}
    ]


def test_ask_stream_headers_preserved_on_error() -> None:
    rag, _, _ = _make_streaming_rag(chain_stream_effect=ValueError("leak /etc/shadow"))
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask/stream", json={"query": "hi"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert resp.headers.get("X-Trace-Id") is None
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("Cache-Control") == "no-cache"
    assert resp.headers.get("X-Accel-Buffering") == "no"


def test_ask_stream_503_on_semaphore_timeout() -> None:
    from threading import Semaphore

    app_instance = create_app(rag_factory=lambda: FakeRAG())
    app_instance.state.semaphore = Semaphore(1)
    app_instance.state.ask_timeout = 0.05
    acquired = app_instance.state.semaphore.acquire()
    assert acquired is True
    try:
        client = TestClient(app_instance)
        resp = client.post("/ask/stream", json={"query": "hi"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Server busy, try again later"
        assert resp.headers.get("X-Trace-Id") is None
    finally:
        app_instance.state.semaphore.release()


def test_ask_503_on_semaphore_timeout() -> None:
    from threading import Semaphore

    app_instance = create_app(rag_factory=lambda: FakeRAG())
    app_instance.state.semaphore = Semaphore(1)
    app_instance.state.ask_timeout = 0.05
    acquired = app_instance.state.semaphore.acquire()
    assert acquired is True
    try:
        client = TestClient(app_instance)
        resp = client.post("/ask", json={"query": "hi"})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Server busy, try again later"
        assert resp.headers.get("X-Trace-Id") is None
    finally:
        app_instance.state.semaphore.release()


def test_ask_stream_no_leak_on_traceback_payload() -> None:
    err = RuntimeError("/app/secret Traceback")
    rag, _, _ = _make_streaming_rag(chain_stream_effect=err)
    client = TestClient(create_app(rag_factory=lambda: rag))
    resp = client.post("/ask/stream", json={"query": "hi"})
    payloads = _collect_sse_payloads(resp)
    err_payload = payloads[0]
    assert err_payload["error"] == "Internal server error"
    raw = json.dumps(payloads)
    assert "Traceback" not in raw
    assert "/app/" not in raw


def test_ask_still_returns_answer_after_stream_fix(monkeypatch) -> None:
    rag = FakeRAG(
        answer="the answer",
        source_documents=[{"id": "d1", "document": "doc", "score": 0.9}],
    )
    client = TestClient(create_app(rag_factory=lambda: rag))
    monkeypatch.setattr("app.check_ollama", lambda base_url: True)
    monkeypatch.setattr("app.check_db", lambda db_path: True)
    resp = client.post("/ask", json={"query": "What is RAG?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "the answer"
    assert resp.headers.get("X-Trace-Id") is None


def test_health_still_never_initializes_pipeline_after_stream_fix(monkeypatch) -> None:
    def factory():
        raise RuntimeError("pipeline should not be initialized during /health")

    client = TestClient(create_app(rag_factory=factory))
    monkeypatch.setattr("app.check_ollama", lambda base_url: True)
    monkeypatch.setattr("app.check_db", lambda db_path: True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("X-Trace-Id") is None


def _fake_scale_report_data() -> dict:
    return {
        "config": {
            "n_values": [100000],
            "retrieve_n": 10,
            "mode": "real",
            "thresholds": {
                "10k": {"p95_ms": 250.0, "p50_ms": 150.0},
                "100k": {"p95_ms": 500.0, "p50_ms": 300.0},
                "1M": {"p95_ms": 800.0, "p50_ms": 500.0},
            },
        },
        "results": [
            {
                "n": 100000,
                "docs_per_sec": 68.6,
                "disk_bytes": 345448740,
                "rss_bytes": 1491320832,
                "p50_ms": 12.66,
                "p95_ms": 31.6,
                "p99_ms": 31.6,
                "retrieve_n": 10,
                "slo_pass": True,
                "slo_reason": "pass",
            }
        ],
        "overall_slo_pass": True,
    }


def test_scale_report_returns_json_when_exists(monkeypatch) -> None:
    import pathlib

    data = _fake_scale_report_data()
    payload = json.dumps(data)
    orig_exists = pathlib.Path.exists
    orig_read_text = pathlib.Path.read_text

    def fake_exists(self: pathlib.Path) -> bool:
        if str(self).endswith("scale_report.json"):
            return True
        return orig_exists(self)

    def fake_read_text(
        self: pathlib.Path,
        encoding: str = "utf-8",
        errors: str | None = None,
    ) -> str:
        if str(self).endswith("scale_report.json"):
            return payload
        return orig_read_text(self, encoding=encoding, errors=errors)  # type: ignore[call-arg]

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    resp = client.get("/scale_report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["config"]["mode"] == "real"
    assert body["config"]["n_values"] == [100000]
    assert body["results"][0]["n"] == 100000
    assert body["results"][0]["docs_per_sec"] == 68.6
    assert body["results"][0]["p50_ms"] == 12.66
    assert body["results"][0]["p95_ms"] == 31.6
    assert body["overall_slo_pass"] is True
    assert resp.headers.get("X-Trace-Id") is None
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("Cache-Control") == "no-cache"
    assert "application/json" in resp.headers.get("content-type", "")


def test_scale_report_404_when_missing(monkeypatch) -> None:
    import pathlib

    orig_exists = pathlib.Path.exists

    def fake_exists(self: pathlib.Path) -> bool:
        if str(self).endswith("scale_report.json"):
            return False
        return orig_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    resp = client.get("/scale_report")
    assert resp.status_code == 404
    body = resp.json()
    assert "Scale report not found" in body["detail"]
    assert "trace_id" not in body
    assert resp.headers.get("X-Trace-Id") is None
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"


def test_scale_report_invalid_json_returns_500(monkeypatch) -> None:
    import pathlib

    orig_exists = pathlib.Path.exists

    def fake_exists(self: pathlib.Path) -> bool:
        if str(self).endswith("scale_report.json"):
            return True
        return orig_exists(self)

    def fake_read_text(
        self: pathlib.Path,
        encoding: str = "utf-8",
        errors: str | None = None,
    ) -> str:
        if str(self).endswith("scale_report.json"):
            return "not json {"
        return pathlib.Path.read_text(self)  # type: ignore[no-untyped-call]

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)
    client = TestClient(
        create_app(rag_factory=lambda: FakeRAG()),
        raise_server_exceptions=False,
    )
    resp = client.get("/scale_report")
    assert resp.status_code == 500
    assert resp.headers.get("X-Trace-Id") is None
    assert "Invalid scale report" in resp.json()["detail"]


def test_scale_report_never_initializes_pipeline(monkeypatch) -> None:
    import pathlib

    orig_exists = pathlib.Path.exists

    def fake_exists(self: pathlib.Path) -> bool:
        if str(self).endswith("scale_report.json"):
            return False
        return orig_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    def factory():
        raise RuntimeError("pipeline should not be initialized during /scale_report")

    client = TestClient(create_app(rag_factory=factory))
    resp = client.get("/scale_report")
    assert resp.status_code == 404


def test_scale_report_headers_on_success(monkeypatch) -> None:
    import pathlib

    data = _fake_scale_report_data()
    payload = json.dumps(data)
    orig_exists = pathlib.Path.exists
    orig_read_text = pathlib.Path.read_text

    def fake_exists(self: pathlib.Path) -> bool:
        if str(self).endswith("scale_report.json"):
            return True
        return orig_exists(self)

    def fake_read_text(
        self: pathlib.Path,
        encoding: str = "utf-8",
        errors: str | None = None,
    ) -> str:
        if str(self).endswith("scale_report.json"):
            return payload
        return orig_read_text(self, encoding=encoding, errors=errors)  # type: ignore[call-arg]

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)
    client = TestClient(create_app(rag_factory=lambda: FakeRAG()))
    resp = client.get("/scale_report")
    assert resp.status_code == 200
    assert resp.headers.get("Cache-Control") == "no-cache"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Trace-Id") is None
