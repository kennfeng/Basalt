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
