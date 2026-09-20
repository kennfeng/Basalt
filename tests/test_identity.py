import importlib
import os
import re
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_basalt_classes_importable():
    from ingest import BasaltIngestor
    from main import BasaltRAG
    from reranker import BasaltReRanker

    assert BasaltRAG is not None
    assert BasaltIngestor is not None
    assert BasaltReRanker is not None


def test_atlas_aliases_removed():
    for mod_name in ("main", "ingest", "reranker"):
        mod = importlib.import_module(mod_name)
        assert not hasattr(mod, "AtlasRAG"), f"{mod_name} still exports AtlasRAG"
        assert not hasattr(mod, "AtlasIngestor"), (
            f"{mod_name} still exports AtlasIngestor"
        )
        assert not hasattr(mod, "AtlasReRanker"), (
            f"{mod_name} still exports AtlasReRanker"
        )


def test_no_atlas_string_in_tracked_files():
    tracked_patterns = [
        "main.py",
        "app.py",
        "ingest.py",
        "reranker.py",
        "rag_pipeline.py",
        "langchain_adapters.py",
        "pyproject.toml",
        "docker-compose.yml",
        ".env.example",
        ".github/workflows/ci.yml",
    ]
    for rel in tracked_patterns:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text()
        assert not re.search(r"atlas", text, flags=re.IGNORECASE), (
            f"{rel} still contains atlas"
        )


def test_env_vars_are_basalt_prefixed():
    for var in (
        "BASALT_DB_PATH",
        "BASALT_RERANKER_MODEL",
        "BASALT_LLM_MODEL",
        "BASALT_PROVIDER",
        "BASALT_N_RESULTS",
        "BASALT_TOP_N",
        "BASALT_BASE_URL",
    ):
        assert var in (REPO_ROOT / "rag_pipeline.py").read_text(), (
            f"{var} missing in rag_pipeline.py"
        )
        assert var in (REPO_ROOT / ".env.example").read_text(), (
            f"{var} missing in .env.example"
        )

    assert "BASALT_DB_PATH" in (REPO_ROOT / "app.py").read_text()
    assert "BASALT_BASE_URL" in (REPO_ROOT / "app.py").read_text()
    assert "ATLAS_" not in (REPO_ROOT / "rag_pipeline.py").read_text()
    assert "ATLAS_" not in (REPO_ROOT / "app.py").read_text()


def test_basalt_env_plumbing():
    from rag_pipeline import LangChainRAG

    with (
        patch("rag_pipeline.BasaltIngestor") as mock_ingestor,
        patch("rag_pipeline.BasaltReRanker"),
        patch("rag_pipeline.create_llm"),
        patch.dict(os.environ, {"BASALT_DB_PATH": "/tmp/basalt_db"}, clear=False),
    ):
        mock_ingestor.return_value.collection.count.return_value = 1
        LangChainRAG.from_defaults()
        mock_ingestor.assert_called_with(db_path="/tmp/basalt_db")


def test_basalt_explicit_over_env():
    from rag_pipeline import LangChainRAG

    with (
        patch("rag_pipeline.BasaltIngestor") as mock_ingestor,
        patch("rag_pipeline.BasaltReRanker") as mock_ranker,
        patch("rag_pipeline.create_llm") as mock_llm,
        patch.dict(
            os.environ,
            {
                "BASALT_DB_PATH": "/env/db",
                "BASALT_RERANKER_MODEL": "env-model",
                "BASALT_LLM_MODEL": "env-llm",
                "BASALT_PROVIDER": "openai",
                "BASALT_N_RESULTS": "7",
                "BASALT_TOP_N": "2",
                "BASALT_BASE_URL": "http://env",
            },
        ),
    ):
        mock_ingestor.return_value.collection.count.return_value = 1
        pipeline = LangChainRAG.from_defaults(
            db_path="/arg/db",
            model_name="arg-reranker",
            llm_model_name="arg-llm",
            provider="ollama",
            n_results=0,
            top_n=5,
            base_url="http://arg",
        )
        mock_ingestor.assert_called_with(db_path="/arg/db")
        mock_ranker.assert_called_with(model_name="arg-reranker")
        mock_llm.assert_called_with(
            provider="ollama",
            model_name="arg-llm",
            temperature=0.0,
            base_url="http://arg",
        )
        assert pipeline.retriever.n_results == 0
        assert pipeline.reranker.top_n == 5


def test_app_health_uses_basalt_env(monkeypatch):
    from fastapi.testclient import TestClient

    from app import create_app

    monkeypatch.setenv("BASALT_DB_PATH", "/tmp/nonexistent_basalt")
    monkeypatch.delenv("ATLAS_DB_PATH", raising=False)

    app = create_app(
        rag_factory=lambda: type(
            "R", (), {"ask": lambda self, q: {"answer": "x", "source_documents": []}}
        )()
    )
    client = TestClient(app, raise_server_exceptions=False)
    with (
        patch("app.check_db", return_value=False) as mock_db,
        patch("app.check_ollama", return_value=False),
    ):
        resp = client.get("/health")
        mock_db.assert_called()
        assert resp.status_code == 503


def test_pyproject_and_compose_renamed():
    assert "basalt-rag" in (REPO_ROOT / "pyproject.toml").read_text().lower()
    compose = (REPO_ROOT / "docker-compose.yml").read_text()
    assert "basalt-api" in compose
    assert "basalt_db" in compose
    assert "BASALT_" in compose
