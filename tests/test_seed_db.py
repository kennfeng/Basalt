from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app import check_db, create_app
from scripts import seed_db


def test_main_seeds_sample_documents_with_default_path(monkeypatch):
    monkeypatch.delenv("BASALT_DB_PATH", raising=False)
    with patch("scripts.seed_db.ensure_seeded") as mock_ensure:
        seed_db.main()
    kwargs = mock_ensure.call_args.kwargs
    assert kwargs["db_path"] == "./basalt_db"
    assert len(kwargs["texts"]) == 9


def test_main_uses_basalt_db_path_env(monkeypatch):
    monkeypatch.setenv("BASALT_DB_PATH", "/data/basalt_db")
    with patch("scripts.seed_db.ensure_seeded") as mock_ensure:
        seed_db.main()
    assert mock_ensure.call_args.kwargs["db_path"] == "/data/basalt_db"


def test_main_idempotent_across_restarts():
    collection = MagicMock()
    collection.count.side_effect = [0, 2]
    ingestor = MagicMock()
    ingestor.collection = collection
    with patch("ingest.BasaltIngestor", return_value=ingestor):
        seed_db.main()
        seed_db.main()
    assert collection.count.call_count == 2
    ingestor.add_documents.assert_called_once()


def test_health_db_ready_after_seed(tmp_path, monkeypatch):
    db_path = str(tmp_path / "basalt_db")
    monkeypatch.setenv("BASALT_DB_PATH", db_path)
    monkeypatch.setattr("app.check_ollama", lambda base_url: True)

    def simulate_seed(
        db_path: str, texts: list[str], ids: list[str] | None = None
    ) -> bool:
        Path(db_path).mkdir(parents=True, exist_ok=True)
        return True

    with patch("scripts.seed_db.ensure_seeded", side_effect=simulate_seed):
        seed_db.main()

    assert check_db(db_path) is True
    resp = TestClient(create_app()).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["db_ready"] is True
    assert body["pipeline_initialized"] is False


def test_entrypoint_runs_prepull_seed_uvicorn_in_order():
    entrypoint = Path(__file__).resolve().parent.parent / "scripts" / "entrypoint.sh"
    content = entrypoint.read_text(encoding="utf-8")
    assert "set -e" in content
    prepull = content.index("pre_pull.py")
    seed = content.index("seed_db")
    uvicorn = content.index("uvicorn")
    assert prepull < seed < uvicorn
