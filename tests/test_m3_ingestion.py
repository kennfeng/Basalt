import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingest import BasaltIngestor, ensure_seeded


def make_ingestor(tmp_path: Path | None = None):
    path = str(tmp_path) if tmp_path else "test_db"
    return BasaltIngestor(db_path=path)


def test_add_documents_uses_upsert_and_batches():
    ingestor = make_ingestor()
    ingestor.collection.upsert = MagicMock()
    ingestor.collection.add = MagicMock()

    docs = [f"doc {i}" for i in range(10)]
    ids = [f"id_{i}" for i in range(10)]
    ingestor.add_documents(docs, ids=ids, batch_size=3)

    assert ingestor.collection.upsert.called
    assert not ingestor.collection.add.called
    assert ingestor.collection.upsert.call_count == 4
    first_call = ingestor.collection.upsert.call_args_list[0]
    assert first_call[1]["ids"] == ["id_0", "id_1", "id_2"]
    assert first_call[1]["documents"] == ["doc 0", "doc 1", "doc 2"]


def test_add_documents_batching_with_chunking():
    ingestor = make_ingestor()
    ingestor.collection.upsert = MagicMock()

    long_text = "word " * 100
    ingestor.add_documents([long_text], chunk_size=20, chunk_overlap=5, batch_size=5)
    assert ingestor.collection.upsert.called
    total_docs = sum(
        len(call[1]["documents"]) for call in ingestor.collection.upsert.call_args_list
    )
    assert total_docs > 1


def test_add_documents_default_batch_size_from_env(monkeypatch):
    monkeypatch.setenv("BASALT_BATCH_SIZE", "2")
    ingestor = make_ingestor()
    ingestor.collection.upsert = MagicMock()
    docs = ["a", "b", "c", "d"]
    ingestor.add_documents(docs, ids=["1", "2", "3", "4"])
    assert ingestor.collection.upsert.call_count == 2


def test_explicit_embedder_device_env(monkeypatch):
    monkeypatch.setenv("BASALT_EMBED_DEVICE", "cpu")
    monkeypatch.setenv("BASALT_EMBED_BATCH_SIZE", "16")
    ingestor = make_ingestor()
    assert ingestor is not None


def test_ensure_seeded_uses_count_only_no_get():
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 5
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        seeded = ensure_seeded("test_db", ["doc1"])
    assert seeded is False
    mock_ingestor.collection.get.assert_not_called()
    mock_ingestor.collection.delete.assert_not_called()
    mock_ingestor.add_documents.assert_not_called()


def test_ensure_seeded_respects_auto_seed_env(monkeypatch):
    monkeypatch.setenv("BASALT_AUTO_SEED", "false")
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 0
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        seeded = ensure_seeded("test_db", ["doc1"])
    assert seeded is False
    mock_ingestor.add_documents.assert_not_called()
    monkeypatch.delenv("BASALT_AUTO_SEED", raising=False)


def test_ensure_seeded_seeds_when_empty_and_auto_seed_true(monkeypatch):
    monkeypatch.delenv("BASALT_AUTO_SEED", raising=False)
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 0
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        seeded = ensure_seeded("test_db", ["doc1"], ids=["id1"])
    assert seeded is True
    mock_ingestor.add_documents.assert_called_once()


def test_bulk_ingest_script_exists():
    assert Path("scripts/bulk_ingest.py").exists()


def test_bulk_ingest_checkpoint_resume(tmp_path):
    from scripts.bulk_ingest import bulk_ingest

    corpus = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps(
            {
                "id": f"doc{i}",
                "abstract": f"abstract {i}",
                "title": f"T{i}",
                "authors": ["A"],
                "year": 2023,
                "category": "cs.IR",
            }
        )
        for i in range(6)
    ]
    corpus.write_text("\n".join(lines))
    checkpoint = tmp_path / "corpus.checkpoint"

    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 0
    mock_ingestor.collection.upsert = MagicMock()

    with patch("scripts.bulk_ingest.BasaltIngestor", return_value=mock_ingestor):
        count = bulk_ingest(
            source=corpus, batch_size=2, checkpoint=checkpoint, resume=False
        )
    assert count == 6
    assert mock_ingestor.collection.upsert.call_count == 3

    mock_ingestor2 = MagicMock()
    mock_ingestor2.collection.upsert = MagicMock()
    with patch("scripts.bulk_ingest.BasaltIngestor", return_value=mock_ingestor2):
        count2 = bulk_ingest(
            source=corpus, batch_size=2, checkpoint=checkpoint, resume=True
        )
    assert count2 == 0
    assert mock_ingestor2.collection.upsert.call_count == 0


def test_bulk_ingest_resume_after_partial(tmp_path):
    from scripts.bulk_ingest import bulk_ingest

    corpus = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps(
            {
                "id": f"doc{i}",
                "abstract": f"abstract {i} content for testing long enough words",
                "title": f"T{i}",
                "authors": ["A"],
                "year": 2023,
                "category": "cs.IR",
            }
        )
        for i in range(4)
    ]
    corpus.write_text("\n".join(lines))
    checkpoint = tmp_path / "chk"

    call_counts = []

    def fake_upsert(**kwargs):
        call_counts.append(len(kwargs.get("ids", [])))
        if len(call_counts) == 1:
            raise RuntimeError("simulated crash")

    mock_ingestor = MagicMock()
    mock_ingestor.collection.upsert.side_effect = fake_upsert
    with patch("scripts.bulk_ingest.BasaltIngestor", return_value=mock_ingestor):
        try:
            bulk_ingest(
                source=corpus, batch_size=2, checkpoint=checkpoint, resume=False
            )
        except RuntimeError:
            pass

    assert checkpoint.exists()
    offset = int(checkpoint.read_text())
    assert offset > 0

    mock_ingestor2 = MagicMock()
    mock_ingestor2.collection.upsert = MagicMock()
    with patch("scripts.bulk_ingest.BasaltIngestor", return_value=mock_ingestor2):
        count = bulk_ingest(
            source=corpus, batch_size=2, checkpoint=checkpoint, resume=True
        )
    assert count == 2
