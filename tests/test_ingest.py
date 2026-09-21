import os
from unittest.mock import MagicMock, patch

import pytest

from ingest import BasaltIngestor, _chunk_text, ensure_seeded


def make_mock_ingestor():
    return BasaltIngestor(db_path="test_db")


def test_add_documents_generate_default_ids():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    ingestor.add_documents(["doc1", "doc2"])
    ingestor.collection.upsert.assert_called_once_with(
        documents=["doc1", "doc2"],
        metadatas=None,
        ids=["id_0", "id_1"],
    )


def test_add_documents_with_custom_ids():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    ingestor.add_documents(["doc1", "doc2"], ids=["custom_id_1", "custom_id_2"])
    ingestor.collection.upsert.assert_called_once_with(
        documents=["doc1", "doc2"],
        metadatas=None,
        ids=["custom_id_1", "custom_id_2"],
    )


def test_add_documents_forwards_metadata_when_present():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    metadata = [{"source": "a"}, {"source": "b"}]
    ingestor.add_documents(["doc1", "doc2"], metadata_list=metadata)
    ingestor.collection.upsert.assert_called_once_with(
        documents=["doc1", "doc2"],
        metadatas=metadata,
        ids=["id_0", "id_1"],
    )


def test_search_returns_documents():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(
        return_value={"documents": [["doc1", "doc2"]], "ids": [["id_0", "id_1"]]}
    )

    results = ingestor.search("query")
    assert results == ["doc1", "doc2"]


def test_search_passes_query_texts_and_n_results():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(
        return_value={"documents": [["doc1"]], "ids": [["id_0"]]}
    )

    results = ingestor.search("query", n_results=5)

    ingestor.collection.query.assert_called_once_with(
        query_texts=["query"], n_results=5, where=None
    )
    assert results == ["doc1"]


def test_search_returns_empty_list_when_no_documents():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(return_value={"documents": [[]], "ids": [[]]})

    results = ingestor.search("query")
    assert results == []


def test_search_with_ids_returns_ids_and_documents():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(
        return_value={"documents": [["doc1", "doc2"]], "ids": [["id_0", "id_1"]]}
    )

    results = ingestor.search_with_ids("query")
    assert results == [("id_0", "doc1"), ("id_1", "doc2")]


def test_search_with_ids_passes_query_texts_and_n_results():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(
        return_value={"documents": [["doc1"]], "ids": [["id_0"]]}
    )

    results = ingestor.search_with_ids("query", n_results=4)

    ingestor.collection.query.assert_called_once_with(
        query_texts=["query"], n_results=4, where=None
    )
    assert results == [("id_0", "doc1")]


def test_search_with_ids_empty_results():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(return_value={"documents": [[]], "ids": [[]]})

    results = ingestor.search_with_ids("query")
    assert results == []


def test_search_passes_where_filter():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(
        return_value={"documents": [["doc1"]], "ids": [["id_0"]]}
    )

    ingestor.search("query", where={"source": "wiki"})
    ingestor.collection.query.assert_called_once_with(
        query_texts=["query"], n_results=10, where={"source": "wiki"}
    )


def test_search_with_ids_passes_where_filter():
    ingestor = make_mock_ingestor()
    ingestor.collection.query = MagicMock(
        return_value={"documents": [["doc1"]], "ids": [["id_0"]]}
    )

    ingestor.search_with_ids("query", where={"source": "wiki"})
    ingestor.collection.query.assert_called_once_with(
        query_texts=["query"], n_results=10, where={"source": "wiki"}
    )


def test_add_documents_with_chunking():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    long_text = "word " * 100
    ingestor.add_documents([long_text], chunk_size=20, chunk_overlap=5)

    add_kwargs = ingestor.collection.upsert.call_args
    documents = add_kwargs[1]["documents"]
    assert len(documents) > 1
    assert all(isinstance(d, str) for d in documents)


def test_add_documents_with_chunking_none():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    ingestor.add_documents(["short text"], chunk_size=None)
    ingestor.collection.upsert.assert_called_once_with(
        documents=["short text"],
        metadatas=None,
        ids=["id_0"],
    )


def test_add_documents_raises_when_overlap_exceeds_chunk_size():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    with pytest.raises(ValueError):
        ingestor.add_documents(["word " * 100], chunk_size=5, chunk_overlap=10)

    ingestor.collection.upsert.assert_not_called()


def test_add_documents_chunk_size_zero_disables_chunking():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    ingestor.add_documents(["word " * 100], chunk_size=0, chunk_overlap=10)
    ingestor.collection.upsert.assert_called_once_with(
        documents=["word " * 100],
        metadatas=None,
        ids=["id_0"],
    )


def test_add_documents_with_chunking_and_metadata():
    ingestor = make_mock_ingestor()
    ingestor.collection.upsert = MagicMock()

    long_text = "word " * 100
    metadata = [{"source": "test"}]
    ingestor.add_documents(
        [long_text], metadata_list=metadata, chunk_size=30, chunk_overlap=5
    )

    add_kwargs = ingestor.collection.upsert.call_args
    metadatas = add_kwargs[1]["metadatas"]
    assert metadatas is not None
    assert len(metadatas) > 1
    assert metadatas[0]["source"] == "test"
    assert "_chunk_index" in metadatas[0]
    assert "_parent_id" in metadatas[0]


def test_accepts_custom_embedding_model():
    ingestor = BasaltIngestor(
        db_path="test_db", embedding_model_name="all-mpnet-base-v2"
    )
    assert ingestor.emb_fn is not None
    assert ingestor.collection is not None


def test_chunk_text_splits_into_multiple_chunks():
    text = "word " * 100
    chunks = _chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_returns_single_for_short_text():
    chunks = _chunk_text("short text", chunk_size=512, chunk_overlap=64)
    assert chunks == ["short text"]


def test_chunk_text_raises_for_zero_chunk_size():
    with pytest.raises(ValueError):
        _chunk_text("some text", chunk_size=0, chunk_overlap=0)


def test_chunk_text_raises_when_overlap_equals_chunk_size():
    with pytest.raises(ValueError):
        _chunk_text("word " * 100, chunk_size=10, chunk_overlap=10)


def test_chunk_text_raises_when_overlap_exceeds_chunk_size():
    with pytest.raises(ValueError):
        _chunk_text("word " * 100, chunk_size=10, chunk_overlap=11)


def test_chunk_text_accepts_max_valid_overlap():
    chunks = _chunk_text("word " * 100, chunk_size=10, chunk_overlap=9)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)


def test_ensure_seeded_seeds_empty_collection():
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 0
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        seeded = ensure_seeded("test_db", ["doc1", "doc2"], ids=["id_1", "id_2"])
    assert seeded is True
    mock_ingestor.add_documents.assert_called_once_with(
        text_list=["doc1", "doc2"], ids=["id_1", "id_2"]
    )


def test_ensure_seeded_seeds_empty_collection_without_ids():
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 0
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        seeded = ensure_seeded("test_db", ["doc1"])
    assert seeded is True
    mock_ingestor.add_documents.assert_called_once_with(text_list=["doc1"], ids=None)


def test_ensure_seeded_noop_on_populated_collection():
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 2
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        seeded = ensure_seeded("test_db", ["doc1"])
    assert seeded is False
    mock_ingestor.add_documents.assert_not_called()


def test_ensure_seeded_clears_and_reingests_when_ids_given_and_populated():
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 2
    mock_ingestor.collection.get.return_value = {"ids": ["old_1", "old_2"]}
    with patch.dict(os.environ, {"BASALT_ALLOW_SEED_WIPE": "true"}):
        with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
            seeded = ensure_seeded("test_db", ["doc1"], ids=["new_1"])
    assert seeded is True
    mock_ingestor.collection.delete.assert_called_once_with(ids=["old_1", "old_2"])
    mock_ingestor.add_documents.assert_called_once_with(
        text_list=["doc1"], ids=["new_1"]
    )


def test_ensure_seeded_skips_delete_when_collection_has_no_existing_ids():
    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 2
    mock_ingestor.collection.get.return_value = {"ids": []}
    with patch.dict(os.environ, {"BASALT_ALLOW_SEED_WIPE": "true"}):
        with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
            seeded = ensure_seeded("test_db", ["doc1"], ids=["new_1"])
    assert seeded is True
    mock_ingestor.collection.delete.assert_not_called()
    mock_ingestor.add_documents.assert_called_once_with(
        text_list=["doc1"], ids=["new_1"]
    )
