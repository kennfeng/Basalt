from unittest.mock import MagicMock, patch

import httpx
import pytest

from main import BasaltRAG
from rag_pipeline import LangChainRAG


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.ask.return_value = {
        "answer": "This is a concise answer.",
        "source_documents": [
            {"id": "id_2", "document": "high", "score": 0.95},
            {"id": "id_3", "document": "mid", "score": 0.85},
        ],
    }
    return pipeline


def test_ask_returns_contract_keys_when_no_candidates(mock_pipeline):
    mock_pipeline.ask.return_value = {
        "answer": "I couldn't find any relevant documents in the database.",
        "source_documents": [],
    }
    rag = BasaltRAG(pipeline=mock_pipeline)
    result = rag.ask("nonexistent query")

    assert set(result) == {"answer", "source_documents"}
    assert result["source_documents"] == []


def test_ask_reranks_candidates_and_returns_llm_response(mock_pipeline):
    rag = BasaltRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert result["answer"] == "This is a concise answer."
    assert (
        result["source_documents"] == mock_pipeline.ask.return_value["source_documents"]
    )


def test_ask_returns_error_message_on_httpx_connect_error(mock_pipeline):
    mock_pipeline.ask.side_effect = httpx.ConnectError("Ollama unavailable")
    rag = BasaltRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert (
        result["answer"]
        == "ERROR: Could not connect to Ollama. Please check that Ollama is running."
    )
    assert result["source_documents"] == []


def test_ask_returns_error_message_on_builtin_connection_error(mock_pipeline):
    mock_pipeline.ask.side_effect = ConnectionError("Ollama unavailable")
    rag = BasaltRAG(pipeline=mock_pipeline)
    result = rag.ask("query")

    assert (
        result["answer"]
        == "ERROR: Could not connect to Ollama. Please check that Ollama is running."
    )
    assert result["source_documents"] == []


def test_ask_propagates_unexpected_errors(mock_pipeline):
    mock_pipeline.ask.side_effect = RuntimeError("bug in reranker")
    rag = BasaltRAG(pipeline=mock_pipeline)

    with pytest.raises(RuntimeError, match="bug in reranker"):
        rag.ask("query")


def test_init_stores_injected_pipeline():
    pipeline = MagicMock()
    rag = BasaltRAG(pipeline=pipeline)

    assert rag.pipeline is pipeline


def test_ask_forwards_to_pipeline(mock_pipeline):
    rag = BasaltRAG(pipeline=mock_pipeline)
    rag.ask("hello")
    mock_pipeline.ask.assert_called_once_with("hello")


def test_atlas_rag_forwards_base_url():
    mock_from_defaults = MagicMock()
    with patch.object(LangChainRAG, "from_defaults", mock_from_defaults):
        BasaltRAG(base_url="http://x")
    kwargs = mock_from_defaults.call_args.kwargs
    assert kwargs["base_url"] == "http://x"


def test_atlas_rag_defaults_are_env_resolvable():
    mock_from_defaults = MagicMock()
    with patch.object(LangChainRAG, "from_defaults", mock_from_defaults):
        BasaltRAG()
    kwargs = mock_from_defaults.call_args.kwargs
    assert kwargs["llm_model_name"] is None
    assert kwargs["provider"] is None
    assert kwargs["n_results"] is None
    assert kwargs["top_n"] is None
    assert kwargs["base_url"] is None
