import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval import run_generation_eval


class FakePipeline:
    def ask(self, query):
        return {
            "answer": f"answer for {query}",
            "source_documents": [
                {"id": "doc_0", "document": "PyTorch is a framework.", "score": 0.9}
            ],
        }


def patch_components(monkeypatch):
    fake_pipeline = FakePipeline()
    mock_from_defaults = MagicMock(return_value=fake_pipeline)
    monkeypatch.setattr(
        run_generation_eval.LangChainRAG, "from_defaults", mock_from_defaults
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"faithfulness": 0.9, "relevance": 0.8, "rationale": "ok"}'
    )
    mock_create_llm = MagicMock(return_value=mock_llm)
    monkeypatch.setattr(run_generation_eval, "create_llm", mock_create_llm)
    monkeypatch.setattr(run_generation_eval, "validate_corpus_db", MagicMock())
    return mock_from_defaults, mock_create_llm


def test_cli_writes_output_json_with_schema(tmp_path, monkeypatch):
    patch_components(monkeypatch)
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation_eval.py",
            "--dataset",
            "eval/eval_dataset.json",
            "--output",
            str(out),
        ],
    )
    run_generation_eval.main()
    assert out.exists()
    data = json.loads(out.read_text())
    assert set(data) == {"config", "summary", "per_query"}
    assert set(data["summary"]) == {
        "avg_faithfulness",
        "avg_relevance",
        "num_queries",
        "num_judge_errors",
        "avg_generation_latency_ms",
        "avg_judge_latency_ms",
    }
    assert isinstance(data["per_query"], list)
    assert len(data["per_query"]) >= 1


def test_cli_builds_pipeline_via_from_defaults(tmp_path, monkeypatch):
    mock_from_defaults, _ = patch_components(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation_eval.py",
            "--dataset",
            "eval/eval_dataset.json",
            "--output",
            str(tmp_path / "out.json"),
            "--db-path",
            "/data/db",
            "--n-results",
            "7",
            "--top-n",
            "2",
            "--provider",
            "ollama",
            "--llm-model",
            "llama3.2:1b",
        ],
    )
    run_generation_eval.main()
    kwargs = mock_from_defaults.call_args.kwargs
    assert kwargs["db_path"] == "/data/db"
    assert kwargs["n_results"] == 7
    assert kwargs["top_n"] == 2
    assert kwargs["provider"] == "ollama"
    assert kwargs["llm_model_name"] == "llama3.2:1b"


def test_cli_builds_judge_via_create_llm(tmp_path, monkeypatch):
    _, mock_create_llm = patch_components(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation_eval.py",
            "--dataset",
            "eval/eval_dataset.json",
            "--output",
            str(tmp_path / "out.json"),
            "--judge-provider",
            "openai",
            "--judge-model",
            "gpt-4o",
        ],
    )
    run_generation_eval.main()
    mock_create_llm.assert_called_once_with(provider="openai", model_name="gpt-4o")
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["per_query"][0]["faithfulness"] == pytest.approx(0.9)


def test_cli_dataset_outside_eval_raises(tmp_path, monkeypatch):
    patch_components(monkeypatch)
    outside = tmp_path / "dataset.json"
    outside.write_text('{"queries": []}')
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation_eval.py",
            "--dataset",
            str(outside),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        run_generation_eval.main()


def test_cli_flags_override_env(tmp_path, monkeypatch):
    mock_from_defaults, _ = patch_components(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation_eval.py",
            "--dataset",
            "eval/eval_dataset.json",
            "--output",
            str(tmp_path / "out.json"),
            "--llm-model",
            "arg-model",
            "--provider",
            "ollama",
            "--n-results",
            "5",
            "--top-n",
            "1",
            "--db-path",
            "/arg/db",
            "--base-url",
            "http://arg",
        ],
    )
    with patch.dict(
        os.environ,
        {
            "BASALT_LLM_MODEL": "env-model",
            "BASALT_PROVIDER": "openai",
            "BASALT_N_RESULTS": "7",
            "BASALT_TOP_N": "2",
            "BASALT_DB_PATH": "/env/db",
            "BASALT_BASE_URL": "http://env",
        },
    ):
        run_generation_eval.main()
    kwargs = mock_from_defaults.call_args.kwargs
    assert kwargs["llm_model_name"] == "arg-model"
    assert kwargs["provider"] == "ollama"
    assert kwargs["n_results"] == 5
    assert kwargs["top_n"] == 1
    assert kwargs["db_path"] == "/arg/db"
    assert kwargs["base_url"] == "http://arg"


def test_cli_default_output_path(monkeypatch):
    default_output = (
        Path(run_generation_eval.__file__).parent / "generation_results.json"
    )
    if default_output.exists():
        default_output.unlink()
    patch_components(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_generation_eval.py", "--dataset", "eval/eval_dataset.json"],
    )
    try:
        run_generation_eval.main()
        assert default_output.exists()
    finally:
        if default_output.exists():
            default_output.unlink()


def test_cli_records_reranker_model_from_env(tmp_path, monkeypatch):
    patch_components(monkeypatch)
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation_eval.py",
            "--dataset",
            "eval/eval_dataset.json",
            "--output",
            str(out),
        ],
    )
    with patch.dict(os.environ, {"BASALT_RERANKER_MODEL": "BAAI/bge-reranker-large"}):
        run_generation_eval.main()
    data = json.loads(out.read_text())
    assert data["config"]["reranker_model"] == "BAAI/bge-reranker-large"


def test_validate_corpus_db_passes_when_all_ids_present(monkeypatch):
    mock_ingestor = MagicMock()
    mock_ingestor.collection.get.return_value = {"ids": ["doc_0", "doc_1"]}
    monkeypatch.setattr(
        run_generation_eval, "BasaltIngestor", MagicMock(return_value=mock_ingestor)
    )
    data = {"corpus": [{"id": "doc_0"}, {"id": "doc_1"}]}
    run_generation_eval.validate_corpus_db("/data/db", data)
    mock_ingestor.collection.get.assert_called_once_with(ids=["doc_0", "doc_1"])


def test_validate_corpus_db_raises_on_missing_ids(monkeypatch):
    mock_ingestor = MagicMock()
    mock_ingestor.collection.get.return_value = {"ids": ["doc_0"]}
    monkeypatch.setattr(
        run_generation_eval, "BasaltIngestor", MagicMock(return_value=mock_ingestor)
    )
    data = {"corpus": [{"id": "doc_0"}, {"id": "doc_1"}]}
    with pytest.raises(ValueError, match="does not contain the eval corpus"):
        run_generation_eval.validate_corpus_db("/data/db", data)


def test_cli_refuses_default_db_missing_corpus(tmp_path, monkeypatch):
    fake_pipeline = FakePipeline()
    mock_from_defaults = MagicMock(return_value=fake_pipeline)
    monkeypatch.setattr(
        run_generation_eval.LangChainRAG, "from_defaults", mock_from_defaults
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"faithfulness": 0.9, "relevance": 0.8, "rationale": "ok"}'
    )
    monkeypatch.setattr(
        run_generation_eval, "create_llm", MagicMock(return_value=mock_llm)
    )
    mock_ingestor = MagicMock()
    mock_ingestor.collection.get.return_value = {"ids": []}
    monkeypatch.setattr(
        run_generation_eval, "BasaltIngestor", MagicMock(return_value=mock_ingestor)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generation_eval.py",
            "--dataset",
            "eval/eval_dataset.json",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(ValueError, match="does not contain the eval corpus"):
        run_generation_eval.main()
    mock_from_defaults.assert_not_called()
