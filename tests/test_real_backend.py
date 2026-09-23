"""Real backend integration exercised only with BASALT_TEST_REAL equal 1.

Default CI stays mocked through tests conftest sys modules entries.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

requires_real_backend = pytest.mark.skipif(
    os.getenv("BASALT_TEST_REAL") != "1",
    reason="needs real chroma+embed",
)


def _run_real_script(
    script: str, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_real_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


@requires_real_backend
def test_real_ingest_upsert_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASALT_CHROMA_HOST", raising=False)
    monkeypatch.delenv("BASALT_CHROMA_PORT", raising=False)
    db_path = str(tmp_path / "real_db")
    script = textwrap.dedent(
        f"""
        from ingest import BasaltIngestor
        ing = BasaltIngestor(
            db_path={db_path!r},
            collection_name="real_upsert",
        )
        texts = [
            "retrieval augmented generation uses vectors",
            "chromadb stores embeddings for search",
            "reranker refines top candidates precisely",
        ]
        ids = ["real_a", "real_b", "real_c"]
        ing.add_documents(
            texts, ids=ids, chunk_size=None, batch_size=2
        )
        assert ing.collection.count() == 3
        hits = ing.search("vectors for search", n_results=2)
        assert len(hits) >= 1
        ing.add_documents(
            texts, ids=ids, chunk_size=None, batch_size=2
        )
        assert ing.collection.count() == 3
        print("ok upsert")
        """
    )
    result = _run_real_script(script)
    _assert_real_success(result)
    assert "ok upsert" in result.stdout


@requires_real_backend
def test_real_abstract_single_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASALT_CHROMA_HOST", raising=False)
    monkeypatch.delenv("BASALT_CHROMA_PORT", raising=False)
    db_path = str(tmp_path / "real_db")
    script = textwrap.dedent(
        f"""
        from ingest import BasaltIngestor
        ing = BasaltIngestor(
            db_path={db_path!r},
            collection_name="real_single",
        )
        texts = [
            "abstract one about hybrid retrieval fusion",
            "abstract two about cross encoder reranking",
        ]
        ing.add_documents(
            texts,
            ids=["single_a", "single_b"],
            chunk_size=None,
            batch_size=512,
        )
        assert ing.collection.count() == 2
        print("ok single")
        """
    )
    result = _run_real_script(script)
    _assert_real_success(result)
    assert "ok single" in result.stdout


@requires_real_backend
def test_real_ensure_seeded_empty_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASALT_CHROMA_HOST", raising=False)
    monkeypatch.delenv("BASALT_CHROMA_PORT", raising=False)
    monkeypatch.delenv("BASALT_AUTO_SEED", raising=False)
    db_path = str(tmp_path / "real_db")
    script = textwrap.dedent(
        f"""
        from ingest import BasaltIngestor, ensure_seeded
        ing = BasaltIngestor(
            db_path={db_path!r},
            collection_name="real_seeded",
        )
        texts = ["seed abstract one", "seed abstract two"]
        first = ensure_seeded(
            db_path={db_path!r},
            texts=texts,
            ids=["seed_a", "seed_b"],
            ingestor=ing,
        )
        assert first is True
        assert ing.collection.count() == 2
        second = ensure_seeded(
            db_path={db_path!r},
            texts=texts,
            ingestor=ing,
        )
        assert second is False
        assert ing.collection.count() == 2
        print("ok seeded")
        """
    )
    result = _run_real_script(script)
    _assert_real_success(result)
    assert "ok seeded" in result.stdout


@requires_real_backend
def test_real_ensure_seeded_ids_wipe_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASALT_CHROMA_HOST", raising=False)
    monkeypatch.delenv("BASALT_CHROMA_PORT", raising=False)
    monkeypatch.delenv("BASALT_AUTO_SEED", raising=False)
    monkeypatch.delenv("BASALT_ALLOW_SEED_WIPE", raising=False)
    db_path = str(tmp_path / "real_db")
    script = textwrap.dedent(
        f"""
        import os
        os.environ.pop("BASALT_ALLOW_SEED_WIPE", None)
        from ingest import BasaltIngestor, ensure_seeded
        ing = BasaltIngestor(
            db_path={db_path!r},
            collection_name="real_wipe",
        )
        first = ensure_seeded(
            db_path={db_path!r},
            texts=["wipe abstract one"],
            ids=["wipe_a"],
            ingestor=ing,
        )
        assert first is True
        try:
            ensure_seeded(
                db_path={db_path!r},
                texts=["wipe abstract two"],
                ids=["wipe_b"],
                ingestor=ing,
            )
        except RuntimeError as exc:
            assert "BASALT_ALLOW_SEED_WIPE" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
        os.environ["BASALT_ALLOW_SEED_WIPE"] = "true"
        ok = ensure_seeded(
            db_path={db_path!r},
            texts=["wipe abstract two"],
            ids=["wipe_b"],
            ingestor=ing,
        )
        assert ok is True
        assert ing.collection.count() == 1
        print("ok wipe")
        """
    )
    result = _run_real_script(script)
    _assert_real_success(result)
    assert "ok wipe" in result.stdout


@requires_real_backend
def test_real_persistent_client_and_env_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASALT_CHROMA_HOST", raising=False)
    monkeypatch.delenv("BASALT_CHROMA_PORT", raising=False)
    explicit_path = str(tmp_path / "real_explicit")
    env_path = str(tmp_path / "real_env")
    monkeypatch.setenv("BASALT_DB_PATH", env_path)
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        assert os.getenv("BASALT_CHROMA_HOST") in (None, "")
        from ingest import BasaltIngestor, _create_client
        client = _create_client({explicit_path!r})
        assert type(client).__module__.startswith("chromadb")
        assert "Http" not in type(client).__name__
        ing = BasaltIngestor(
            db_path={explicit_path!r},
            collection_name="real_prec",
        )
        assert ing.db_path == {explicit_path!r}
        assert ing.db_path != "./basalt_db"
        assert ing.collection.count() == 0
        assert Path({explicit_path!r}).exists()
        print("ok precedence")
        """
    )
    result = _run_real_script(script)
    _assert_real_success(result)
    assert "ok precedence" in result.stdout


@requires_real_backend
def test_real_rag_pipeline_with_mocked_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASALT_CHROMA_HOST", raising=False)
    monkeypatch.delenv("BASALT_CHROMA_PORT", raising=False)
    db_path = str(tmp_path / "real_db")
    script = textwrap.dedent(
        f"""
        from unittest.mock import MagicMock
        from langchain_core.runnables import RunnableLambda
        import rag_pipeline
        rag_pipeline.create_llm = MagicMock(
            return_value=RunnableLambda(lambda x: "mocked answer")
        )
        rag = rag_pipeline.LangChainRAG.from_defaults(
            db_path={db_path!r},
            sample_docs=[
                "RAG uses vectors for search",
                "Chroma stores embeddings locally",
            ],
        )
        out = rag.ask("what is RAG")
        assert out["answer"] == "mocked answer"
        assert len(out["source_documents"]) > 0
        print("ok pipeline")
        """
    )
    result = _run_real_script(script, timeout=600)
    _assert_real_success(result)
    assert "ok pipeline" in result.stdout
