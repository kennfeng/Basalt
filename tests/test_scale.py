import json
import os
import sys
from pathlib import Path

import pytest


def test_gen_corpus_script_exists_and_importable():
    p = Path("scripts/gen_corpus.py")
    assert p.exists(), "scripts/gen_corpus.py missing"
    spec = __import__("importlib.util").util.spec_from_file_location("gen_corpus", p)
    mod = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules["gen_corpus"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore
    finally:
        sys.modules.pop("gen_corpus", None)
    assert hasattr(mod, "gen_corpus") or hasattr(mod, "main")


def test_gen_corpus_generates_jsonl(tmp_path):
    from scripts.gen_corpus import gen_corpus

    out = tmp_path / "synth.jsonl"
    count = gen_corpus(n=5, output=out, seed=42)
    assert count == 5
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        obj = json.loads(line)
        for f in ("id", "abstract", "title", "authors", "year", "category"):
            assert f in obj


def test_bench_scale_script_exists_and_importable():
    p = Path("scripts/bench_scale.py")
    assert p.exists(), "scripts/bench_scale.py missing"
    spec = __import__("importlib.util").util.spec_from_file_location("bench_scale", p)
    mod = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules["bench_scale"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore
    finally:
        sys.modules.pop("bench_scale", None)
    assert hasattr(mod, "bench") or hasattr(mod, "main")


def test_bench_scale_produces_report(tmp_path):
    from scripts.bench_scale import bench

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"doc{i}",
                    "abstract": f"abstract number {i} about retrieval and ranking",
                    "title": f"Title {i}",
                    "authors": ["A"],
                    "year": 2023,
                    "category": "cs.IR",
                }
            )
            for i in range(10)
        )
    )
    report = tmp_path / "report.json"
    bench(corpus=corpus, output=report, n_values=[10], retrieve_n=5)
    assert report.exists()
    data = json.loads(report.read_text())
    assert "results" in data or "benchmarks" in data
    blob = data.get("results") or data.get("benchmarks")
    assert isinstance(blob, list) and len(blob) >= 1
    entry = blob[0]
    for key in (
        "n",
        "docs_per_sec",
        "disk_bytes",
        "rss_bytes",
        "p50_ms",
        "p95_ms",
        "p99_ms",
    ):
        assert key in entry, f"missing {key} in bench entry"
    assert entry["n"] == 10


def test_bench_scale_real_corpus_validation(tmp_path):
    from scripts.bench_scale import bench

    sample = Path("data/sample.jsonl")
    assert sample.exists(), "data/sample.jsonl missing for real-corpus validation"
    report = tmp_path / "report.json"
    bench(corpus=sample, output=report, n_values=[5], retrieve_n=5)
    assert report.exists()


def test_bench_scale_stub_warns_and_mode(tmp_path: Path) -> None:
    import warnings

    from scripts.bench_scale import bench

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"doc{i}",
                    "abstract": f"abstract number {i} about retrieval",
                    "title": f"Title {i}",
                    "authors": ["A"],
                    "year": 2023,
                    "category": "cs.IR",
                }
            )
            for i in range(4)
        )
    )
    report = tmp_path / "report.json"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = bench(corpus=corpus, output=report, n_values=[2], retrieve_n=5)
        assert any("stub" in str(x.message).lower() for x in w)
    assert data["config"]["mode"] == "stub"
    assert json.loads(report.read_text())["config"]["mode"] == "stub"


def test_bench_scale_real_mode_mocked(tmp_path: Path) -> None:
    import warnings
    from unittest.mock import MagicMock, patch

    from scripts.bench_scale import bench

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"doc{i}",
                    "abstract": f"abstract number {i} about retrieval",
                    "title": f"Title {i}",
                    "authors": ["A"],
                    "year": 2023,
                    "category": "cs.IR",
                }
            )
            for i in range(4)
        )
    )
    report = tmp_path / "report.json"
    mock_ingestor = MagicMock()
    mock_ingestor.add_documents.return_value = None
    mock_ingestor.search_with_ids.return_value = [("doc0", "abstract 0")]
    mock_ingestor.collection.query.return_value = {
        "ids": [["doc0"]],
        "documents": [["abstract 0"]],
    }
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = bench(
                corpus=corpus, output=report, n_values=[2], retrieve_n=5, real=True
            )
            assert not any("stub mode" in str(x.message).lower() for x in w)
    assert report.exists()
    assert data["config"]["mode"] == "real"
    blob = data.get("results") or data.get("benchmarks")
    assert isinstance(blob, list) and len(blob) >= 1
    entry = blob[0]
    for key in (
        "n",
        "docs_per_sec",
        "disk_bytes",
        "rss_bytes",
        "p50_ms",
        "p95_ms",
        "p99_ms",
    ):
        assert key in entry
    assert entry["n"] == 2
    assert mock_ingestor.add_documents.called
    called_kwargs = mock_ingestor.add_documents.call_args.kwargs
    called_args = mock_ingestor.add_documents.call_args.args
    batch_ok = called_kwargs.get("batch_size") == 512
    if not batch_ok and called_args:
        batch_ok = True
    assert batch_ok


def test_bench_scale_real_mode_db_path(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from scripts.bench_scale import bench

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
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
            for i in range(3)
        )
    )
    report = tmp_path / "report.json"
    custom = tmp_path / "custom_db"
    mock_ingestor = MagicMock()
    mock_ingestor.add_documents.return_value = None
    mock_ingestor.search_with_ids.return_value = [("doc0", "a")]
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor) as mp:
        bench(
            corpus=corpus,
            output=report,
            n_values=[2],
            retrieve_n=5,
            real=True,
            db_path=custom,
        )
        assert mp.called
        call_str = str(mp.call_args)
        assert str(custom) in call_str
    assert report.exists()
    assert json.loads(report.read_text())["config"]["mode"] == "real"


def test_analyzer_facet_extensions():
    from eval.analyzer import EvalReporter

    sample = {
        "config": {"k": 3, "retrieve_n": 10, "num_queries": 2},
        "summary": [
            {
                "name": "Retrieval Only",
                "avg_hit_rate": 1.0,
                "avg_precision": 0.5,
                "avg_mrr": 0.6,
                "avg_latency_ms": 10.0,
                "k": 3,
            },
            {
                "name": "Retrieval + Re-rank",
                "avg_hit_rate": 1.0,
                "avg_precision": 0.6,
                "avg_mrr": 0.7,
                "avg_latency_ms": 20.0,
                "k": 3,
            },
        ],
        "per_query": {
            "retrieval_only": [
                {
                    "query": "q1",
                    "retrieved_ids": ["a"],
                    "precision": 0.5,
                    "hit_rate": 1.0,
                    "mrr": 0.6,
                    "latency_ms": 10.0,
                },
                {
                    "query": "q2",
                    "retrieved_ids": ["b"],
                    "precision": 0.5,
                    "hit_rate": 1.0,
                    "mrr": 0.6,
                    "latency_ms": 12.0,
                },
            ],
            "retrieval_plus_rerank": [
                {
                    "query": "q1",
                    "retrieved_ids": ["a"],
                    "precision": 0.6,
                    "hit_rate": 1.0,
                    "mrr": 0.7,
                    "latency_ms": 20.0,
                },
                {
                    "query": "q2",
                    "retrieved_ids": ["b"],
                    "precision": 0.6,
                    "hit_rate": 1.0,
                    "mrr": 0.7,
                    "latency_ms": 22.0,
                },
            ],
        },
    }
    reporter = EvalReporter.from_dict(sample)
    assert hasattr(reporter, "latency_percentiles")
    assert hasattr(reporter, "difficulty_breakdown")
    assert hasattr(reporter, "worst_queries")
    perc = reporter.latency_percentiles("retrieval_only", quantiles=[0.5, 0.9])
    assert 0.5 in perc or len(perc) == 2


def test_hnsw_metadata_reflects_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ingest import _hnsw_metadata

    monkeypatch.setenv("BASALT_HNSW_M", "8")
    monkeypatch.setenv("BASALT_HNSW_CONSTRUCTION_EF", "100")
    monkeypatch.setenv("BASALT_HNSW_SEARCH_EF", "50")
    meta = _hnsw_metadata()
    assert meta["hnsw:space"] == "cosine"
    assert meta["hnsw:M"] == 8
    assert meta["hnsw:construction_ef"] == 100
    assert meta["hnsw:search_ef"] == 50


def test_hnsw_metadata_creation_time_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ingest import BasaltIngestor

    monkeypatch.setenv("BASALT_HNSW_M", "16")
    monkeypatch.setenv("BASALT_HNSW_CONSTRUCTION_EF", "200")
    monkeypatch.setenv("BASALT_HNSW_SEARCH_EF", "10")
    old = BasaltIngestor(db_path=str(tmp_path / "old_db"), collection_name="col_old")
    old_kwargs = old.client.get_or_create_collection.call_args.kwargs
    old_meta = dict(old_kwargs["metadata"])
    monkeypatch.setenv("BASALT_HNSW_M", "8")
    monkeypatch.setenv("BASALT_HNSW_SEARCH_EF", "50")
    new = BasaltIngestor(db_path=str(tmp_path / "new_db"), collection_name="col_new")
    new_kwargs = new.client.get_or_create_collection.call_args.kwargs
    new_meta = dict(new_kwargs["metadata"])
    assert old_meta.get("hnsw:M") == 16
    assert new_meta.get("hnsw:M") == 8
    assert new_meta.get("hnsw:search_ef") == 50
    assert old_meta.get("hnsw:M") == 16


def test_hnsw_metadata_docs_mention_retune() -> None:
    from ingest import _hnsw_metadata

    doc = _hnsw_metadata.__doc__ or ""
    lowered = doc.lower()
    assert "creation" in lowered
    assert "rm -rf" in doc or "fresh" in lowered


def test_bm25_save_large_requires_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from hybrid import BM25Index

    idx = BM25Index(corpus_ids=["a"], corpus_texts=["hello world"])
    target = tmp_path / "bm25.json"
    target.write_text("{}", encoding="utf-8")
    orig_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object) -> object:
        if "bm25.json" in str(self):
            mock = MagicMock()
            mock.st_size = 201 * 1024 * 1024
            mock.st_mode = 0o100644
            return mock
        return orig_stat(self, *args, **kwargs)  # type: ignore

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.delenv("BASALT_BM25_SHARDED", raising=False)
    monkeypatch.delenv("BASALT_BM25_TANTIVY", raising=False)
    with pytest.raises(RuntimeError, match="BASALT_BM25_SHARDED"):
        idx.save(target)
    assert target.read_text(encoding="utf-8") == "{}"


def test_bm25_save_large_with_sharded_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warnings
    from unittest.mock import MagicMock

    from hybrid import BM25Index

    idx = BM25Index(corpus_ids=["a"], corpus_texts=["hello world"])
    target = tmp_path / "bm25.json"
    orig_stat = Path.stat

    def fake_stat(self: Path, *args: object, **kwargs: object) -> object:
        if "bm25.json" in str(self):
            mock = MagicMock()
            mock.st_size = 201 * 1024 * 1024
            mock.st_mode = 0o100644
            return mock
        return orig_stat(self, *args, **kwargs)  # type: ignore

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setenv("BASALT_BM25_SHARDED", "true")
    monkeypatch.delenv("BASALT_BM25_TANTIVY", raising=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        idx.save(target)
        assert any(issubclass(x.category, UserWarning) for x in rec)
    assert target.exists()


def test_load_or_build_bm25_no_sample_when_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as js
    from unittest.mock import MagicMock, patch

    from hybrid import load_or_build_bm25

    jsonl = tmp_path / "sample.jsonl"
    jsonl.write_text(
        js.dumps({"id": "x", "abstract": "hello world"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "bm25.json"
    mock_collection = MagicMock()
    mock_collection.count.return_value = 5000
    mock_collection.get.return_value = {
        "ids": ["a", "b"],
        "documents": ["doc a", "doc b"],
    }
    with patch(
        "hybrid.BM25Index.build_from_jsonl",
        side_effect=AssertionError("sample must not open"),
    ) as mock_build:
        result = load_or_build_bm25(
            json_path=out,
            jsonl_path=jsonl,
            db_count=5000,
            collection=mock_collection,
        )
        mock_build.assert_not_called()
    assert mock_collection.get.called
    kwargs = mock_collection.get.call_args.kwargs
    assert kwargs.get("limit") == 10000
    assert "documents" in kwargs.get("include", [])
    assert "ids" in kwargs.get("include", [])
    assert result is not None
    assert result.corpus_ids == ["a", "b"]


def test_from_defaults_no_sample_when_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock, patch

    from rag_pipeline import LangChainRAG

    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 5000
    mock_ingestor.collection.get.return_value = {
        "ids": [],
        "documents": [],
    }
    with (
        patch("rag_pipeline.BasaltIngestor", return_value=mock_ingestor),
        patch("rag_pipeline.BasaltReRanker"),
        patch("rag_pipeline.create_llm"),
        patch("rag_pipeline.load_or_build_bm25") as mock_loader,
    ):
        mock_loader.return_value = None
        LangChainRAG.from_defaults(db_path=str(tmp_path / "db"))
        for call in mock_loader.call_args_list:
            blob = str(call.args) + str(call.kwargs)
            assert "sample.jsonl" not in blob


def test_from_defaults_paginated_get_limit(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from rag_pipeline import LangChainRAG

    mock_ingestor = MagicMock()
    mock_ingestor.collection.count.return_value = 5000
    mock_ingestor.collection.get.return_value = {
        "ids": [],
        "documents": [],
    }
    with (
        patch("rag_pipeline.BasaltIngestor", return_value=mock_ingestor),
        patch("rag_pipeline.BasaltReRanker"),
        patch("rag_pipeline.create_llm"),
    ):
        LangChainRAG.from_defaults(db_path=str(tmp_path / "db2"))
        assert mock_ingestor.collection.get.called
        found = False
        for call in mock_ingestor.collection.get.call_args_list:
            kwargs = call.kwargs
            if kwargs.get("limit") == 10000:
                assert "documents" in kwargs.get("include", [])
                assert "ids" in kwargs.get("include", [])
                found = True
        assert found


def test_slo_thresholds_frozen() -> None:
    from scripts.bench_scale import _threshold_for_n, _thresholds_config

    t10 = _threshold_for_n(10000)
    assert t10["p95"] == 250.0
    assert t10["p50"] == 150.0
    assert t10["rss"] == 1.2 * 1024 * 1024 * 1024
    assert t10["disk"] == 1.2 * 1024 * 1024 * 1024
    t100 = _threshold_for_n(100000)
    assert t100["p95"] == 500.0
    assert t100["p50"] == 300.0
    assert t100["rss"] == 2.5 * 1024 * 1024 * 1024
    assert t100["disk"] == 3.0 * 1024 * 1024 * 1024
    t1m = _threshold_for_n(1000000)
    assert t1m["p95"] == 800.0
    assert t1m["p50"] == 500.0
    assert t1m["rss"] == 6.0 * 1024 * 1024 * 1024
    assert t1m["disk"] == 8.0 * 1024 * 1024 * 1024
    cfg = _thresholds_config()
    assert cfg["10k"]["p95_ms"] == 250.0
    assert cfg["10k"]["p50_ms"] == 150.0
    assert cfg["100k"]["p95_ms"] == 500.0
    assert cfg["1M"]["p95_ms"] == 800.0


def test_bench_error_mode_no_overwrite_exit2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json as js
    import sys as sysmod

    from scripts.bench_scale import main

    report = tmp_path / "report.json"
    report.write_text(
        js.dumps(
            {
                "config": {"mode": "real"},
                "results": [],
                "overall_slo_pass": True,
            }
        ),
        encoding="utf-8",
    )
    before = report.read_text(encoding="utf-8")
    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(
        sysmod,
        "argv",
        [
            "bench_scale",
            "--corpus",
            str(missing),
            "--output",
            str(report),
            "--n-values",
            "10",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert report.read_text(encoding="utf-8") == before


def test_bench_real_cleans_tmp_db(tmp_path: Path) -> None:
    import json as js
    from unittest.mock import MagicMock, patch

    from scripts.bench_scale import bench

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            js.dumps({"id": f"doc{i}", "abstract": f"abstract {i}"}) for i in range(3)
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    mock_ingestor = MagicMock()
    mock_ingestor.add_documents.return_value = None
    mock_ingestor.search_with_ids.return_value = [("doc0", "a")]
    with patch("ingest.BasaltIngestor", return_value=mock_ingestor):
        bench(corpus=corpus, output=report, n_values=[2], real=True)
    leftovers = list(report.parent.glob("bench_chroma_*"))
    assert leftovers == []
    assert report.exists()


@pytest.mark.skipif(os.getenv("BASALT_NIGHTLY") != "1", reason="nightly 1M only")
def test_1m_nightly_thresholds_and_hnsw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ingest import _hnsw_metadata
    from scripts.bench_scale import _threshold_for_n

    thr = _threshold_for_n(1000000)
    assert thr["p95"] == 800.0
    assert thr["p50"] == 500.0
    assert thr["rss"] == 6.0 * 1024 * 1024 * 1024
    monkeypatch.setenv("BASALT_HNSW_M", "8")
    monkeypatch.setenv("BASALT_HNSW_SEARCH_EF", "50")
    meta = _hnsw_metadata()
    assert meta["hnsw:M"] == 8
    assert meta["hnsw:search_ef"] == 50
