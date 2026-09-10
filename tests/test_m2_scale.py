import json
import sys
from pathlib import Path


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
