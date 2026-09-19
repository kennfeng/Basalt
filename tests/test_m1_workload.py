import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_fetch_arxiv_script_exists_and_importable():
    path = Path("scripts/fetch_arxiv.py")
    assert path.exists(), "scripts/fetch_arxiv.py missing"
    spec = __import__("importlib.util").util.spec_from_file_location(
        "fetch_arxiv", path
    )
    assert spec is not None
    mod = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules["fetch_arxiv"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore
    finally:
        sys.modules.pop("fetch_arxiv", None)
    assert hasattr(mod, "fetch_arxiv") or hasattr(mod, "main"), (
        "fetch_arxiv must expose fetch_arxiv() or main()"
    )


def test_fetch_arxiv_writes_jsonl_with_required_fields(tmp_path):
    from scripts.fetch_arxiv import fetch_arxiv

    sample_xml = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <published>2023-01-15T00:00:00Z</published>
    <title>Sample Title One</title>
    <summary>Sample abstract one about retrieval.</summary>
    <author><name>John Doe</name></author>
    <author><name>Jane Smith</name></author>
    <category term="cs.IR" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <published>2022-06-10T00:00:00Z</published>
    <title>Second Title</title>
    <summary>Second abstract content.</summary>
    <author><name>Alice</name></author>
    <category term="cs.CL" />
  </entry>
</feed>"""

    mock_resp = MagicMock()
    mock_resp.text = sample_xml
    mock_resp.raise_for_status = MagicMock()
    out = tmp_path / "out.jsonl"
    with patch("scripts.fetch_arxiv.httpx.get", return_value=mock_resp):
        count = fetch_arxiv(
            "http://export.arxiv.org/api/query?search_query=cat:cs.IR", n=2, output=out
        )
    assert count == 2
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        for field in ("id", "abstract", "title", "authors", "year", "category"):
            assert field in obj, f"missing {field}"
        assert isinstance(obj["authors"], list)
        assert isinstance(obj["year"], int)


def test_fetch_arxiv_respects_n_limit(tmp_path):
    from scripts.fetch_arxiv import fetch_arxiv

    sample_xml = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <published>2023-01-01T00:00:00Z</published>
    <title>T1</title><summary>A1</summary>
    <author><name>A</name></author><category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00002v1</id>
    <published>2023-01-01T00:00:00Z</published>
    <title>T2</title><summary>A2</summary>
    <author><name>B</name></author><category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2301.00003v1</id>
    <published>2023-01-01T00:00:00Z</published>
    <title>T3</title><summary>A3</summary>
    <author><name>C</name></author><category term="cs.AI"/>
  </entry>
</feed>"""
    mock_resp = MagicMock()
    mock_resp.text = sample_xml
    mock_resp.raise_for_status = MagicMock()
    out = tmp_path / "out.jsonl"
    with patch("scripts.fetch_arxiv.httpx.get", return_value=mock_resp):
        count = fetch_arxiv(
            "http://export.arxiv.org/api/query?search_query=cat:cs.AI", n=1, output=out
        )
    assert count == 1
    assert len(out.read_text().strip().splitlines()) == 1


def test_committed_demo_corpus_exists_and_valid():
    candidates = [
        Path("data/sample.jsonl"),
        Path("data/basalt_sample.jsonl"),
        Path("data/demo.jsonl"),
    ]
    exists = [p for p in candidates if p.exists()]
    assert exists, (
        "No committed demo corpus found under data/ (expected data/sample.jsonl)"
    )
    p = exists[0]
    lines = p.read_text().strip().splitlines()
    assert 10 <= len(lines) <= 500, f"demo corpus size {len(lines)} not in 10-500"
    for line in lines[:5]:
        obj = json.loads(line)
        for field in ("id", "abstract", "title", "authors", "year", "category"):
            assert field in obj


def test_demo_corpus_abstracts_are_short_chunks():
    p = Path("data/sample.jsonl")
    if not p.exists():
        p = next((x for x in Path("data").glob("*.jsonl") if x.exists()), None)
        assert p is not None
    for line in p.read_text().strip().splitlines():
        obj = json.loads(line)
        abstract = obj["abstract"]
        assert 20 <= len(abstract.split()) <= 500, (
            "abstract should be 20-500 words (single chunk at 512)"
        )
        assert len(abstract) > 0


def test_generation_eval_targets_literature():
    text = Path("eval/run_generation_eval.py").read_text()
    assert "faithfulness" in text.lower()
    assert "relevance" in text.lower()
