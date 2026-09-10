import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_arxiv_id(raw_id: str) -> str:
    raw_id = raw_id.strip()
    if "/abs/" in raw_id:
        tail = raw_id.split("/abs/")[-1]
        return tail.split("v")[0]
    return raw_id


def _parse_year(published: str) -> int:
    m = re.match(r"(\d{4})", published.strip())
    if m:
        return int(m.group(1))
    return 0


def _parse_feed(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id_elem = entry.find("atom:id", ATOM_NS)
        raw_id = (
            raw_id_elem.text if raw_id_elem is not None and raw_id_elem.text else ""
        )
        arxiv_id = _parse_arxiv_id(raw_id)
        title_elem = entry.find("atom:title", ATOM_NS)
        title = (
            title_elem.text.strip().replace("\n", " ").strip()
            if title_elem is not None and title_elem.text
            else ""
        )
        title = re.sub(r"\s+", " ", title)
        summary_elem = entry.find("atom:summary", ATOM_NS)
        abstract = (
            summary_elem.text.strip().replace("\n", " ").strip()
            if summary_elem is not None and summary_elem.text
            else ""
        )
        abstract = re.sub(r"\s+", " ", abstract)
        published_elem = entry.find("atom:published", ATOM_NS)
        published = (
            published_elem.text
            if published_elem is not None and published_elem.text
            else ""
        )
        year = _parse_year(published)
        authors = []
        for author in entry.findall("atom:author", ATOM_NS):
            name_elem = author.find("atom:name", ATOM_NS)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        cat_elem = entry.find("atom:category", ATOM_NS)
        category = cat_elem.get("term", "") if cat_elem is not None else ""
        entries.append(
            {
                "id": arxiv_id,
                "abstract": abstract,
                "title": title,
                "authors": authors,
                "year": year,
                "category": category,
            }
        )
    return entries


def fetch_arxiv(query_url: str, n: int, output: Path) -> int:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    url = query_url
    if "max_results" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}max_results={n}"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    entries = _parse_feed(resp.text)
    entries = entries[:n]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv abstracts to JSONL")
    parser.add_argument(
        "--query-url",
        default="http://export.arxiv.org/api/query?search_query=cat:cs.IR&sortBy=submittedDate",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/sample.jsonl")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="shorthand for search_query, e.g. cat:cs.LG",
    )
    args = parser.parse_args()
    query_url = args.query_url
    if args.query:
        query_url = f"http://export.arxiv.org/api/query?search_query={args.query}&sortBy=submittedDate"
    out = Path(args.output)
    count = fetch_arxiv(query_url, n=args.n, output=out)
    print(f"Wrote {count} records to {out}")


if __name__ == "__main__":
    main()
