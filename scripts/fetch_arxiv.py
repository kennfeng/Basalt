import argparse
import ipaddress
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

try:
    import defusedxml.ElementTree as DET  # type: ignore
except ImportError:
    DET = None  # type: ignore

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MAX_XML_BYTES = 10 * 1024 * 1024
_MAX_ENTRIES = 10000


def _parse_arxiv_id(raw_id: str) -> str:
    # extract the short arxiv id from a full entry url
    raw_id = raw_id.strip()
    if "/abs/" in raw_id:
        tail = raw_id.split("/abs/")[-1]
        return tail.split("v")[0]
    return raw_id


def _parse_year(published: str) -> int:
    # pull the 4-digit year from a published timestamp
    m = re.match(r"(\d{4})", published.strip())
    if m:
        return int(m.group(1))
    return 0


def _validate_url(url: str) -> None:
    # block unsafe fetch targets before any network call
    # allow http or https only, reject local and cloud metadata hosts
    # ssrf = call internal addresses
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must be http or https: {url}")
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"URL missing host: {url}")
    lowered = host.lower()
    if lowered in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise ValueError(f"URL host not allowed: {host}")
    if lowered == "169.254.169.254" or lowered == "metadata.google.internal":
        raise ValueError(f"URL host not allowed: {host}")
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"URL host not allowed: {host}")
    except ValueError as exc:
        if "not allowed" in str(exc):
            raise


def _parse_feed(xml_text: str) -> list[dict[str, Any]]:
    # turn an arxiv atom feed into a list of record dicts
    # enforce size and dtd limits, clean title and abstract spacing
    # atom = arxiv xml format, dtd = forbidden embedded markup rules
    if len(xml_text.encode("utf-8")) > _MAX_XML_BYTES:
        raise ValueError("XML response too large")
    if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        raise ValueError("DTD and entities are forbidden")
    if DET is not None:
        root = DET.fromstring(xml_text)
    else:
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
    # download up to n abstracts and store them as jsonl
    # validate url, fetch feed, trim to n, write one json per line
    if n < 1 or n > _MAX_ENTRIES:
        raise ValueError(f"n must be between 1 and {_MAX_ENTRIES}, got {n}")
    _validate_url(query_url)
    url = query_url
    if "max_results" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}max_results={n}"
    _validate_url(url)
    resp = httpx.get(url, timeout=30.0, follow_redirects=False)
    resp.raise_for_status()
    if len(resp.content) > _MAX_XML_BYTES:
        raise ValueError("Response too large")
    entries = _parse_feed(resp.text)
    entries = entries[:n]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return len(entries)


def main() -> None:
    # parse cli flags and fetch one arxiv slice to disk
    # support full query url or short category shorthand
    parser = argparse.ArgumentParser(description="Fetch arXiv abstracts to JSONL")
    parser.add_argument(
        "--query-url",
        default="https://export.arxiv.org/api/query?search_query=cat:cs.IR&sortBy=submittedDate",
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
        query_url = f"https://export.arxiv.org/api/query?search_query={args.query}&sortBy=submittedDate"
    out = Path(args.output)
    count = fetch_arxiv(query_url, n=args.n, output=out)
    print(f"Wrote {count} records to {out}")


if __name__ == "__main__":
    main()
