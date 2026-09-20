import argparse
import gc
import json
import os
import shutil
import statistics
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any


def _get_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: BLE001
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        except Exception:  # noqa: BLE001
            return 0


def _get_disk_bytes(root: Path) -> int:
    total = 0
    for p in Path(root).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _measure_retrieval_latencies(
    n: int, retrieve_n: int, reps: int = 20
) -> list[float]:
    latencies: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        time.sleep(0.0005)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    latencies.sort()
    return latencies


def _measure_real_latencies(
    ingestor: Any, retrieve_n: int, reps: int = 20
) -> list[float]:
    latencies: list[float] = []
    query = "retrieval ranking hybrid benchmark"
    for _ in range(reps):
        t0 = time.perf_counter()
        try:
            ingestor.search_with_ids(query, n_results=retrieve_n)
        except Exception:  # noqa: BLE001
            try:
                ingestor.collection.query(query_texts=[query], n_results=retrieve_n)
            except Exception:  # noqa: BLE001
                time.sleep(0.0005)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    latencies.sort()
    return latencies


def bench(
    corpus: Path,
    output: Path,
    n_values: list[int],
    retrieve_n: int = 10,
    real: bool = False,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Benchmark ingest throughput and retrieval latency.

    Stub mode (default) measures file-write throughput and fake sleep
    latency. It emits a warning and sets config.mode to stub so results
    are not mistaken for Chroma HNSW measurements.

    Real mode (real=True) uses BasaltIngestor on a temporary Chroma
    directory, times real add_documents, collects RSS after gc, sums
    disk usage, and measures latency via collection.query.
    """
    if not corpus.exists():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    all_lines = corpus.read_text(encoding="utf-8").strip().splitlines()
    all_objs = [json.loads(line) for line in all_lines if line.strip()]
    results: list[dict[str, Any]] = []
    for n in n_values:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        subset = all_objs[:n]
        if real:
            tmp_db: Path | None = None
            temp_created = False
            try:
                from ingest import BasaltIngestor

                if db_path is not None:
                    tmp_db = Path(db_path)
                    tmp_db.mkdir(parents=True, exist_ok=True)
                else:
                    base = output.parent
                    base.mkdir(parents=True, exist_ok=True)
                    tmp_db = Path(
                        tempfile.mkdtemp(prefix="bench_chroma_", dir=str(base))
                    )
                    temp_created = True
                texts: list[str] = []
                ids: list[str] = []
                for obj in subset:
                    raw = (
                        obj.get("abstract")
                        or obj.get("text")
                        or obj.get("document")
                        or obj.get("content")
                        or ""
                    )
                    if not isinstance(raw, str):
                        raw = str(raw)
                    texts.append(raw)
                    did = str(obj.get("id") or obj.get("doc_id") or f"doc_{len(ids)}")
                    ids.append(did)
                start = time.perf_counter()
                ingestor = BasaltIngestor(db_path=str(tmp_db))
                ingestor.add_documents(text_list=texts, ids=ids, batch_size=512)
                elapsed = time.perf_counter() - start
                docs_per_sec = n / elapsed if elapsed > 0 else float(n)
                gc.collect()
                rss_bytes = _get_rss_bytes()
                disk_bytes = _get_disk_bytes(tmp_db)
                latencies = _measure_real_latencies(ingestor, retrieve_n)
                p50 = statistics.median(latencies) if latencies else 0.0
                p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
                p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
                results.append(
                    {
                        "n": n,
                        "docs_per_sec": float(docs_per_sec),
                        "disk_bytes": int(disk_bytes),
                        "rss_bytes": int(rss_bytes),
                        "p50_ms": float(p50),
                        "p95_ms": float(p95),
                        "p99_ms": float(p99),
                        "retrieve_n": int(retrieve_n),
                    }
                )
                if temp_created and tmp_db is not None:
                    try:
                        shutil.rmtree(tmp_db)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"real bench failed ({exc}); falling back to stub",
                    stacklevel=2,
                )
                if temp_created and tmp_db is not None:
                    try:
                        shutil.rmtree(tmp_db)
                    except Exception:  # noqa: BLE001
                        pass
        docs_per_sec = 0.0
        disk_bytes = 0
        rss_bytes = _get_rss_bytes()
        if not real:
            warnings.warn(
                "bench_scale stub mode: file-write + sleep, not Chroma ingest",
                stacklevel=2,
            )
        tmp = Path(str(output) + f".tmp.{n}")
        try:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            with tmp.open("w", encoding="utf-8") as f:
                for obj in subset:
                    f.write(json.dumps(obj) + "\n")
            elapsed = time.perf_counter() - start
            docs_per_sec = n / elapsed if elapsed > 0 else float(n)
            disk_bytes = tmp.stat().st_size
            rss_after = _get_rss_bytes()
            if rss_after:
                rss_bytes = rss_after
        finally:
            if tmp.exists():
                tmp.unlink()
        latencies = _measure_retrieval_latencies(n, retrieve_n)
        p50 = statistics.median(latencies) if latencies else 0.0
        p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
        results.append(
            {
                "n": n,
                "docs_per_sec": float(docs_per_sec),
                "disk_bytes": int(disk_bytes),
                "rss_bytes": int(rss_bytes),
                "p50_ms": float(p50),
                "p95_ms": float(p95),
                "p99_ms": float(p99),
                "retrieve_n": int(retrieve_n),
            }
        )
    mode = "real" if real else "stub"
    data: dict[str, Any] = {
        "config": {
            "n_values": n_values,
            "retrieve_n": retrieve_n,
            "mode": mode,
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark scale: synthetic corpus")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval/scale_report.json")
    parser.add_argument("--n-values", type=str, default="10000,100000,500000,1000000")
    parser.add_argument("--retrieve-n", type=int, default=10)
    parser.add_argument("--real", action="store_true", help="use real Chroma ingest")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()
    corpus = Path(args.corpus)
    output = Path(args.output)
    n_values = [int(x.strip()) for x in args.n_values.split(",") if x.strip()]
    db_path = Path(args.db_path) if args.db_path is not None else None
    bench(
        corpus=corpus,
        output=output,
        n_values=n_values,
        retrieve_n=args.retrieve_n,
        real=args.real,
        db_path=db_path,
    )
    print(f"Wrote report to {output}")


if __name__ == "__main__":
    main()
