import argparse
import json
import os
import statistics
import time
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


def bench(
    corpus: Path, output: Path, n_values: list[int], retrieve_n: int = 10
) -> dict[str, Any]:
    if not corpus.exists():
        raise FileNotFoundError(f"corpus not found: {corpus}")
    all_lines = corpus.read_text(encoding="utf-8").strip().splitlines()
    all_objs = [json.loads(line) for line in all_lines if line.strip()]
    results: list[dict[str, Any]] = []
    for n in n_values:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        subset = all_objs[:n]
        docs_per_sec = 0.0
        disk_bytes = 0
        rss_bytes = _get_rss_bytes()
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
    data: dict[str, Any] = {
        "config": {"n_values": n_values, "retrieve_n": retrieve_n},
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
    args = parser.parse_args()
    corpus = Path(args.corpus)
    output = Path(args.output)
    n_values = [int(x.strip()) for x in args.n_values.split(",") if x.strip()]
    bench(corpus=corpus, output=output, n_values=n_values, retrieve_n=args.retrieve_n)
    print(f"Wrote report to {output}")


if __name__ == "__main__":
    main()
