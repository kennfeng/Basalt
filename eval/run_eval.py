import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.analyzer import EvalReporter
from ingest import BasaltIngestor, ensure_seeded
from reranker import BasaltReRanker


def resolve_safe_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve()


def is_under_path(path: Path, base: Path) -> bool:
    return path.is_relative_to(base)


def validate_dataset_path(file_path: str, base_dir: Path) -> Path:
    dataset_path = resolve_safe_path(str(file_path))
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset path does not exist or is not a file: {dataset_path}"
        )
    if not is_under_path(dataset_path, base_dir):
        raise ValueError(f"Dataset path must be under {base_dir}: {dataset_path}")
    return dataset_path


def validate_db_path(db_path: str, base_dir: Path) -> Path:
    resolved = resolve_safe_path(str(db_path))
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Database path exists and is not a directory: {resolved}")
    if not is_under_path(resolved, base_dir):
        raise ValueError(f"Database path must be under {base_dir}: {resolved}")
    return resolved


def load_dataset(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / len(top_k)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    return 1.0 if any(rid in relevant_set for rid in top_k) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for idx, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0


def run_retrieval_only(
    ingestor: BasaltIngestor, query: str, n_results: int
) -> tuple[list[str], float]:
    start = time.perf_counter()
    candidates = ingestor.search_with_ids(query, n_results=n_results)

    elapsed_ms = (time.perf_counter() - start) * 1000
    retrieved_ids = [doc_id for doc_id, _ in candidates]

    return retrieved_ids, elapsed_ms


def run_retrieval_plus_rerank(
    ingestor: BasaltIngestor,
    ranker: BasaltReRanker,
    query: str,
    n_results: int,
    top_n: int,
) -> tuple[list[str], float]:
    start = time.perf_counter()
    candidates = ingestor.search_with_ids(query, n_results=n_results)
    reranked = ranker.rerank_with_ids(query, candidates, top_n=top_n)
    elapsed_ms = (time.perf_counter() - start) * 1000
    retrieved_ids = [r["id"] for r in reranked]
    return retrieved_ids, elapsed_ms


def summarize(
    name: str, per_query_results: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    if not per_query_results:
        return {
            "name": name,
            "avg_hit_rate": 0.0,
            "avg_precision": 0.0,
            "avg_mrr": 0.0,
            "avg_latency_ms": 0.0,
            "k": k,
        }

    df = pd.DataFrame(per_query_results)
    avg_metrics = df.mean(numeric_only=True)

    return {
        "name": name,
        "avg_hit_rate": float(avg_metrics.get("hit_rate", 0.0)),
        "avg_precision": float(avg_metrics.get("precision", 0.0)),
        "avg_mrr": float(avg_metrics.get("mrr", 0.0)),
        "avg_latency_ms": float(avg_metrics.get("latency_ms", 0.0)),
        "k": k,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval vs retrieval+re-ranking"
    )
    parser.add_argument(
        "--dataset", default=Path(__file__).parent / "eval_dataset.json"
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of top results to evaluate"
    )
    parser.add_argument(
        "--retrieve-n",
        type=int,
        default=10,
        help="Number of candidates to retrieve before re-ranking",
    )
    parser.add_argument(
        "--db-path",
        default=Path(__file__).parent / "eval_db",
        help="Path to ChromaDB database directory",
    )
    parser.add_argument(
        "--output", default=None, help="Path to save evaluation results as JSON"
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Directory to export summary and per_query CSVs via EvalReporter",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to another results JSON to compare against (uses EvalReporter)",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Keep the ChromaDB database after evaluation",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive removal of an existing evaluation database",
    )
    args = parser.parse_args()

    if args.k < 1 or args.k > args.retrieve_n:
        raise ValueError(
            f"--k ({args.k}) must be at least 1 "
            f"and must not exceed --retrieve-n ({args.retrieve_n})"
        )

    eval_base_dir = Path(__file__).parent.resolve()
    args.dataset = validate_dataset_path(args.dataset, eval_base_dir)
    args.db_path = validate_db_path(args.db_path, eval_base_dir)

    data = load_dataset(args.dataset)
    corpus = data["corpus"]
    queries = data["queries"]

    db_exists = args.db_path.exists()
    if db_exists and not args.keep_db:
        if not args.yes:
            raise ValueError(
                f"Refusing to remove existing database at {args.db_path}."
                " Pass --yes to confirm destructive action."
            )
        print(f"Removing existing database at {args.db_path}...")
        shutil.rmtree(args.db_path)

    print("Initializing Ingestor and ingesting corpus...")
    ingestor = BasaltIngestor(db_path=args.db_path)
    if db_exists and args.keep_db:
        print(f"Clearing existing collection at {args.db_path}...")
    ensure_seeded(
        db_path=str(args.db_path),
        texts=[doc["text"] for doc in corpus],
        ids=[doc["id"] for doc in corpus],
        ingestor=ingestor,
    )

    print("Loading cross-encoder model for re-ranking...")
    ranker = BasaltReRanker()

    if queries:
        print("Warming up retrieval and re-ranking...")
        run_retrieval_only(ingestor, queries[0]["query"], args.retrieve_n)
        run_retrieval_plus_rerank(
            ingestor, ranker, queries[0]["query"], args.retrieve_n, args.k
        )

    retrieval_results = []
    rerank_results = []

    print(
        f"\nRunning evaluation on {len(queries)} queries "
        f"with k={args.k} and retrieve_n={args.retrieve_n}...\n"
    )

    for item in queries:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        ids_only, latency_only = run_retrieval_only(ingestor, query, args.retrieve_n)
        ids_only_at_k = ids_only[: args.k]
        retrieval_results.append(
            {
                "query": query,
                "retrieved_ids": ids_only_at_k,
                "precision": precision_at_k(ids_only, relevant_ids, args.k),
                "hit_rate": hit_rate_at_k(ids_only, relevant_ids, args.k),
                "mrr": reciprocal_rank(ids_only_at_k, relevant_ids),
                "latency_ms": latency_only,
            }
        )

        ids_reranked, latency_rerank = run_retrieval_plus_rerank(
            ingestor, ranker, query, args.retrieve_n, args.k
        )
        rerank_results.append(
            {
                "query": query,
                "retrieved_ids": ids_reranked,
                "precision": precision_at_k(ids_reranked, relevant_ids, args.k),
                "hit_rate": hit_rate_at_k(ids_reranked, relevant_ids, args.k),
                "mrr": reciprocal_rank(ids_reranked[: args.k], relevant_ids),
                "latency_ms": latency_rerank,
            }
        )

        print(f"Query: {query}")
        print(f" Relevant IDs: {sorted(relevant_ids)}")
        print(f" Retrieval Only: {ids_only_at_k}")
        print(f" +Rerank: {ids_reranked}")
        print()

    summary_only = summarize("Retrieval Only", retrieval_results, args.k)
    summary_rerank = summarize("Retrieval + Re-rank", rerank_results, args.k)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print_table([summary_only, summary_rerank])

    full_results = {
        "config": {
            "k": args.k,
            "retrieve_n": args.retrieve_n,
            "num_queries": len(queries),
        },
        "summary": [summary_only, summary_rerank],
        "per_query": {
            "retrieval_only": retrieval_results,
            "retrieval_plus_rerank": rerank_results,
        },
    }

    reporter = EvalReporter.from_dict(full_results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nFull results written to {args.output}")

    if args.export_csv:
        export_dir = Path(args.export_csv)
        reporter.export_csv(export_dir, which="all")
        print(f"CSVs exported to {export_dir}")

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            raise FileNotFoundError(f"Comparison file not found: {compare_path}")
        other_reporter = EvalReporter.from_file(compare_path)
        comparison = reporter.compare(other_reporter)
        print("\n" + "=" * 60)
        print(f"Comparison with k={other_reporter.config.get('k')}")
        print("=" * 60)
        print(f"Baseline: {args.output or 'current run'}")
        print(f"Compared: {compare_path}")
        print("mean_*_delta = compared - baseline (positive means compared is better)")
        print(comparison.to_string(index=False))

    if not args.keep_db and os.path.exists(args.db_path):
        del ingestor
        del ranker

        gc.collect()

        try:
            shutil.rmtree(args.db_path)
        except PermissionError as e:
            print(f"\nWarning: could not remove {args.db_path} ({e}).")


if __name__ == "__main__":
    main()
