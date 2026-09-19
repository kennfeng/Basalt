import argparse
import json
import random
from pathlib import Path


def gen_corpus(n: int, output: Path, seed: int = 42) -> int:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    random.seed(seed)
    categories = ["cs.IR", "cs.CL", "cs.LG", "cs.AI", "cs.CV"]
    titles = [
        "Hybrid Retrieval for Research Copilots",
        "Cross-Encoder Re-ranking at Scale",
        "Vector Search with HNSW Tuning",
        "Faithfulness in Grounded Generation",
        "Benchmarking Retrieval over ArXiv",
        "Latency Tradeoffs for Re-ranking",
        "Research Copilot over One Million Papers",
        "Efficient Ingestion with Checkpointing",
    ]
    abstracts = [
        (
            "We study hybrid retrieval combining dense embeddings and BM25 "
            "with reciprocal rank fusion. Cross-encoder re-ranking improves "
            "mean reciprocal rank. Experiments on synthetic abstracts validate "
            "throughput and memory."
        ),
        (
            "This work evaluates dense retrieval at scale. Our corpus contains "
            "abstract-level documents. We measure retrieval latency percentiles "
            "and disk usage. Results inform deployment on constrained instances."
        ),
        (
            "We present ingestion with batched upserts and byte-offset "
            "checkpoints. The method handles interruptions and avoids duplicate "
            "identifiers. Throughput scales linearly with batch size."
        ),
        (
            "We analyze HNSW parameters for vector search. Tuning M and "
            "efConstruction trades recall for build time. Our benchmark "
            "reports p50, p95, and p99 latencies."
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for i in range(n):
            cat = random.choice(categories)
            year = random.choice([2020, 2021, 2022, 2023, 2024])
            title = f"{random.choice(titles)} {i}"
            abstract = (
                random.choice(abstracts) + f" Synthetic document {i} for scale testing."
            )
            authors = [f"Author{i % 100}", f"CoAuthor{random.randint(0, 99)}"]
            obj = {
                "id": f"synth_{i:06d}",
                "abstract": abstract,
                "title": title,
                "authors": authors,
                "year": year,
                "category": cat,
            }
            f.write(json.dumps(obj) + "\n")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic corpus JSONL")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/synth.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    count = gen_corpus(n=args.n, output=out, seed=args.seed)
    print(f"Wrote {count} records to {out}")


if __name__ == "__main__":
    main()
