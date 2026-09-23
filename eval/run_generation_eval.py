import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import run_eval
from eval.generation_eval import LLMJudge, run_generation_eval
from ingest import BasaltIngestor
from langchain_adapters import create_llm
from rag_pipeline import LangChainRAG


def validate_corpus_db(db_path: str, data: dict[str, Any]) -> None:
    # stop generation eval when the db misses eval corpus docs
    # compare expected ids to stored ids and list first missing one
    # eval corpus = fixed test docs, not the demo sample docs
    expected_ids = [doc["id"] for doc in data["corpus"]]
    ingestor = BasaltIngestor(db_path=db_path)
    found = set(ingestor.collection.get(ids=expected_ids)["ids"])
    missing = sorted(doc_id for doc_id in expected_ids if doc_id not in found)
    if missing:
        raise ValueError(
            f"Database at {db_path} does not contain the eval corpus: "
            f"{len(missing)} of {len(expected_ids)} expected doc ids missing "
            f"(e.g. {missing[0]}). Generation evaluation must target the eval "
            "corpus, not the sample documents. Seed it with "
            "`python eval/run_eval.py --keep-db --yes` and pass "
            "--db-path eval/eval_db."
        )


def main() -> None:
    # run generation scoring and write summary plus per-query rows
    # build pipeline and judge, score all queries, save json report
    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation faithfulness and relevance"
    )
    parser.add_argument("--dataset", default="eval/eval_dataset.json")
    parser.add_argument("--output", default="eval/generation_results.json")
    parser.add_argument("--db-path", default="./basalt_db")
    parser.add_argument("--n-results", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--llm-model", default="llama3.2:1b")
    parser.add_argument("--judge-provider", default="ollama")
    parser.add_argument("--judge-model", default="llama3.2:1b")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    eval_base_dir = Path(__file__).parent.resolve()
    dataset_path = run_eval.validate_dataset_path(args.dataset, eval_base_dir)
    data = run_eval.load_dataset(dataset_path)

    validate_corpus_db(args.db_path, data)

    pipeline = LangChainRAG.from_defaults(
        db_path=args.db_path,
        provider=args.provider,
        llm_model_name=args.llm_model,
        n_results=args.n_results,
        top_n=args.top_n,
        base_url=args.base_url,
    )

    judge_kwargs: dict[str, Any] = {}
    if args.base_url is not None:
        judge_kwargs["base_url"] = args.base_url
    judge = LLMJudge(
        create_llm(
            provider=args.judge_provider,
            model_name=args.judge_model,
            **judge_kwargs,
        )
    )

    config_dict: dict[str, Any] = {
        "num_queries": len(data["queries"]),
        "provider": args.provider,
        "llm_model": args.llm_model,
        "reranker_model": os.environ.get(
            "BASALT_RERANKER_MODEL", "BAAI/bge-reranker-base"
        ),
        "n_results": args.n_results,
        "top_n": args.top_n,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "db_path": args.db_path,
    }

    results = run_generation_eval(pipeline, judge, data, config_dict)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_df = pd.DataFrame([results["summary"]])
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
