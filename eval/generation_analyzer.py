import json
from pathlib import Path

import pandas as pd

_PER_QUERY_COLUMNS = (
    "query",
    "answer",
    "context",
    "source_ids",
    "faithfulness",
    "relevance",
    "rationale",
    "judge_error",
    "generation_latency_ms",
    "judge_latency_ms",
)

_SUMMARY_COLUMNS = (
    "avg_faithfulness",
    "avg_relevance",
    "num_queries",
    "num_judge_errors",
    "avg_generation_latency_ms",
    "avg_judge_latency_ms",
)


class GenerationReporter:
    # turn raw generation json into tables for review and comparison
    # hold config plus per-query and summary frames built with pandas
    def __init__(self, results: dict):
        self._results = results
        self._config = results.get("config", {})
        self._summary_raw = results.get("summary", {})
        self._per_query = results.get("per_query", [])

        self._per_query_df = self._build_per_query_df()
        self._summary_df = self._build_summary_df()

    @classmethod
    def from_dict(cls, results: dict) -> "GenerationReporter":
        # wrap an in-memory results dict without touching disk
        return cls(results)

    @classmethod
    def from_file(cls, path: Path) -> "GenerationReporter":
        # load a saved results file or fail with a clear path message
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @property
    def config(self) -> dict:
        return self._config

    @property
    def summary(self) -> dict:
        return self._summary_raw

    @property
    def per_query_df(self) -> pd.DataFrame:
        return self._per_query_df

    @property
    def summary_df(self) -> pd.DataFrame:
        return self._summary_df

    def _build_per_query_df(self) -> pd.DataFrame:
        # build one labeled frame from stored per-query rows
        if not self._per_query:
            return pd.DataFrame(columns=list(_PER_QUERY_COLUMNS))
        return pd.DataFrame(self._per_query)

    def _build_summary_df(self) -> pd.DataFrame:
        # normalize the summary dict into a one-row frame
        if not self._summary_raw:
            return pd.DataFrame(columns=list(_SUMMARY_COLUMNS))
        return pd.DataFrame([self._summary_raw])

    def worst_queries(self, metric: str = "faithfulness", n: int = 5) -> pd.DataFrame:
        # surface lowest scoring answers for debugging
        # sort ascending and return first n rows, nulls last
        # faithfulness = supported by context, low means hallucination risk
        df = self._per_query_df.copy()
        if df.empty:
            return df
        sorted_df = df.sort_values(by=metric, ascending=True, na_position="last")
        return sorted_df.head(n)

    def export_csv(self, path: Path, which: str = "all") -> None:
        path = Path(path)

        if which == "per_query":
            self._per_query_df.to_csv(path, index=False)
        elif which == "summary":
            self._summary_df.to_csv(path, index=False)
        elif which == "all":
            out_dir = path
            out_dir.mkdir(parents=True, exist_ok=True)
            self._summary_df.to_csv(out_dir / "summary.csv", index=False)
            self._per_query_df.to_csv(out_dir / "per_query.csv", index=False)
        else:
            raise ValueError(
                f"Unknown which: {which!r}. Use 'per_query', 'summary', or 'all'."
            )
