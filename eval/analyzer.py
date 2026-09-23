import json
from pathlib import Path

import pandas as pd

_METRIC_COLUMNS = (
    "mean_hit_rate",
    "mean_precision",
    "mean_mrr",
    "mean_latency_ms",
)


class EvalReporter:
    # turn raw eval json into tables for review and comparison
    # hold config plus per-query and summary frames built with pandas

    def __init__(self, results: dict):
        self._results = results
        self._config = results.get("config", {})
        self._summary_raw = results.get("summary", [])
        self._per_query = results.get("per_query", {})

        self._per_query_df = self._build_per_query_df()
        self._summary_df = self._build_summary_df()

    @classmethod
    def from_dict(cls, results: dict) -> "EvalReporter":
        # wrap an in-memory results dict without touching disk
        return cls(results)

    @classmethod
    def from_file(cls, path: Path) -> "EvalReporter":
        # load a saved results file or fail
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
    def per_query_df(self) -> pd.DataFrame:
        return self._per_query_df

    @property
    def summary_df(self) -> pd.DataFrame:
        return self._summary_df

    def _build_per_query_df(self) -> pd.DataFrame:
        # stack each strategy rows into one labeled frame
        frames = []
        for strategy, rows in self._per_query.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["strategy"] = strategy
            frames.append(df)
        if not frames:
            return pd.DataFrame(
                columns=[
                    "query",
                    "strategy",
                    "precision",
                    "hit_rate",
                    "mrr",
                    "latency_ms",
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def _build_summary_df(self) -> pd.DataFrame:
        # normalize summary dicts into mean metric columns
        # map avg names to mean names and keep run name plus k
        if not self._summary_raw:
            return pd.DataFrame()

        k = self._config.get("k")
        records = []
        for entry in self._summary_raw:
            records.append(
                {
                    "name": entry["name"],
                    "k": entry.get("k", k),
                    "mean_hit_rate": entry.get("avg_hit_rate", 0.0),
                    "mean_precision": entry.get("avg_precision", 0.0),
                    "mean_mrr": entry.get("avg_mrr", 0.0),
                    "mean_latency_ms": entry.get("avg_latency_ms", 0.0),
                }
            )
        return pd.DataFrame(records)

    def latency_percentiles(self, strategy: str, quantiles=None) -> pd.DataFrame:
        # show tail behavior for one strategy across chosen quantiles
        # filter rows then compute latency distribution points
        # quantile = share of runs below a latency value
        if quantiles is None:
            quantiles = [0.0, 0.25, 0.5, 0.75, 0.9]

        df = self._per_query_df
        subset = df[df["strategy"] == strategy]
        if subset.empty:
            raise ValueError(f"Unknown strategy: {strategy}")

        result = subset["latency_ms"].quantile(quantiles)
        return pd.DataFrame(
            {"quantile": quantiles, "latency_ms": [float(v) for v in result]}
        )

    def worst_queries(
        self, strategy: str, metric: str = "mrr", n: int = 5
    ) -> pd.DataFrame:
        # surface the lowest scoring queries for debugging
        # sort ascending by metric and return the first n rows
        # mrr = rank quality, low means relevant doc was buried
        df = self._per_query_df
        subset = df[df["strategy"] == strategy].copy()
        sorted_df = subset.sort_values(by=metric, ascending=True)
        return sorted_df.head(n)

    def difficulty_breakdown(self, strategy: str) -> pd.DataFrame:
        # count queries in low, medium, and high precision buckets
        # bin precision scores then tally rows per bucket
        df = self._per_query_df
        subset = df[df["strategy"] == strategy].copy()

        bins = [0.0, 0.33, 0.67, 1.0]
        labels = ["low", "medium", "high"]
        subset["precision_bucket"] = pd.cut(
            subset["precision"], bins=bins, labels=labels, include_lowest=True
        )
        grouped = (
            subset.groupby("precision_bucket", observed=False)
            .size()
            .reset_index(name="count")
        )
        return grouped

    def compare(self, other: "EvalReporter") -> pd.DataFrame:
        # show metric deltas between two runs for regression checks
        # join on name and k, delta is other minus baseline
        # positive delta means the compared run scored higher
        left = self.summary_df.copy()
        right = other.summary_df.copy()

        merged = left.merge(
            right, on=["name", "k"], how="outer", suffixes=("", "_right")
        )
        for metric in _METRIC_COLUMNS:
            merged[f"{metric}_delta"] = merged[f"{metric}_right"] - merged[metric]
            merged[metric] = merged[metric].fillna(merged[f"{metric}_right"])
        merged = merged.drop(columns=[f"{metric}_right" for metric in _METRIC_COLUMNS])
        return merged.reset_index(drop=True)

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
