import json
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of RAG answers. Score the answer on two dimensions. "
    "Faithfulness: every claim in the answer is supported by the provided context "
    "(1.0 = fully supported, 0.0 = unsupported). Relevance: the answer addresses the "
    "user's question (1.0 = complete address, 0.0 = does not address it at all). "
    "Return only JSON with the keys faithfulness, relevance, and rationale."
)

JUDGE_HUMAN_TEMPLATE = (
    "Question: {query}\n\nAnswer: {answer}\n\nContext:\n{context}\n\n"
    "Return only a JSON object with keys faithfulness, relevance, and rationale."
)


def _extract_json_block(text: str) -> dict | None:
    # pull the first balanced json object from free-form model text
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_judge_response(text: str) -> dict | None:
    # validate judge output into faithfulness plus relevance scores
    trimmed = text.strip()
    try:
        data = json.loads(trimmed)
    except json.JSONDecodeError:
        data = _extract_json_block(trimmed)
    if not isinstance(data, dict):
        return None
    faithfulness = data.get("faithfulness")
    relevance = data.get("relevance")
    rationale = data.get("rationale")
    if (
        isinstance(faithfulness, (int, float))
        and not isinstance(faithfulness, bool)
        and 0.0 <= faithfulness <= 1.0
        and isinstance(relevance, (int, float))
        and not isinstance(relevance, bool)
        and 0.0 <= relevance <= 1.0
        and isinstance(rationale, str)
    ):
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "rationale": rationale,
        }
    return None


def build_context(source_documents: list[dict[str, Any]]) -> str:
    # join cited docs into one prompt context block
    # separate with blank lines to preserve doc boundaries
    return "\n\n".join(doc["document"] for doc in source_documents)


def summarize_generation(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    # average judge scores and latencies while counting failures
    # skip judge errors for means but report their count separately
    valid = [row for row in per_query if not row["judge_error"]]
    if valid:
        avg_faithfulness = float(sum(row["faithfulness"] for row in valid) / len(valid))
        avg_relevance = float(sum(row["relevance"] for row in valid) / len(valid))
    else:
        avg_faithfulness = 0.0
        avg_relevance = 0.0
    num_queries = len(per_query)
    num_judge_errors = len(per_query) - len(valid)
    if per_query:
        avg_generation_latency_ms = float(
            sum(row.get("generation_latency_ms", 0.0) for row in per_query)
            / len(per_query)
        )
        avg_judge_latency_ms = float(
            sum(row.get("judge_latency_ms", 0.0) for row in per_query) / len(per_query)
        )
    else:
        avg_generation_latency_ms = 0.0
        avg_judge_latency_ms = 0.0
    return {
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "num_queries": num_queries,
        "num_judge_errors": num_judge_errors,
        "avg_generation_latency_ms": avg_generation_latency_ms,
        "avg_judge_latency_ms": avg_judge_latency_ms,
    }


class LLMJudge:
    # score answers for support and relevance with a second model
    # hold prompt plus chain that asks for strict json scores
    # judge = llm reviewer
    def __init__(self, llm: Runnable) -> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", JUDGE_SYSTEM_PROMPT),
                ("human", JUDGE_HUMAN_TEMPLATE),
            ]
        )
        self._chain: Runnable | None = None

    @property
    def chain(self) -> Runnable:
        # build the judge prompt plus model plus text parser once
        # reuse for every query to avoid rebuilding the chain
        if self._chain is None:
            self._chain = (
                self.prompt
                | (lambda messages: self.llm.invoke(messages))
                | StrOutputParser()
            )
        return self._chain

    def judge(self, query: str, answer: str, context: str) -> dict[str, Any]:
        # score one answer against its sources with safe error shape
        # invoke chain, parse json, mark judge_error when unparsable
        response = self.chain.invoke(
            {"query": query, "answer": answer, "context": context}
        )
        parsed = parse_judge_response(str(response))
        if parsed is None:
            return {
                "faithfulness": None,
                "relevance": None,
                "rationale": None,
                "judge_error": True,
            }
        return {
            "faithfulness": parsed["faithfulness"],
            "relevance": parsed["relevance"],
            "rationale": parsed["rationale"],
            "judge_error": False,
        }


def run_generation_eval(
    pipeline: Any,
    judge: Any,
    dataset: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    # score every dataset answer for faithfulness and relevance
    # ask pipeline, build context, judge, track both latencies per row
    rows: list[dict[str, Any]] = []
    for item in dataset["queries"]:
        query = item["query"]
        start = time.perf_counter()
        result = pipeline.ask(query)
        generation_latency_ms = (time.perf_counter() - start) * 1000
        context = build_context(result["source_documents"])
        judge_start = time.perf_counter()
        judge_result = judge.judge(query, result["answer"], context)
        judge_latency_ms = (time.perf_counter() - judge_start) * 1000
        rows.append(
            {
                "query": query,
                "answer": result["answer"],
                "context": context,
                "source_ids": [doc["id"] for doc in result["source_documents"]],
                "faithfulness": judge_result["faithfulness"],
                "relevance": judge_result["relevance"],
                "rationale": judge_result["rationale"],
                "judge_error": judge_result["judge_error"],
                "generation_latency_ms": generation_latency_ms,
                "judge_latency_ms": judge_latency_ms,
            }
        )
    return {"config": config, "summary": summarize_generation(rows), "per_query": rows}
