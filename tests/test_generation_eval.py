from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from eval import generation_eval


def test_parse_judge_response_valid_json():
    parsed = generation_eval.parse_judge_response(
        '{"faithfulness": 0.8, "relevance": 0.9, "rationale": "ok"}'
    )
    assert parsed == {"faithfulness": 0.8, "relevance": 0.9, "rationale": "ok"}


def test_parse_judge_response_markdown_fenced():
    text = '```json\n{"faithfulness": 0.8, "relevance": 0.9, "rationale": "ok"}\n```'
    parsed = generation_eval.parse_judge_response(text)
    assert parsed == {"faithfulness": 0.8, "relevance": 0.9, "rationale": "ok"}


def test_parse_judge_response_garbage_returns_none():
    assert generation_eval.parse_judge_response("I don't know") is None


def test_parse_judge_response_prose_wrapped_json_with_brace_in_rationale():
    text = 'Result: {"faithfulness": 0.9, "relevance": 0.8, "rationale": "a } b {c}"}'
    parsed = generation_eval.parse_judge_response(text)
    assert parsed == {
        "faithfulness": 0.9,
        "relevance": 0.8,
        "rationale": "a } b {c}",
    }


def test_parse_judge_response_prose_wrapped_json_with_escaped_quote_in_rationale():
    text = (
        'Note: {"faithfulness": 1.0, "relevance": 1.0, '
        '"rationale": "quotes \\" inside \\\\ ok"}'
    )
    parsed = generation_eval.parse_judge_response(text)
    assert parsed == {
        "faithfulness": 1.0,
        "relevance": 1.0,
        "rationale": 'quotes " inside \\ ok',
    }


def test_parse_judge_response_unterminated_block_returns_none():
    text = 'Result: {"faithfulness": 0.9, "relevance": 0.8, "rationale'
    assert generation_eval.parse_judge_response(text) is None


def test_parse_judge_response_out_of_range_score_returns_none():
    text = '{"faithfulness": 1.5, "relevance": 0.9, "rationale": "ok"}'
    assert generation_eval.parse_judge_response(text) is None


def test_parse_judge_response_missing_key_returns_none():
    text = '{"faithfulness": 0.8, "rationale": "ok"}'
    assert generation_eval.parse_judge_response(text) is None


def test_build_context_joins_documents_with_double_newline():
    docs = [{"document": "doc_a"}, {"document": "doc_b"}]
    assert generation_eval.build_context(docs) == "doc_a\n\ndoc_b"


def test_build_context_empty_returns_empty_string():
    assert generation_eval.build_context([]) == ""


def test_llm_judge_judge_returns_scores_for_valid_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"faithfulness": 0.8, "relevance": 0.9, "rationale": "ok"}'
    )
    judge = generation_eval.LLMJudge(mock_llm)
    result = judge.judge("query", "answer", "context")
    assert result == {
        "faithfulness": 0.8,
        "relevance": 0.9,
        "rationale": "ok",
        "judge_error": False,
    }


def test_llm_judge_judge_garbage_sets_judge_error():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "nope"
    judge = generation_eval.LLMJudge(mock_llm)
    result = judge.judge("query", "answer", "context")
    assert result["judge_error"] is True
    assert result["faithfulness"] is None
    assert result["relevance"] is None
    assert result["rationale"] is None


def test_llm_judge_chain_invoked_with_query_answer_context():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = (
        '{"faithfulness": 0.8, "relevance": 0.9, "rationale": "ok"}'
    )
    mock_llm = MagicMock()
    with patch.object(
        generation_eval.LLMJudge, "chain", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_chain
        judge = generation_eval.LLMJudge(mock_llm)
        result = judge.judge("the query", "the answer", "the context")
    mock_chain.invoke.assert_called_once_with(
        {"query": "the query", "answer": "the answer", "context": "the context"}
    )
    assert result["judge_error"] is False


class FakePipeline:
    def ask(self, query):
        return {
            "answer": f"answer for {query}",
            "source_documents": [
                {"id": "d1", "document": "doc one", "score": 0.9},
                {"id": "d2", "document": "doc two", "score": 0.8},
            ],
        }


class FakeJudge:
    def __init__(self):
        self.calls = []

    def judge(self, query, answer, context):
        self.calls.append((query, answer, context))
        return {
            "faithfulness": 0.9,
            "relevance": 0.8,
            "rationale": "ok",
            "judge_error": False,
        }


class ScriptedJudge:
    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    def judge(self, query, answer, context):
        return self._verdicts.pop(0)


def test_run_generation_eval_returns_schema_keys():
    pipeline = FakePipeline()
    judge = FakeJudge()
    dataset = {"queries": [{"query": "q1"}, {"query": "q2"}]}
    config = {"n_results": 10, "top_n": 3}
    result = generation_eval.run_generation_eval(pipeline, judge, dataset, config)
    assert set(result) == {"config", "summary", "per_query"}
    assert result["config"] == config
    assert set(result["summary"]) == {
        "avg_faithfulness",
        "avg_relevance",
        "num_queries",
        "num_judge_errors",
        "avg_generation_latency_ms",
        "avg_judge_latency_ms",
    }
    expected_row_keys = {
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
    }
    for row in result["per_query"]:
        assert set(row) == expected_row_keys


def test_run_generation_eval_builds_context_and_source_ids():
    pipeline = FakePipeline()
    judge = FakeJudge()
    dataset = {"queries": [{"query": "q1"}]}
    result = generation_eval.run_generation_eval(pipeline, judge, dataset, {})
    row = result["per_query"][0]
    assert row["query"] == "q1"
    assert row["answer"] == "answer for q1"
    assert row["context"] == "doc one\n\ndoc two"
    assert row["source_ids"] == ["d1", "d2"]


def test_run_generation_eval_means_exclude_judge_errors():
    pipeline = FakePipeline()
    judge = ScriptedJudge(
        [
            {
                "faithfulness": 1.0,
                "relevance": 1.0,
                "rationale": "good",
                "judge_error": False,
            },
            {
                "faithfulness": 0.0,
                "relevance": 0.5,
                "rationale": "ok",
                "judge_error": False,
            },
            {
                "faithfulness": None,
                "relevance": None,
                "rationale": None,
                "judge_error": True,
            },
        ]
    )
    dataset = {"queries": [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]}
    result = generation_eval.run_generation_eval(pipeline, judge, dataset, {})
    summary = result["summary"]
    assert summary["num_queries"] == 3
    assert summary["num_judge_errors"] == 1
    assert summary["avg_faithfulness"] == pytest.approx(0.5)
    assert summary["avg_relevance"] == pytest.approx(0.75)


def test_run_generation_eval_latencies_non_negative():
    pipeline = FakePipeline()
    judge = FakeJudge()
    dataset = {"queries": [{"query": "q1"}, {"query": "q2"}]}
    result = generation_eval.run_generation_eval(pipeline, judge, dataset, {})
    for row in result["per_query"]:
        assert row["generation_latency_ms"] >= 0.0
        assert row["judge_latency_ms"] >= 0.0


def test_summarize_generation_empty_returns_zeros():
    summary = generation_eval.summarize_generation([])
    assert summary["avg_faithfulness"] == 0.0
    assert summary["avg_relevance"] == 0.0
    assert summary["num_queries"] == 0
    assert summary["num_judge_errors"] == 0
    assert summary["avg_generation_latency_ms"] == 0.0
    assert summary["avg_judge_latency_ms"] == 0.0

    all_errors = [
        {
            "faithfulness": None,
            "relevance": None,
            "rationale": None,
            "judge_error": True,
        },
        {
            "faithfulness": None,
            "relevance": None,
            "rationale": None,
            "judge_error": True,
        },
    ]
    summary = generation_eval.summarize_generation(all_errors)
    assert summary["avg_faithfulness"] == 0.0
    assert summary["avg_relevance"] == 0.0
    assert summary["num_judge_errors"] == 2
