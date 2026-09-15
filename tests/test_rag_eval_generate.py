"""Tests for scripts/rag_eval/generate.py's record-shaping logic.

Stubs the chain per Rules.md (mock at the provider boundary) - this only
exercises build_eval_records, not the real generate/score pipeline, which
needs a live LLM and ragas' own isolated venv (see scripts/rag_eval/README.md).
"""

from scripts.rag_eval.generate import build_eval_records


class _StubChain:
    def __init__(self, answers_by_question: dict):
        self.answers_by_question = answers_by_question

    def invoke(self, inputs):
        question = inputs["input"]
        answer, contexts = self.answers_by_question[question]
        return {"answer": answer, "context": contexts}


class _FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


def test_build_eval_records_shapes_question_answer_contexts_and_ground_truth():
    eval_set = [
        {"question": "What sizes are available?", "ground_truth": "S, M, L, XL"},
    ]
    chain = _StubChain(
        {
            "What sizes are available?": (
                "S, M, L, and XL.",
                [_FakeDoc("Available in sizes S, M, L, XL.")],
            )
        }
    )

    records = build_eval_records(eval_set, chain)

    assert records == [
        {
            "question": "What sizes are available?",
            "answer": "S, M, L, and XL.",
            "contexts": ["Available in sizes S, M, L, XL."],
            "ground_truth": "S, M, L, XL",
        }
    ]


def test_build_eval_records_handles_multiple_questions_in_order():
    eval_set = [
        {"question": "Q1", "ground_truth": "GT1"},
        {"question": "Q2", "ground_truth": "GT2"},
    ]
    chain = _StubChain(
        {
            "Q1": ("A1", [_FakeDoc("ctx1")]),
            "Q2": ("A2", [_FakeDoc("ctx2a"), _FakeDoc("ctx2b")]),
        }
    )

    records = build_eval_records(eval_set, chain)

    assert [r["question"] for r in records] == ["Q1", "Q2"]
    assert records[1]["contexts"] == ["ctx2a", "ctx2b"]
