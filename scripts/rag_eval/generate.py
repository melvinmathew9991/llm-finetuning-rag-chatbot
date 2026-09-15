"""Phase 1 of the RAG eval harness: run the real app pipeline over a fixed
eval set and record (question, answer, retrieved contexts, ground_truth).

Runs in the app's own venv (same deps as app/, no ragas involved) so the
answers/contexts come from the exact retrieval + generation path a user
would actually see. Scoring happens separately in score.py, in ragas' own
isolated venv - see README.md in this directory for why the split exists.

Usage:
    python -m scripts.rag_eval.generate [--eval-set PATH] [--out PATH]
"""

import argparse
import json

from dotenv import load_dotenv

from app.config import load_settings
from app.ingestion import ensure_default_kb
from app.rag import build_qa_chain, build_vector_db, get_answer

DEFAULT_EVAL_SET = "data/eval/kb_questions.json"
DEFAULT_OUTPUT = "data/eval/results.json"


def build_eval_records(eval_set: list, chain) -> list:
    """Runs each eval-set question through the real chain and shapes a
    ragas-ready record: question, answer, retrieved contexts, ground_truth.
    """
    records = []
    for item in eval_set:
        response = get_answer(item["question"], chain)
        records.append(
            {
                "question": item["question"],
                "answer": response["answer"],
                "contexts": [source["content"] for source in response["sources"]],
                "ground_truth": item["ground_truth"],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    load_dotenv()
    settings = load_settings()
    ensure_default_kb(settings.tmp_dir, settings.kb_source_dir)

    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)

    db = build_vector_db(settings)
    chain = build_qa_chain(settings, db)
    records = build_eval_records(eval_set, chain)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Wrote {len(records)} eval records to {args.out}")


if __name__ == "__main__":
    main()
