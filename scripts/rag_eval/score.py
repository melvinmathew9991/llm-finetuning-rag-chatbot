"""Phase 2 of the RAG eval harness: score generate.py's output with ragas.

Runs in ragas' own isolated venv (see README.md in this directory) - ragas
pulls a langchain-core version incompatible with the app's pinned
langchain-chroma/langchain-ollama, so scoring is a separate process from
generation rather than a shared import.

Judge model defaults to OpenAI (ragas' own default, gpt-4o-mini - requires
OPENAI_API_KEY), or pass --judge-provider ollama to use the same local
OLLAMA_MODEL/OLLAMA_BASE_URL as the app, with local sentence-transformers
embeddings (no API key needed either way for embeddings).

Usage:
    python -m scripts.rag_eval.score [--judge-provider openai|ollama] [--results PATH] [--out PATH]
"""

import argparse
import json
import os

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

DEFAULT_RESULTS = "data/eval/results.json"
DEFAULT_OUTPUT = "data/eval/scores.json"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def to_ragas_dataset(records: list) -> Dataset:
    return Dataset.from_list(records)


def _local_embeddings():
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(model_name=model_name))


def build_judge(provider: str):
    """Returns (llm, embeddings) for ragas' evaluate(), or (None, None) to
    use ragas' own OpenAI defaults."""
    if provider == "openai":
        return None, None

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=os.environ.get("OLLAMA_MODEL", "llama3"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return LangchainLLMWrapper(llm), _local_embeddings()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-provider", choices=["openai", "ollama"], default="openai")
    args = parser.parse_args()

    load_dotenv()

    with open(args.results, encoding="utf-8") as f:
        records = json.load(f)

    dataset = to_ragas_dataset(records)
    llm, embeddings = build_judge(args.judge_provider)
    result = evaluate(dataset, metrics=METRICS, llm=llm, embeddings=embeddings)

    scores = result.to_pandas()
    per_question = scores.to_dict(orient="records")
    summary = {metric.name: float(result[metric.name]) for metric in METRICS}

    report = {"summary": summary, "per_question": per_question}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("Summary (mean across eval set):")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")
    print(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
