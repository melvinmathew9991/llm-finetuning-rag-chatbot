# RAG evaluation harness

Measures whether retrieval/chunking changes help or hurt, using
[ragas](https://github.com/explodinggradients/ragas) (faithfulness, answer
relevancy, context precision, context recall) against a small fixed set of
KB questions (`data/eval/kb_questions.json`).

## Why two scripts in two venvs

`ragas` pulls a `langchain-core` version that conflicts with this app's
pinned `langchain-chroma`/`langchain-ollama` (both require
`langchain-core<0.4`; recent `ragas` wants `>=1.0`, and even the older
`ragas==0.1.21` pulls `langchain-core==0.2.43`, which `langchain-ollama`
rejects). Installing both in one environment breaks one or the other.

So the harness is split into two independent steps that only communicate
through a JSON file:

1. **`generate.py`** runs in the app's own venv (`requirements-dev.txt`,
   i.e. the same environment `pytest` uses). It builds the real vector
   store + retrieval chain via `app.rag`, answers each eval-set question
   for real, and writes `(question, answer, contexts, ground_truth)` to
   `data/eval/results.json`. This is the exact retrieval/generation path a
   user of the Streamlit app would see.
2. **`score.py`** runs in a separate venv with `requirements-eval.txt`
   installed. It reads `results.json` and scores it with `ragas.evaluate`,
   writing `data/eval/scores.json`. It never imports anything from `app/`.

## Running it

```bash
# one-time setup
python -m venv .venv-eval
.venv-eval/Scripts/pip install -r requirements-eval.txt

# 1. generate real answers (app venv - the one requirements-dev.txt installs)
python -m scripts.rag_eval.generate

# 2. score them (ragas venv)
.venv-eval/Scripts/python -m scripts.rag_eval.score
```

Both scripts accept `--eval-set`/`--results`/`--out` to point at different
files, e.g. to compare two chunking configurations by generating twice into
different result files and scoring both.

Requires `OPENAI_API_KEY` in `.env` - `score.py`'s ragas judge model
(`gpt-4o-mini` by default) is independent of the app's own `LLM_PROVIDER`.

## Updating the eval set

`data/eval/kb_questions.json` is a flat list of `{"question", "ground_truth"}`
pairs grounded in `data/kb/*.txt`. Keep it small and hand-checked - it's a
regression check, not a benchmark.
