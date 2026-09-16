# PRD — LLM Fine-Tuning & RAG Shopping Chatbot

## What this is

A two-part project:

1. **Fine-tuning walkthrough** (`notebooks/llm_labs.ipynb`) — teaches and demonstrates
   LLM fundamentals through prompting (zero/one/few-shot), full fine-tuning, and
   LoRA/PEFT fine-tuning of `google/flan-t5-base` on dialogue summarization.
2. **RAG shopping chatbot** (`app/`) — a Streamlit app that answers questions about a
   product knowledge base, grounded via retrieval so answers come from real KB content
   instead of the model inventing product details.

## Target users

- **Primary:** the project owner / reviewers of it as a portfolio piece — people
  evaluating hands-on LLM fine-tuning and RAG implementation skill.
- **Secondary:** anyone using the RAG chatbot as a template for a small,
  self-hosted, knowledge-grounded support/shopping assistant.

Not built for: end-shoppers on a real e-commerce site, or any multi-tenant /
production deployment.

## Core features

### Fine-tuning notebook
- Prompt engineering demos (zero/one/few-shot inference)
- Full fine-tuning of FLAN-T5-base on `knkarthick/dialogsum`
- LoRA/PEFT fine-tuning of the same base model
- Qualitative + ROUGE evaluation comparing base vs. fine-tuned outputs

### RAG chatbot app
- Chat UI (Streamlit) over a product knowledge base
- Swappable LLM provider: OpenAI API (`gpt-3.5-turbo` by default) or local Ollama —
  picked per-session from the sidebar, or defaulted via `LLM_PROVIDER`
- Local embeddings (`sentence-transformers`, `all-MiniLM-L6-v2`) regardless of chat
  provider, so no API key is needed just to build the vector index
- KB source: bundled sample docs (`data/kb/`) or user-uploaded `.txt` files via the
  sidebar
- Answers grounded in retrieved chunks (Chroma vector store), returned with
  similarity-scored source chunks rendered in the UI
- Multi-turn conversational memory: a follow-up question is condensed against
  the chat history into a standalone question (LangChain's
  `create_history_aware_retriever`) before retrieval
- RAG evaluation harness (`scripts/rag_eval/`, ragas: faithfulness, answer
  relevancy, context precision/recall) against a fixed KB question set, so
  retrieval/chunking changes have a measurable before/after

## Out of scope (current version)

- Real inventory/pricing integration (no live product database or API)
- User authentication / multi-user sessions
- Structured KB formats (CSV/JSON with price, stock, SKU) — currently `.txt` only
- Production deployment (hosting, scaling, rate limiting, monitoring)

## Success criteria

- Notebook cells run top-to-bottom without error as a smoke test
  (`REAL_TRAINING` unset); with `REAL_TRAINING=1` they reproduce the
  documented ROUGE deltas — not yet fully true end-to-end, since neither
  `Trainer.train()` call currently saves its result (see `docs/Audit.md`)
- Chatbot answers are grounded in KB content and don't hallucinate product details
  not present in the retrieved chunks
- `pytest` suite passes in CI on every push/PR
- A new contributor can get the app running from a fresh clone using only the
  README's setup instructions

## Key constraints

- Python 3.8.10 (pinned dependency versions assume this)
- Fine-tuning stack (`torch`, `transformers`, `peft`, ...) is heavy and kept
  separate from the lightweight app stack — see `requirements-*.txt`
- Large model artifacts (full fine-tune weights, optimizer/scheduler state) are
  intentionally gitignored — see the note in `README.md` under `models/`
