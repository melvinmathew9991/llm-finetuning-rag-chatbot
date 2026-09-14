# Rules

Boundaries for anyone (human or AI) making changes in this repo.

## Dependencies

- Keep the three-way split: `requirements-app.txt` (light, RAG app),
  `requirements-notebook.txt` (heavy, fine-tuning), `requirements-dev.txt`
  (adds pytest on top of the app stack). Don't merge them back together or
  add fine-tuning deps to the app stack.
- Pin exact versions (`==`), matching the existing style. Several pins exist
  to work around real conflicts (see the comments in `requirements-*.txt`
  around `huggingface_hub` — don't remove those pins without re-testing the
  conflict they avoid).
- Don't add a new dependency for something the standard library or an
  already-installed package can already do.

## Code style / structure

- `app/` stays a plain, flat package (`config`, `providers`, `ingestion`,
  `rag`, `main`) — one clear responsibility per module. New functionality
  should extend one of these or add a new same-level module, not grow
  `main.py` into a monolith.
- All configuration goes through `app/config.py::Settings`, sourced from
  environment variables. Never hardcode API keys, URLs, model names, or file
  paths elsewhere in `app/` — add a `Settings` field instead.
- Provider-specific logic (OpenAI vs. Ollama, or any future provider) stays
  isolated in `app/providers.py` behind `get_chat_model()` /
  `get_embeddings()`. Nothing else in `app/` should import `langchain_openai`
  or `langchain_ollama` directly.
- Secrets (`OPENAI_API_KEY`, etc.) load only via `.env` (gitignored) or real
  environment variables — never commit a key, even a test/placeholder-looking
  one, and never print one in logs or UI.

## Error handling

- Fail with a clear, user-facing message, not a silent fallback. Follow the
  existing pattern: raise `app.providers.ProviderError` for anything the user
  needs to fix (missing API key, unreachable Ollama server), and let
  `main.py` catch it and show `st.error(...)`.
- Don't swallow exceptions to "keep the demo running" — an unhandled error
  during retrieval or generation should surface, not silently return an
  empty/wrong answer.
- Validate configuration eagerly (`load_settings()` already rejects an
  unknown `LLM_PROVIDER` at startup) rather than failing deep inside a
  request.

## Libraries: prefer / avoid

- **Prefer:** LangChain's current LCEL-style composition
  (`create_retrieval_chain`, `RunnableSequence`, etc.) for any *new*
  retrieval/generation logic. The existing `load_qa_chain(chain_type="stuff")`
  in `app/rag.py` is legacy/deprecated LangChain API — kept for now, planned
  for replacement (see `Phases.md`), don't build more on top of it.
- **Avoid** introducing a second vector store, a second embeddings library,
  or a second LLM orchestration framework alongside LangChain/Chroma without
  a clear reason recorded in `Architecture.md`.
- **Avoid** notebook cells that silently depend on execution order beyond
  what's already documented — if a cell needs a prior cell's output, say so
  in a markdown cell above it (existing notebook style).

## Testing

- Every new `app/` module or non-trivial function gets a corresponding test
  in `tests/`, mirroring the existing `test_config.py` / `test_ingestion.py`
  / `test_providers.py` pattern. `app/rag.py` currently has no tests — new
  work there should add them, not extend the gap.
- `pytest` must pass locally and in CI (`.github/workflows/tests.yml`) before
  a change is considered done.
- Don't hit real external APIs (OpenAI, a live Ollama server) in tests — mock
  at the `app/providers.py` boundary.

## Git / commits

- **No AI attribution lines** (`Co-Authored-By: Claude ...`,
  `Claude-Session: ...`, or equivalent for any AI tool) in commit messages or
  PR descriptions for this repo, ever. Commits are authored solely as
  `Melvin Mathew <melvinmathew9991@gmail.com>`.
- Prefer small, focused commits with a clear message over large mixed ones.
- Don't force-push over shared history without explicit confirmation from
  the repo owner.

## What not to do

- Don't commit large model artifacts (`pytorch_model.bin`,
  optimizer/scheduler state, raw training logs) — see the gitignore note in
  `README.md` under `models/`. These are reproducible from the notebook and
  don't belong in version control.
- Don't remove the `max_steps=1` smoke-test safeguard from the notebook's
  demo run *and* leave real training on by default — if enabling real
  training, make it an explicit, documented opt-in (see `Phases.md`), not a
  silent change that makes every notebook run take hours.
- Don't change the RAG chatbot's default provider away from OpenAI, or the
  default embedding model, without updating `README.md` and `.env.example`
  to match.
