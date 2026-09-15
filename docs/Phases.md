# Phases

Breaking remaining work into stages. Phase 0 is already done (for context);
Phases 1-3 are the active roadmap, ordered by effort and dependency, not
strictly by priority — see notes per phase.

## Branching convention

Starting with Phase 1, each phase is developed on its own branch off `master`
and merged back via PR — proper per-phase SDLC instead of committing straight
to `master`:

- `phase-0` — tags everything done before this convention started (see below).
  Branched from and identical to `master` at the time it was cut.
- `phase-1`, `phase-2`, `phase-3`, ... — one branch per phase above. Work for
  that phase happens there; open a PR into `master` when the phase's items are
  done (or a meaningful subset), review, then merge.
- Branch off `master` (not the previous phase branch) unless a phase explicitly
  depends on unmerged work from the one before it.

## Phase 0 — Done (branch: `phase-0`)

- Initial import: fine-tuning notebook + RAG chatbot app
- Restructured `app/` into a proper package with an OpenAI/Ollama provider
  toggle
- Audited project structure: fixed dependency conflicts, trimmed large model
  artifacts out of git, split requirements into `app`/`notebook`/`dev`
- Removed `docs/` (presentation + methodology PDFs) and stale README
  references
- Fixed the KB demo bug, added `pyproject.toml` packaging + GitHub Actions CI
- Pushed to GitHub under the owner's own account; scrubbed AI attribution
  from all commit history

## Phase 1 — Quick wins

Low effort, no architectural change. Good next PRs.

- [ ] Show retrieved sources/citations in the chat UI (`get_answer` already
      returns them — `app/rag.py:29-32` — `main.py` just doesn't render them)
- [ ] Add conversation memory so follow-up questions keep context (currently
      only `messages[-1]` is answered — `app/main.py:62`)
- [ ] Increase chunk overlap (`CHUNK_OVERLAP`, currently 20 on a 1000-char
      chunk — thin) and validate against the actual KB content
- [ ] Add `tests/test_rag.py` covering `build_vector_db` / `get_answer`
      against a stub/in-memory vector store

## Phase 2 — Make fine-tuning real, add RAG evaluation

Medium effort. Depends on nothing in Phase 1; can run in parallel.

- [ ] Un-stub the notebook training: remove/parameterize `max_steps=1` for a
      real run (documented opt-in per `Rules.md`), on a larger sample of
      `dialogsum` than the current ~124-row subsample
- [ ] Run a small LoRA sweep (`r`, `target_modules`, alpha/r ratio) and log
      ROUGE vs. trainable-param% in the notebook as a comparison table
- [ ] Try QLoRA (4-bit base + LoRA) as an additional efficiency comparison
- [ ] Add a RAG evaluation harness (e.g. RAGAS: faithfulness, context
      precision/recall) against a small fixed eval set of KB questions, so
      future retrieval/chunking changes have a measurable before/after
- [ ] Migrate `app/rag.py` off the deprecated `load_qa_chain` to an LCEL
      retrieval chain (`create_retrieval_chain` or a custom
      `RunnableSequence`), per `Rules.md`

## Phase 3 — Bigger bets

Higher effort, higher payoff — the differentiators, not required for the
project to "work."

- [ ] Fine-tune a model specifically for the RAG task itself (grounded-answer
      generation or query rewriting/expansion), instead of the unrelated
      dialogue-summarization dataset — connects the two halves of the
      project instead of leaving them separate
- [ ] Add structured KB ingestion (CSV/JSON with price/stock/SKU fields) with
      metadata filtering combined with vector search
- [ ] Add hybrid search (BM25 + embeddings) and/or a reranker on top of
      retrieval
- [ ] Move off "rebuild the whole vector DB on every run" toward incremental
      upsert/delete as the KB changes

## How to use this file

- Check a box and note the commit/PR that closed it, rather than deleting
  the line — keeps a record of what's been tried.
- If a phase's scope changes materially, update `PRD.md` / `Architecture.md`
  first, then reflect the change here — this file tracks *sequencing*, not
  requirements or design.
