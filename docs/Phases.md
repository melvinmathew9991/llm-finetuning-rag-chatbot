# Phases

Breaking remaining work into stages. Phase 0 is already done (for context);
Phases 1-3 are the active roadmap, ordered by effort and dependency, not
strictly by priority — see notes per phase.

## Branching convention

Starting with Phase 1, work happens on branches off `main` and merges back via
PR — proper per-phase, per-task SDLC instead of committing straight to `main`:

- `phase-0` — tags everything done before this convention started (see below).
  Branched from and identical to `main` at the time it was cut.
- Each phase's items are built one per branch: `phase-<n>/<task-slug>` (e.g.
  `phase-1/add-rag-tests`, `phase-1/citations-ui`). Each task branch gets its
  own PR into `main`, reviewed and merged independently — keeps diffs small
  and bisectable instead of bundling a whole phase into one PR.
- Branch off `main` (not another in-flight task branch) unless a task
  explicitly depends on unmerged work from another one in the same phase.
- Tick the item off in the phase's checklist below with the merging PR/commit
  once its branch is merged.

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

## Phase 1 — Quick wins — Done

Low effort, no architectural change.

- [x] Add `tests/test_rag.py` covering `build_vector_db` / `get_answer`
      against a stub/in-memory vector store (PR #1, `phase-1/add-rag-tests`)
- [x] Show retrieved sources/citations in the chat UI (PR #2,
      `phase-1/citations-ui`)
- [x] Increase chunk overlap (`CHUNK_OVERLAP` 20 → 150) and validate against
      real content (PR #3, `phase-1/chunk-overlap-tuning`)
- [x] Add conversation memory so follow-up questions keep context, via a
      condense-question step before retrieval (PR #4,
      `phase-1/conversation-memory`)

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
