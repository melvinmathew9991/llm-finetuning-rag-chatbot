# Audit

End-to-end senior-DS audit of the project (app/, notebooks/, scripts/rag_eval/,
tests/, docs/), performed 2026-09-16. Findings are verified against actual
runtime state where possible (live `chroma_db/`, checked-in eval artifacts,
`git ls-files`), not just read from code.

## How to use this file

- Each finding gets a status: **Open**, **In progress** (branch/PR pushed,
  not yet merged), or **Fixed** (PR merged - note the PR/commit).
- Update the entry the same day a fix for it merges - don't batch it up.
- Add a dated line to the Changelog at the bottom for every change this file
  tracks, mirroring `Phases.md`'s convention of noting the closing PR/commit
  rather than deleting the historical record.

## Findings

### Bugs / correctness

- **[HIGH] `build_vector_db` silently duplicated embeddings on every
  rebuild** — `app/rag.py`. `Chroma.from_documents()` appended to whatever
  collection already existed at `persist_dir` instead of replacing it.
  Confirmed live: `chroma_db/chroma.sqlite3` had 4 embeddings for 2 unique
  KB docs; visible in the duplicated `contexts` in `data/eval/scores.json`.
  **Status: Fixed** (PR #11, `fix/chroma-duplicate-embeddings`) - drops the
  collection via Chroma's own API before rebuilding.
- **[MEDIUM] `REAL_TRAINING=1` never saves a checkpoint** — neither
  `Trainer.train()` call in the notebook is followed by
  `trainer.save_model(...)`, so real training produces weights that are
  immediately discarded. Self-documented in `README.md`. The Phase 2
  "enable real training" item is checked off but doesn't deliver a usable
  artifact. **Status: Open**
- **[LOW] 13 broken `../assets/images/...` references** in the notebook's
  RNN/LSTM/attention/decoding sections, left over from the `assets/images/`
  folder removed in `59a53c5`. README's Project Structure tree still lists
  `assets/` too. **Status: Open**

### Git hygiene

- **[MEDIUM] `models/peft/adapter_model.bin` (14.2MB) committed as a plain
  git blob**, no LFS despite `.gitattributes` marking `*.bin` binary and
  despite Rules.md's own rule against committing large model artifacts.
  ~43% of the 33MB `.git` history. **Status: Open**

### Documentation drift

- **[MEDIUM] `docs/Architecture.md` request-flow diagram is stale** — still
  shows `load_qa_chain`/`chain.run(...)` (replaced by the Phase 2 LCEL
  migration) and says sources aren't rendered in the UI (done in Phase 1).
  Folder structure is missing `scripts/`, `data/eval/`,
  `requirements-eval.txt`. Fine-tuning pipeline diagram doesn't mention
  `REAL_TRAINING`, the LoRA sweep, or QLoRA. **Status: Open**
- **[MEDIUM] `docs/PRD.md` stale in two places** — "Out of scope" still
  lists multi-turn memory as unbuilt (done, PR #4); "Success criteria"
  still calls real training "blocked" (resolved, Phase 2). **Status: Open**
- **[LOW] `README.md` Tech Stack / Project Structure don't mention**
  `bitsandbytes`, the RAG eval harness (`scripts/rag_eval/`,
  `requirements-eval.txt`, `data/eval/`), or the LoRA sweep/QLoRA notebook
  sections. **Status: Open**

### Methodology rigor (fine-tuning + eval)

- **[MEDIUM] ROUGE evaluated on a fixed 10-example slice**
  (`dataset['test'][0:10]`) everywhere - full-FT, PEFT, the LoRA sweep, and
  QLoRA all reuse it. Not random, no confidence interval; too small to
  support comparative claims like "which LoRA config wins." **Status: Open**
- **[MEDIUM] No random seed set anywhere** in the notebook
  (`transformers.set_seed`/`torch.manual_seed`) - training and
  sampling-based generation aren't reproducible run-to-run, undercutting
  PRD.md's own "reproduce the documented ROUGE deltas" criterion.
  **Status: Open**
- **[LOW] No validation-based checkpoint selection or early stopping** -
  `eval_dataset` is wired into `Trainer` but nothing acts on the eval-loss
  trend. **Status: Open**
- **[LOW] Full-FT vs. PEFT vs. QLoRA aren't matched conditions** (different
  LR/epoch counts) and the notebook doesn't caveat this. **Status: Open**

### RAG design

- **[LOW] Demo KB never exercises chunking** - both KB files (974, 412
  chars) are under the default `CHUNK_SIZE=1000`, so each is exactly one
  chunk. Phase 1's `CHUNK_OVERLAP` tuning (20→150) has zero observable
  effect on this KB. **Status: Open**
- **[uncertain→confirmed] `answer_relevancy` anomaly was mostly the
  duplicate-embedding bug, not a ragas quirk.** After regenerating against
  the fixed retrieval path, 4/5 questions now score 0.96-1.0; only the
  Russia/South Korea yes/no question still scores exactly 0.0, which does
  match ragas' "noncommittal answer" heuristic on short direct answers.
  **Status: Fixed** (isolated to one explainable case, not a pervasive
  issue).
- Checked-in `data/eval/scores.json` was generated against a KB with
  duplicated embeddings (see the HIGH bug above, fixed by PR #11).
  **Status: Fixed** - regenerated 2026-09-16 against the fixed retrieval
  path with the real OpenAI judge (gpt-4o-mini): faithfulness 0.807→1.000,
  answer_relevancy 0.392→0.788, context_recall 0.680→1.000,
  context_precision 0.944→0.900 (small drop, expected - precision is more
  sensitive to ranking now that there's no duplicate padding). Confirms the
  bug was suppressing these metrics, not just adding noise.

### Test coverage

- **[LOW] `app/main.py` has zero tests** - every other `app/` module has
  one per Rules.md's own convention; the actual Streamlit entry point
  (KB upload, cache-key construction, source rendering, session state)
  doesn't. **Status: Open**

## Changelog

- 2026-09-16: Initial audit performed; this file created with all findings
  above.
- 2026-09-16: PR #11 (`fix/chroma-duplicate-embeddings`) merged, fixing the
  HIGH duplicate-embedding bug.
- 2026-09-16: Regenerated `data/eval/results.json` + `scores.json` against
  the fixed retrieval path. First attempt with the default OpenAI judge
  failed (`.env`'s `OPENAI_API_KEY` was still the unfilled `sk-...`
  placeholder); a `--judge-provider ollama` fallback with the local
  `llama3.2:3b` model mostly timed out and returned `faithfulness: nan` -
  discarded, not committed. Re-ran with a real OpenAI key once the user set
  one: clean run, all four metrics improved or held, confirming the
  duplicate-embedding bug had been suppressing them.
