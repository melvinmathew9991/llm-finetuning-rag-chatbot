# Architecture

## Folder structure

```
.
├── README.md
├── docs/
│   ├── PRD.md
│   ├── Architecture.md
│   ├── Rules.md
│   ├── Phases.md
│   └── Audit.md
├── pyproject.toml               # Project metadata + pytest config
├── requirements.txt             # Full install: notebook + app
├── requirements-notebook.txt    # Fine-tuning notebook stack only (heavy: torch, transformers...)
├── requirements-app.txt         # RAG chatbot app stack only (lightweight)
├── requirements-dev.txt         # Adds pytest on top of requirements-app.txt
├── requirements-eval.txt        # RAG eval harness only (ragas, own isolated venv)
├── .github/workflows/tests.yml  # CI: runs pytest on push/PR
├── .env.example
├── app/                         # Streamlit RAG shopping chatbot, as a package
│   ├── main.py                  #   Streamlit UI / entry point
│   ├── config.py                #   Settings read from environment variables
│   ├── providers.py             #   OpenAI <-> local Ollama model selection
│   ├── ingestion.py             #   Load + chunk the knowledge-base docs
│   └── rag.py                   #   Vector store + retrieval-QA chain
├── tests/                       # pytest unit tests for app/ and scripts/rag_eval/
├── scripts/
│   └── rag_eval/                # RAG evaluation harness (ragas) - see its own README.md
├── notebooks/
│   └── llm_labs.ipynb           # Fundamentals -> prompting -> full fine-tuning -> LoRA/PEFT ->
│                                 # LoRA sweep -> QLoRA -> RAG
├── data/
│   ├── kb/                      # Sample knowledge-base docs for the chatbot demo
│   └── eval/                    # Fixed eval question set + checked-in generate.py/score.py output
├── models/
│   ├── full/                    # Full fine-tuned FLAN-T5-base checkpoint (config only)
│   └── peft/                    # LoRA/PEFT adapter checkpoint
```

## Tech stack

See `README.md` → **Tech Stack** for the full list. Summary: Python 3.10,
PyTorch + Hugging Face (`transformers`, `peft`, `datasets`, `evaluate`) for
fine-tuning; LangChain + ChromaDB + `sentence-transformers` for RAG; Streamlit
for the UI; OpenAI API or local Ollama as swappable chat providers; pytest +
GitHub Actions for testing/CI.

## RAG app — request flow

```
Streamlit UI (app/main.py)
   │
   ├─ load_settings()              [app/config.py]   reads env vars -> Settings
   ├─ ensure_default_kb()          [app/ingestion.py] seeds tmp/ from data/kb/ on first run
   │
   ├─ build_vector_db(settings)    [app/rag.py]
   │     ├─ load_docs()            [app/ingestion.py] DirectoryLoader over tmp/*.txt
   │     ├─ split_docs()           [app/ingestion.py] RecursiveCharacterTextSplitter
   │     ├─ get_embeddings()       [app/providers.py] local sentence-transformers, always
   │     ├─ delete any existing collection at persist_dir first (Chroma's own
   │     │  API, not shutil.rmtree - see the docstring), so rebuilds replace
   │     │  rather than duplicate the persisted chunks
   │     └─ Chroma.from_documents()                   persisted to chroma_db/
   │
   ├─ build_qa_chain(settings, db) [app/rag.py]        LCEL, not the deprecated load_qa_chain
   │     ├─ get_chat_model()       [app/providers.py] ChatOpenAI or ChatOllama, by settings.provider
   │     ├─ create_history_aware_retriever()           condenses a follow-up + chat_history into
   │     │                                              a standalone question before retrieving
   │     ├─ _scored_retriever(db)                       wraps similarity_search_with_score so the
   │     │                                              score survives into the retrieved Documents
   │     ├─ create_stuff_documents_chain()              stuffs retrieved context into the answer prompt
   │     └─ create_retrieval_chain()                    composes the two into one Runnable
   │
   └─ on user message:
         get_answer(query, chain, chat_history)   [app/rag.py]
            ├─ chain.invoke({"input", "chat_history"})
            └─ returns {"answer", "sources"}      (sources rendered in an expander per message,
                                                     see _render_sources() in app/main.py)
```

Caching: `_load_qa_chain` is `@st.cache_resource`-keyed on
`{provider}:{openai_model}:{ollama_model}:{sorted KB filenames}`, so switching
provider or KB contents rebuilds the vector DB/chain instead of serving a
stale cached one.

Provider abstraction: `app/providers.py` is the single seam between the app
and any specific model backend. Adding a new provider (e.g. Anthropic,
Groq, a different local runtime) means adding a branch in `get_chat_model()`
and a value in `Settings.VALID_PROVIDERS` — nothing else in `app/` should
need to change.

## Fine-tuning notebook — pipeline stages

```
Load flan-t5-base + tokenizer
   → Zero/one/few-shot inference (no training)
   → Full fine-tuning
        preprocess dialogsum -> tokenize -> Trainer.train() -> save checkpoint
   → LoRA/PEFT fine-tuning
        LoraConfig(r=32, alpha=32, target_modules=['q','v']) -> get_peft_model()
        -> Trainer.train() -> save adapter
   → Evaluation
        qualitative (side-by-side generations) + ROUGE (base vs. full-FT vs. PEFT)
   → LoRA hyperparameter sweep
        5 configs varying r / target_modules / alpha-r ratio -> ROUGE vs.
        trainable-param% comparison table
   → QLoRA (4-bit base + LoRA)
        bitsandbytes 4-bit quantized base + the same LoRA config as the
        single run above -> base-model memory footprint vs. ROUGE
   → Knowledge grounding (RAG) walkthrough — conceptual bridge into app/
```

`Trainer.train()` calls are capped at `max_steps=1` (smoke test only) unless
the `REAL_TRAINING=1` environment variable is set before launching Jupyter,
in which case both the full fine-tune and PEFT cells run their real
`num_train_epochs` instead. The LoRA sweep and QLoRA sections are gated the
same way (QLoRA is additionally gated on `torch.cuda.is_available()`, since
bitsandbytes 4-bit has no CPU kernel). Checkpoints referenced by the
evaluation cells are pre-saved under `models/`; as of this writing, neither
`Trainer.train()` call is followed by an explicit `trainer.save_model(...)`,
so a real `REAL_TRAINING=1` run does not itself refresh those checkpoints -
see `docs/Audit.md`.

## Configuration

All runtime configuration flows through `app/config.py::Settings`, populated
from environment variables (see `.env.example`). No config is hardcoded
elsewhere in `app/` — new settings should be added to `Settings` and read via
`os.environ.get(...)` in `load_settings()`, not read ad hoc in other modules.
