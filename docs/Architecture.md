# Architecture

## Folder structure

```
.
├── README.md
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── pyproject.toml               # Project metadata + pytest config
├── requirements.txt             # Full install: notebook + app
├── requirements-notebook.txt    # Fine-tuning notebook stack only (heavy: torch, transformers...)
├── requirements-app.txt         # RAG chatbot app stack only (lightweight)
├── requirements-dev.txt         # Adds pytest on top of requirements-app.txt
├── .github/workflows/tests.yml  # CI: runs pytest on push/PR
├── .env.example
├── app/                         # Streamlit RAG shopping chatbot, as a package
│   ├── main.py                  #   Streamlit UI / entry point
│   ├── config.py                #   Settings read from environment variables
│   ├── providers.py             #   OpenAI <-> local Ollama model selection
│   ├── ingestion.py             #   Load + chunk the knowledge-base docs
│   └── rag.py                   #   Vector store + retrieval-QA chain
├── tests/                       # pytest unit tests for app/
├── notebooks/
│   └── llm_labs.ipynb           # Fundamentals -> prompting -> full fine-tuning -> LoRA/PEFT -> RAG
├── data/
│   └── kb/                      # Sample knowledge-base docs for the chatbot demo
├── models/
│   ├── full/                    # Full fine-tuned FLAN-T5-base checkpoint (config only)
│   └── peft/                    # LoRA/PEFT adapter checkpoint
```

## Tech stack

See `README.md` → **Tech Stack** for the full list. Summary: Python 3.8,
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
   │     └─ Chroma.from_documents()                   persisted to chroma_db/
   │
   ├─ build_qa_chain(settings)     [app/rag.py]
   │     ├─ get_chat_model()       [app/providers.py] ChatOpenAI or ChatOllama, by settings.provider
   │     └─ load_qa_chain(chain_type="stuff")
   │
   └─ on user message:
         get_answer(query, db, chain)   [app/rag.py]
            ├─ db.similarity_search_with_score(query)   top-k retrieval
            ├─ chain.run(input_documents=..., question=...)
            └─ returns {"answer", "sources"}            (sources computed but not yet
                                                           rendered in the UI — see Phases.md)
```

Caching: `_load_db_and_chain` is `@st.cache_resource`-keyed on
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
   → Knowledge grounding (RAG) walkthrough — conceptual bridge into app/
```

Checkpoints referenced by the evaluation cells are pre-saved under `models/`;
the notebook's own `Trainer.train()` calls are capped at `max_steps=1` (smoke
test only) — see `Phases.md` Phase 1 for the plan to make training real.

## Configuration

All runtime configuration flows through `app/config.py::Settings`, populated
from environment variables (see `.env.example`). No config is hardcoded
elsewhere in `app/` — new settings should be added to `Settings` and read via
`os.environ.get(...)` in `load_settings()`, not read ad hoc in other modules.
