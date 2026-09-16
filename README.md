# LLM Project to Build and Fine Tune a Large Language Model

In today's data-driven world, the ability to process and generate natural language text at scale has become a transformative force across industries. Large Language Models (LLMs) represent a cutting-edge advancement in natural language processing, enabling businesses to extract valuable insights, automate tasks, and enhance user experiences. By harnessing the power of LLMs, organizations can improve customer service, automate content creation, and gain a competitive edge in the digital landscape.

This project builds the foundation for Large Language Models by diving deep into the details of their inner workings. It shows how to optimize their use through prompt engineering and fine-tuning techniques such as LoRA.

Prompt engineering techniques involve crafting specific instructions or queries given to the language model to influence its output, guiding LLMs toward desired responses through zero-shot, one-shot, and few-shot inference.

Fine-tuning entails training a pre-trained language model on a specific task or dataset to adapt it for a particular application. This project explores both full fine-tuning and Parameter Efficient Fine Tuning (PEFT/LoRA), a technique that optimizes the fine-tuning process by focusing on a subset of the model's parameters, making it more resource-efficient.

The project also applies Retrieval Augmented Generation (RAG) using OpenAI's GPT-3.5 Turbo, resulting in a knowledge-grounded chatbot for online shopping. Knowledge grounding with RAG mitigates hallucinations and provides trustworthy responses by incorporating information from external sources to validate and support generated text. For example, in an e-commerce chatbot, RAG ensures product information, availability, and prices are sourced from a trusted knowledge base rather than invented by the model.

## Tech Stack

- **Language:** Python 3.8
- **Fine-tuning:** PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`/`rouge_score`, `peft`/`loralib` (LoRA), `bitsandbytes` (QLoRA, 4-bit)
- **Base model:** `google/flan-t5-base`
- **RAG / chatbot:** LangChain (`langchain`, `langchain-community`, `langchain-openai`, `langchain-ollama`, `langchain-chroma`, `langchain-text-splitters`), ChromaDB (vector store), `sentence-transformers` (`all-MiniLM-L6-v2` embeddings)
- **RAG evaluation:** `ragas` (faithfulness, answer relevancy, context precision/recall) - see `scripts/rag_eval/`
- **LLM providers:** OpenAI API (GPT-3.5 Turbo) or local Ollama
- **App/UI:** Streamlit
- **Testing/CI:** pytest, GitHub Actions

## Project Structure

```
.
├── README.md
├── pyproject.toml               # Project metadata + pytest config
├── requirements.txt             # Full install: notebook + app
├── requirements-notebook.txt    # Fine-tuning notebook stack only (heavy: torch, transformers...)
├── requirements-app.txt         # RAG chatbot app stack only (lightweight)
├── requirements-dev.txt         # Adds pytest on top of requirements-app.txt
├── requirements-eval.txt        # RAG eval harness only (ragas, own isolated venv)
├── .github/workflows/tests.yml  # CI: runs pytest on push/PR
├── .gitignore
├── .gitattributes
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
│   └── llm_labs.ipynb          # Main walkthrough: fundamentals -> prompting -> full fine-tuning
│                                # -> LoRA/PEFT -> LoRA sweep -> QLoRA -> RAG
├── data/
│   ├── kb/                     # Sample knowledge-base docs for the chatbot demo
│   │   ├── apparel_products.txt
│   │   └── paper_products.txt
│   └── eval/                   # Fixed eval question set + checked-in eval harness output
├── models/
│   ├── full/                   # Full fine-tuned FLAN-T5-base checkpoint (config only - see note)
│   └── peft/                   # LoRA/PEFT adapter checkpoint
└── docs/                       # PRD, Architecture, Rules, Phases, Audit
```

> Note: `models/` only tracks what's small and documents the setup -
> configs and LoRA hyperparameters. Everything else - optimizer/scheduler/
> rng state, the full fine-tune's 990MB `pytorch_model.bin`, the LoRA
> adapter's 14MB `adapter_model.bin`, and the raw per-step training logs -
> is gitignored: reproducible by re-running the fine-tuning cells in the
> notebook, not needed for inference, and (for `training_args.bin`) a
> pickle file with no real reason to carry it in version control. **This
> means neither `models/full/` nor `models/peft/` can be loaded as-is
> from a fresh clone** — you need to actually run the fine-tuning cells in
> the notebook (see "Fine-tuning" below) to regenerate the weight files.
> (`adapter_model.bin` was tracked in git history up through this commit -
> it's untracked going forward, but still present in old commits/tags.)

## Execution Instructions

### Python version 3.8.10

To create a virtual environment and install requirements, pick the file that
matches what you want to run - installing only what you need avoids pulling
in the multi-GB fine-tuning stack (torch, transformers, datasets, peft, ...)
just to run the chatbot:

| Want to...                          | Install                       |
|--------------------------------------|--------------------------------|
| Run the RAG chatbot app only         | `requirements-app.txt`         |
| Run the fine-tuning notebook only    | `requirements-notebook.txt`    |
| Both                                  | `requirements.txt`             |
| Run the test suite (needs the app)   | `requirements-dev.txt`         |

**Windows:**
```
cd "C:\path\to\project"
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements-app.txt
```

**Linux/Mac:**
```
cd /path/to/project
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements-app.txt
```

If you have multiple Python versions installed, use the Python Launcher to target 3.8.10 specifically:

- Windows: `py -3.8 -m venv myenv`
- Linux/Mac: `python3.8 -m venv myenv`

then activate and install the requirements file for what you're running, as above.

Running tests needs the dev extras too: `pip install -r requirements-dev.txt`.

> **Windows long paths**: if your project lives at a deeply nested path
> (long folder names, especially with spaces), `pip install` can fail with
> `OSError: [Errno 2] No such file or directory` on packages that have deeply
> nested files (`transformers` is a common one), because Windows' default
> 260-character path limit gets exceeded once combined with the venv's own
> `site-packages` path. Either [enable Windows long path support](https://pip.pypa.io/warnings/enable-long-paths)
> (one-time, needs admin), or create the venv somewhere shorter (e.g.
> `C:\envs\<project>`) and point it at this project instead of creating it
> inside the project folder.

### Running the project

Run the main notebook (needs `requirements-notebook.txt` installed; from the
project root, so its relative paths to `assets/images` and `models/` resolve
correctly):

```
jupyter notebook notebooks/llm_labs.ipynb
```

Run the RAG shopping chatbot (also from the project root — it creates
`tmp/` and `chroma_db/` working directories there):

```
streamlit run app/main.py
```

### Model provider: OpenAI API or local Ollama

The chatbot's chat model can come from either OpenAI's API or a locally
running [Ollama](https://ollama.com) server — pick per-session from the
sidebar radio button, or set a default with `LLM_PROVIDER` in `.env`.
Embeddings always run locally via `sentence-transformers`
(`all-MiniLM-L6-v2`) regardless of which chat provider you pick, so no
API key or Ollama model is needed just to build the vector index.

**OpenAI API** (default) — provide your API key, never edit it directly
into the app code. It loads via `python-dotenv`, so the simplest way is a
`.env` file in the project root:

```
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

`.env` is gitignored, so the real key never gets committed. Alternatively,
set it directly as an environment variable for the session (this also
overrides any value in `.env`):

```
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."

# Linux/Mac
export OPENAI_API_KEY="sk-..."
```

**Local Ollama** — no API key needed, but [install Ollama](https://ollama.com/download),
start the server, and pull a model first:

```
ollama serve
ollama pull llama3
```

Then in `.env` (or as env vars) set:

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434   # default, only needed if Ollama runs elsewhere
```

You can still switch providers per-session from the sidebar without
restarting the app — switching rebuilds the vector store/chain for the
newly selected provider.

### Running tests

```
pip install -r requirements-dev.txt
pytest
```

### RAG evaluation harness

Measures whether retrieval/chunking changes actually help or hurt, using
[ragas](https://github.com/explodinggradients/ragas) (faithfulness, answer
relevancy, context precision/recall) against a small fixed set of KB
questions. Runs in its own isolated venv, separate from `requirements-app.txt`
- see `scripts/rag_eval/README.md` for why and for the full usage.

```
python -m venv .venv-eval
.venv-eval/Scripts/pip install -r requirements-eval.txt

python -m scripts.rag_eval.generate   # app's own venv - real answers via app.rag
.venv-eval/Scripts/python -m scripts.rag_eval.score   # ragas venv - scores them
```

### Fine-tuning

`notebooks/llm_labs.ipynb` walks through both full fine-tuning and
LoRA/PEFT of `google/flan-t5-base` on the `knkarthick/dialogsum` dataset,
subsampled to every 20th row (~623 training examples). **Important**: the
notebook's own `Trainer.train()` calls are capped at `max_steps=1` (a
smoke test that the training loop runs), then it loads the already-trained
checkpoints from `models/full/` / `models/peft/` for the evaluation cells -
running the notebook top-to-bottom as-is will **not** reproduce those
checkpoints (and `models/full/pytorch_model.bin` isn't tracked in git at
all - see the note on `models/` above).

To actually fine-tune for real:
1. Set the `REAL_TRAINING=1` environment variable before launching Jupyter
   (e.g. `REAL_TRAINING=1 jupyter notebook`) - both `TrainingArguments`
   cells use it to swap `max_steps=1` for their real `num_train_epochs`
   (1 for the full fine-tune, 20 for PEFT/LoRA). On CPU, the full fine-tune
   takes roughly an hour; the PEFT cell will take considerably longer -
   reduce `num_train_epochs` if you just want to confirm training works
   end-to-end
2. Optionally loosen the `% 20 == 0` dataset subsampling filter further to
   train on more than ~623 examples
3. After `trainer.train()`, explicitly save the result -
   `trainer.save_model('../models/full/')` and
   `tokenizer.save_pretrained('../models/full/')` (or a new path) - the
   notebook doesn't currently do this, since it expects a checkpoint to
   already exist
4. Re-run the qualitative and ROUGE evaluation cells to compare against
   the base model

Two further comparisons build on the single LoRA run above, both also
gated on `REAL_TRAINING=1`:
- **LoRA hyperparameter sweep** - 5 configs varying `r`, `target_modules`,
  and the `lora_alpha`/`r` ratio, logging ROUGE next to trainable-param% in
  a comparison table
- **QLoRA (4-bit base + LoRA)** - quantizes the base model to 4-bit via
  `bitsandbytes` and attaches the same LoRA config as the single run above,
  reporting base-model memory footprint alongside ROUGE. Needs a CUDA GPU
  (bitsandbytes 4-bit has no CPU kernel) - additionally gated on
  `torch.cuda.is_available()`, so it's skipped with a clear message on a
  CPU-only machine
