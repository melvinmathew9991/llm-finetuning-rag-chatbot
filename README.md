# LLM Project to Build and Fine Tune a Large Language Model

In today's data-driven world, the ability to process and generate natural language text at scale has become a transformative force across industries. Large Language Models (LLMs) represent a cutting-edge advancement in natural language processing, enabling businesses to extract valuable insights, automate tasks, and enhance user experiences. By harnessing the power of LLMs, organizations can improve customer service, automate content creation, and gain a competitive edge in the digital landscape.

This project builds the foundation for Large Language Models by diving deep into the details of their inner workings. It shows how to optimize their use through prompt engineering and fine-tuning techniques such as LoRA.

Prompt engineering techniques involve crafting specific instructions or queries given to the language model to influence its output, guiding LLMs toward desired responses through zero-shot, one-shot, and few-shot inference.

Fine-tuning entails training a pre-trained language model on a specific task or dataset to adapt it for a particular application. This project explores both full fine-tuning and Parameter Efficient Fine Tuning (PEFT/LoRA), a technique that optimizes the fine-tuning process by focusing on a subset of the model's parameters, making it more resource-efficient.

The project also applies Retrieval Augmented Generation (RAG) using OpenAI's GPT-3.5 Turbo, resulting in a knowledge-grounded chatbot for online shopping. Knowledge grounding with RAG mitigates hallucinations and provides trustworthy responses by incorporating information from external sources to validate and support generated text. For example, in an e-commerce chatbot, RAG ensures product information, availability, and prices are sourced from a trusted knowledge base rather than invented by the model.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   └── llm_app.py              # Streamlit RAG shopping chatbot
├── notebooks/
│   └── llm_labs.ipynb          # Main walkthrough: fundamentals -> prompting ->
│                                # full fine-tuning -> LoRA/PEFT -> RAG
├── data/
│   └── kb/                     # Sample knowledge-base docs for the chatbot demo
│       ├── apparel_products.txt
│       └── paper_products.txt
├── models/
│   ├── full/                   # Full fine-tuned FLAN-T5-base checkpoint
│   └── peft/                   # LoRA/PEFT adapter checkpoint
├── assets/
│   └── images/                 # Diagrams referenced by the notebook
└── docs/
    ├── LLM+1+Presentation.pdf
    └── llm-finetuning-solution-methodology.pdf
```

> Note: `models/` contains multi-gigabyte training checkpoints (`optimizer.pt`,
> `scheduler.pt`, `rng_state.pth`). These are excluded via `.gitignore` since
> they're reproducible by re-running the fine-tuning cells in the notebook —
> only the model weights/config themselves are needed for inference.

## Execution Instructions

### Python version 3.8.10

To create a virtual environment and install requirements:

**Windows:**
```
cd "C:\path\to\project"
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```
cd /path/to/project
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

If you have multiple Python versions installed, use the Python Launcher to target 3.8.10 specifically:

- Windows: `py -3.8 -m venv myenv`
- Linux/Mac: `python3.8 -m venv myenv`

then activate and `pip install -r requirements.txt` as above.

### Running the project

Run the main notebook (from the project root, so its relative paths to
`assets/images` and `models/` resolve correctly):

```
jupyter notebook notebooks/llm_labs.ipynb
```

Run the RAG shopping chatbot (also from the project root — it creates
`tmp/` and `chroma_db/` working directories there):

```
streamlit run app/llm_app.py
```

Before running the chatbot, provide your OpenAI API key — never edit it
directly into `app/llm_app.py`. The app loads it via `python-dotenv`, so
the simplest way is a `.env` file in the project root:

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
