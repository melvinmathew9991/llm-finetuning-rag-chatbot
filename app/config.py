"""Runtime settings for the RAG chatbot, read from environment variables."""

import os
from dataclasses import dataclass

VALID_PROVIDERS = ("openai", "ollama")


@dataclass(frozen=True)
class Settings:
    provider: str
    openai_model: str
    ollama_model: str
    ollama_base_url: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    tmp_dir: str
    persist_dir: str
    kb_source_dir: str


def load_settings() -> Settings:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}', expected one of {VALID_PROVIDERS}"
        )

    return Settings(
        provider=provider,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3"),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "20")),
        tmp_dir=os.environ.get("KB_TMP_DIR", "tmp"),
        persist_dir=os.environ.get("CHROMA_PERSIST_DIR", "chroma_db"),
        kb_source_dir=os.environ.get("KB_SOURCE_DIR", "data/kb"),
    )
