"""Chat model / embeddings selection for the OpenAI vs. local Ollama toggle."""

import os

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import Settings


class ProviderError(RuntimeError):
    """Raised when the selected model provider isn't ready to use."""


def get_chat_model(settings: Settings):
    if settings.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Add it to .env or switch the "
                "sidebar to 'Local (Ollama)'."
            )
        return ChatOpenAI(model=settings.openai_model, api_key=api_key)

    if settings.provider == "ollama":
        if not _ollama_is_reachable(settings.ollama_base_url):
            raise ProviderError(
                f"Ollama server not reachable at {settings.ollama_base_url}. "
                f"Start it with `ollama serve` and pull the model with "
                f"`ollama pull {settings.ollama_model}`."
            )
        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)

    raise ProviderError(f"Unknown provider '{settings.provider}'")


def get_embeddings(settings: Settings):
    # Kept local regardless of provider - free, no API key or Ollama pull needed.
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _ollama_is_reachable(base_url: str) -> bool:
    try:
        response = requests.get(base_url, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False
