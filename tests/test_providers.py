import pytest

from app.config import load_settings
from app.providers import ProviderError, get_chat_model


def test_openai_without_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings()
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        get_chat_model(settings)


def test_ollama_when_server_unreachable_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:1")
    settings = load_settings()
    with pytest.raises(ProviderError, match="Ollama server not reachable"):
        get_chat_model(settings)
