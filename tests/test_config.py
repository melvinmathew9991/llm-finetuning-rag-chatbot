import pytest

from app.config import load_settings


def test_defaults_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = load_settings()
    assert settings.provider == "openai"
    assert settings.openai_model == "gpt-3.5-turbo"


def test_reads_ollama_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    settings = load_settings()
    assert settings.provider == "ollama"
    assert settings.ollama_model == "mistral"


def test_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError):
        load_settings()
