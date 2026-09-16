"""Tests for app/main.py - the Streamlit entry point.

Uses Streamlit's own AppTest harness (streamlit.testing.v1) to actually run
the script and interact with widgets, rather than unit-testing pieces of it
in isolation - main.py is mostly wiring (session state, sidebar, cache key)
that only really means something exercised end-to-end.

Embeddings/chat models are stubbed per Rules.md: never hit a real
embeddings download or LLM API in tests, mock at the app.rag boundary
(same fakes as tests/test_rag.py) - AppTest runs the script in-process, so
monkeypatching app.rag.get_embeddings/get_chat_model before at.run() is
visible to build_vector_db()/build_qa_chain() regardless of when the
script body executes.
"""

from streamlit.testing.v1 import AppTest

import app.rag as rag


class _FakeEmbeddings:
    """Deterministic, offline stand-in for a real embeddings model - same
    as tests/test_rag.py's, so a stored chunk is its own nearest neighbor."""

    def _embed(self, text: str):
        return [float((hash(text) >> (8 * i)) & 0xFF) for i in range(8)]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _set_kb_env(monkeypatch, tmp_path, source_dir):
    monkeypatch.setenv("KB_SOURCE_DIR", str(source_dir))
    monkeypatch.setenv("KB_TMP_DIR", str(tmp_path / "kb_tmp"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-with-fake-chat-model")


def test_no_kb_shows_placeholder(tmp_path, monkeypatch):
    empty_source = tmp_path / "empty_kb"
    empty_source.mkdir()
    _set_kb_env(monkeypatch, tmp_path, empty_source)

    at = AppTest.from_file("app/main.py")
    at.run()

    assert not at.exception
    assert "No KB Loaded" in at.header[0].value
    assert len(at.chat_input) == 0


def test_chat_flow_renders_answer_and_sources(tmp_path, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "shirts.txt").write_text("A cotton t-shirt available in sizes S, M, L, XL.")
    _set_kb_env(monkeypatch, tmp_path, kb_dir)

    monkeypatch.setattr(rag, "get_embeddings", lambda settings: _FakeEmbeddings())
    monkeypatch.setattr(
        rag,
        "get_chat_model",
        lambda settings: FakeListChatModel(responses=["It comes in sizes S, M, L, XL."]),
    )

    at = AppTest.from_file("app/main.py")
    # AppTest's default 3s per-run timeout is tight for a real (if faked)
    # embed -> Chroma -> LCEL chain round trip on a cold CI runner - bump it
    # rather than risk a flaky RuntimeError("... timed out after 3(s)").
    at.default_timeout = 30
    at.run()

    assert not at.exception
    assert "shirts.txt" in at.sidebar.json[-1].value

    at.chat_input[0].set_value("What sizes are available?").run()

    assert not at.exception
    messages_by_role = {cm.name: cm for cm in at.chat_message}
    assert messages_by_role["user"].markdown[0].value == "What sizes are available?"
    assert messages_by_role["assistant"].markdown[0].value == "It comes in sizes S, M, L, XL."
    assert "Sources" in messages_by_role["assistant"].expander[0].label
