"""Tests for app/rag.py - vector store build + retrieval-QA plumbing.

Embeddings/chat models are stubbed per Rules.md: never hit a real
embeddings download or LLM API in tests, mock at the provider boundary.
"""

from types import SimpleNamespace

from langchain_core.documents import Document

import app.rag as rag
from app.config import Settings


class _FakeEmbeddings:
    """Deterministic, offline stand-in for a real embeddings model.

    Same text always maps to the same vector, so a query identical to a
    stored chunk is guaranteed to be its own nearest neighbor - enough to
    prove the vector store round-trips without needing real semantics.
    """

    def _embed(self, text: str):
        return [float((hash(text) >> (8 * i)) & 0xFF) for i in range(8)]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class _StubLLM:
    """Stands in for chain.llm_chain.llm - the model used to condense a
    follow-up question into a standalone one from chat history."""

    def __init__(self, condensed: str):
        self.condensed = condensed
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.condensed)


class _StubChain:
    def __init__(self, answer: str, llm=None):
        self.answer = answer
        self.calls = []
        self.llm_chain = SimpleNamespace(llm=llm)

    def run(self, input_documents, question):
        self.calls.append({"input_documents": input_documents, "question": question})
        return self.answer


class _StubDb:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def similarity_search_with_score(self, query):
        self.queries.append(query)
        return self.results


def test_get_answer_returns_answer_and_sources():
    doc = Document(
        page_content="A cotton t-shirt in sizes S-XL.",
        metadata={"source": "apparel_products.txt"},
    )
    db = _StubDb([(doc, 0.12)])
    chain = _StubChain("It comes in sizes S through XL.")

    result = rag.get_answer("What sizes are available?", db, chain)

    assert result == {
        "answer": "It comes in sizes S through XL.",
        "sources": [{"content": doc.page_content, "metadata": doc.metadata, "score": 0.12}],
    }
    assert chain.calls == [{"input_documents": [doc], "question": "What sizes are available?"}]
    assert db.queries == ["What sizes are available?"]


def test_get_answer_with_no_matches_returns_empty_sources():
    db = _StubDb([])
    chain = _StubChain("I don't have information on that.")

    result = rag.get_answer("Do you sell shoes?", db, chain)

    assert result["sources"] == []
    assert chain.calls[0]["input_documents"] == []


def test_get_answer_condenses_followup_question_using_chat_history():
    doc = Document(
        page_content="Available in blue and black.",
        metadata={"source": "apparel_products.txt"},
    )
    db = _StubDb([(doc, 0.2)])
    llm = _StubLLM("What colors does the cotton t-shirt come in?")
    chain = _StubChain("It comes in blue and black.", llm=llm)
    chat_history = [
        {"role": "user", "content": "Tell me about the cotton t-shirt."},
        {"role": "assistant", "content": "It's a 100% cotton t-shirt in sizes S-XL."},
    ]

    result = rag.get_answer("What about colors?", db, chain, chat_history)

    assert llm.prompts, "expected the condense-question prompt to be invoked"
    assert db.queries == ["What colors does the cotton t-shirt come in?"]
    assert chain.calls[0]["question"] == "What colors does the cotton t-shirt come in?"
    assert result["answer"] == "It comes in blue and black."


def test_get_answer_skips_condensing_without_chat_history():
    db = _StubDb([])
    chain = _StubChain("Sure.")

    rag.get_answer("A first question with no history.", db, chain, chat_history=[])

    assert db.queries == ["A first question with no history."]


def _settings(tmp_path, **overrides):
    defaults = dict(
        provider="openai",
        openai_model="gpt-3.5-turbo",
        ollama_model="llama3",
        ollama_base_url="http://localhost:11434",
        embedding_model="unused-with-fake-embeddings",
        chunk_size=1000,
        chunk_overlap=20,
        tmp_dir=str(tmp_path / "kb"),
        persist_dir=str(tmp_path / "chroma_db"),
        kb_source_dir=str(tmp_path / "kb"),
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_build_vector_db_returns_queryable_store(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "shirts.txt").write_text("A cotton t-shirt available in sizes S, M, L, XL.")

    monkeypatch.setattr(rag, "get_embeddings", lambda settings: _FakeEmbeddings())

    db = rag.build_vector_db(_settings(tmp_path))

    results = db.similarity_search("A cotton t-shirt available in sizes S, M, L, XL.")
    assert len(results) == 1
    assert "cotton t-shirt" in results[0].page_content
