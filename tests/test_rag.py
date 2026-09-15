"""Tests for app/rag.py - vector store build + LCEL retrieval chain.

Embeddings/chat models are stubbed per Rules.md: never hit a real
embeddings download or LLM API in tests, mock at the provider boundary.
FakeListChatModel (langchain_core's own offline test double) exercises the
real create_history_aware_retriever/create_retrieval_chain composition
end-to-end without any network calls.
"""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

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


class _StubDb:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def similarity_search_with_score(self, query):
        self.queries.append(query)
        return self.results


class _StubChain:
    """Stands in for the LCEL chain get_answer() invokes - mirrors the
    {"answer", "context"} shape create_retrieval_chain returns."""

    def __init__(self, answer: str, context: list):
        self.answer = answer
        self.context = context
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        return {"answer": self.answer, "context": self.context}


def test_get_answer_returns_answer_and_sources():
    doc = Document(
        page_content="A cotton t-shirt in sizes S-XL.",
        metadata={"source": "apparel_products.txt", "score": 0.12},
    )
    chain = _StubChain("It comes in sizes S through XL.", [doc])

    result = rag.get_answer("What sizes are available?", chain)

    assert result == {
        "answer": "It comes in sizes S through XL.",
        "sources": [
            {
                "content": doc.page_content,
                "metadata": {"source": "apparel_products.txt"},
                "score": 0.12,
            }
        ],
    }
    assert chain.calls == [{"input": "What sizes are available?", "chat_history": []}]


def test_get_answer_with_no_matches_returns_empty_sources():
    chain = _StubChain("I don't have information on that.", [])

    result = rag.get_answer("Do you sell shoes?", chain)

    assert result["sources"] == []


def test_get_answer_converts_chat_history_to_langchain_messages():
    chain = _StubChain("answer", [])
    chat_history = [
        {"role": "user", "content": "Tell me about the cotton t-shirt."},
        {"role": "assistant", "content": "It's cotton, sizes S-XL."},
    ]

    rag.get_answer("What about colors?", chain, chat_history)

    messages = chain.calls[0]["chat_history"]
    assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage"]
    assert [m.content for m in messages] == [
        "Tell me about the cotton t-shirt.",
        "It's cotton, sizes S-XL.",
    ]


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


def test_build_qa_chain_condenses_followup_using_chat_history(tmp_path, monkeypatch):
    doc = Document(
        page_content="Available in blue and black.",
        metadata={"source": "apparel_products.txt"},
    )
    db = _StubDb([(doc, 0.2)])
    llm = FakeListChatModel(
        responses=["What colors does the cotton t-shirt come in?", "It comes in blue and black."]
    )
    monkeypatch.setattr(rag, "get_chat_model", lambda settings: llm)

    chain = rag.build_qa_chain(_settings(tmp_path), db)
    chat_history = [
        {"role": "user", "content": "Tell me about the cotton t-shirt."},
        {"role": "assistant", "content": "It's a 100% cotton t-shirt in sizes S-XL."},
    ]

    result = rag.get_answer("What about colors?", chain, chat_history)

    assert db.queries == ["What colors does the cotton t-shirt come in?"]
    assert result["answer"] == "It comes in blue and black."
    assert result["sources"] == [
        {"content": doc.page_content, "metadata": doc.metadata, "score": 0.2}
    ]


def test_build_qa_chain_skips_condensing_without_history(tmp_path, monkeypatch):
    doc = Document(page_content="Ships within 3 days.", metadata={"source": "paper_products.txt"})
    db = _StubDb([(doc, 0.05)])
    llm = FakeListChatModel(responses=["It ships within 3 days."])
    monkeypatch.setattr(rag, "get_chat_model", lambda settings: llm)

    chain = rag.build_qa_chain(_settings(tmp_path), db)
    result = rag.get_answer("How fast is shipping?", chain)

    assert db.queries == ["How fast is shipping?"]
    assert result["answer"] == "It ships within 3 days."
