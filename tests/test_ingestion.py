import os

from app.ingestion import load_docs, split_docs


def test_load_docs_reads_kb_txt_files():
    kb_dir = os.path.join(os.path.dirname(__file__), "..", "data", "kb")
    documents = load_docs(kb_dir)
    assert len(documents) == 2


def test_split_docs_respects_chunk_size():
    kb_dir = os.path.join(os.path.dirname(__file__), "..", "data", "kb")
    documents = load_docs(kb_dir)
    chunks = split_docs(documents, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= len(documents)
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)
