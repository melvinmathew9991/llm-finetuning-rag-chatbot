import os

from langchain_core.documents import Document

from app.ingestion import ensure_default_kb, load_docs, split_docs


def test_ensure_default_kb_seeds_from_source(tmp_path):
    source = tmp_path / "kb"
    source.mkdir()
    (source / "a.txt").write_text("hello")
    target = tmp_path / "tmp"

    ensure_default_kb(str(target), str(source))

    assert (target / "a.txt").read_text() == "hello"


def test_ensure_default_kb_noop_if_target_exists(tmp_path):
    target = tmp_path / "tmp"
    target.mkdir()

    ensure_default_kb(str(target), str(tmp_path / "missing_source"))

    assert list(target.iterdir()) == []


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


def test_split_docs_overlap_produces_overlapping_chunks():
    # The bundled sample KB is smaller than chunk_size, so it never exercises
    # overlap - use synthetic text long enough to force multiple chunks.
    text = "abcdefghij" * 100
    documents = [Document(page_content=text)]

    chunks = split_docs(documents, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert chunks[0].page_content[-50:] == chunks[1].page_content[:50]
