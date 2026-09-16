import os

from langchain_core.documents import Document

from app.ingestion import _merge_short_chunks, ensure_default_kb, load_docs, split_docs


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
    # min_chunk_size=0 disables short-chunk merging - this test is about the
    # underlying splitter's size contract, not the merge feature (see
    # test_split_docs_merges_short_chunks below).
    chunks = split_docs(documents, chunk_size=200, chunk_overlap=20, min_chunk_size=0)
    assert len(chunks) >= len(documents)
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_merge_short_chunks_folds_leading_title_chunk_forward():
    # Regression test: a short standalone chunk (e.g. a title line split off
    # from the rest of a document, as happened with the real apparel_products
    # KB entry) is generic enough to win top-1 similarity search against a
    # genuinely relevant chunk from another document. A short chunk with no
    # earlier same-source chunk (it's first) gets folded into the next one.
    title = Document(page_content="Product Title", metadata={"source": "a.txt"})
    body = Document(page_content="x" * 900, metadata={"source": "a.txt"})

    merged = _merge_short_chunks([title, body], min_chunk_size=50)

    assert len(merged) == 1
    assert merged[0].page_content == f"Product Title\n{'x' * 900}"


def test_merge_short_chunks_folds_trailing_chunk_backward():
    body = Document(page_content="x" * 900, metadata={"source": "a.txt"})
    tail = Document(page_content="fin.", metadata={"source": "a.txt"})

    merged = _merge_short_chunks([body, tail], min_chunk_size=50)

    assert len(merged) == 1
    assert merged[0].page_content == f"{'x' * 900}\nfin."


def test_merge_short_chunks_does_not_cross_document_boundaries():
    short_doc = Document(page_content="Short.", metadata={"source": "a.txt"})
    other_doc = Document(page_content="Unrelated content from a different file.", metadata={"source": "b.txt"})

    merged = _merge_short_chunks([short_doc, other_doc], min_chunk_size=50)

    assert len(merged) == 2
    assert merged[0].page_content == "Short."
    assert merged[1].page_content == "Unrelated content from a different file."


def test_split_docs_overlap_produces_overlapping_chunks():
    # The bundled sample KB is smaller than chunk_size, so it never exercises
    # overlap - use synthetic text long enough to force multiple chunks.
    text = "abcdefghij" * 100
    documents = [Document(page_content=text)]

    chunks = split_docs(documents, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert chunks[0].page_content[-50:] == chunks[1].page_content[:50]
