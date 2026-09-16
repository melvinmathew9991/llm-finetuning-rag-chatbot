"""Loading and chunking the knowledge-base text files."""

import os
import shutil
from itertools import groupby

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ensure_default_kb(tmp_dir: str, source_dir: str) -> None:
    """Seed tmp_dir from the bundled sample knowledge base on first run."""
    if os.path.exists(tmp_dir) or not os.path.isdir(source_dir):
        return
    shutil.copytree(source_dir, tmp_dir)


def load_docs(directory: str):
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()


def split_docs(documents, chunk_size: int = 1000, chunk_overlap: int = 20, min_chunk_size: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return _merge_short_chunks(chunks, min_chunk_size)


def _merge_short_chunks(chunks, min_chunk_size: int):
    """Folds any chunk shorter than min_chunk_size into a neighbor from the
    same source document.

    RecursiveCharacterTextSplitter can leave a near-empty chunk standing
    alone (e.g. a lone title line split off from the rest of a document) -
    on its own, a chunk that small is generic enough to win top-1 similarity
    rankings against genuinely relevant chunks from other documents. Merging
    can push a chunk past chunk_size by up to ~min_chunk_size; that's an
    accepted trade-off since the alternative is a chunk with essentially no
    retrievable content.
    """
    result = []
    for _, group_iter in groupby(chunks, key=lambda c: c.metadata.get("source")):
        group = list(group_iter)

        i = 0
        while i < len(group) - 1:
            if len(group[i].page_content) < min_chunk_size:
                merged_content = f"{group[i].page_content}\n{group[i + 1].page_content}"
                group[i + 1] = Document(page_content=merged_content, metadata=group[i + 1].metadata)
                del group[i]
            else:
                i += 1

        # A short chunk with no following chunk from the same document (it's
        # the last one) can't be merged forward - fold it into the previous
        # chunk instead, unless it's the only chunk for this document.
        if len(group) > 1 and len(group[-1].page_content) < min_chunk_size:
            merged_content = f"{group[-2].page_content}\n{group[-1].page_content}"
            group[-2] = Document(page_content=merged_content, metadata=group[-2].metadata)
            group.pop()

        result.extend(group)
    return result
