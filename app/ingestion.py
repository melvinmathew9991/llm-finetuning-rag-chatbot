"""Loading and chunking the knowledge-base text files."""

import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ensure_default_kb(tmp_dir: str, source_dir: str) -> None:
    """Seed tmp_dir from the bundled sample knowledge base on first run."""
    if os.path.exists(tmp_dir) or not os.path.isdir(source_dir):
        return
    shutil.copytree(source_dir, tmp_dir)


def load_docs(directory: str):
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()


def split_docs(documents, chunk_size: int = 1000, chunk_overlap: int = 20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
