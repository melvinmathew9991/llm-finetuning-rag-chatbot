"""Vector store + retrieval-QA chain built on top of the selected provider."""

from langchain.chains.question_answering import load_qa_chain
from langchain_chroma import Chroma

from app.config import Settings
from app.ingestion import load_docs, split_docs
from app.providers import get_chat_model, get_embeddings


def build_vector_db(settings: Settings):
    documents = load_docs(settings.tmp_dir)
    docs = split_docs(documents, settings.chunk_size, settings.chunk_overlap)
    embeddings = get_embeddings(settings)
    return Chroma.from_documents(docs, embeddings, persist_directory=settings.persist_dir)


def build_qa_chain(settings: Settings):
    llm = get_chat_model(settings)
    return load_qa_chain(llm, chain_type="stuff", verbose=True)


def get_answer(query: str, db, chain) -> dict:
    """Queries the model with a given question and returns the answer + sources."""
    matching_docs_score = db.similarity_search_with_score(query)
    matching_docs = [doc for doc, score in matching_docs_score]
    answer = chain.run(input_documents=matching_docs, question=query)

    sources = [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in matching_docs_score
    ]
    return {"answer": answer, "sources": sources}
