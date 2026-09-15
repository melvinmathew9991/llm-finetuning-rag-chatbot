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


_CONDENSE_QUESTION_PROMPT = (
    "Given the conversation history and a follow-up question, rephrase the "
    "follow-up question to be a standalone question that captures all "
    "relevant context from the history. If the follow-up question is "
    "already standalone, return it unchanged. Reply with only the "
    "standalone question, nothing else.\n\n"
    "Conversation history:\n{history}\n\n"
    "Follow-up question: {question}\n"
    "Standalone question:"
)


def _condense_question(query: str, chat_history: list, llm) -> str:
    """Rephrases a follow-up question into a standalone one using chat history.

    A lightweight stand-in for LangChain's history-aware retriever - kept
    minimal here since app/rag.py is planned to migrate to LCEL
    (create_history_aware_retriever) next phase, at which point this swaps
    out rather than needing a rewrite.
    """
    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in chat_history)
    prompt = _CONDENSE_QUESTION_PROMPT.format(history=history_text, question=query)
    return llm.invoke(prompt).content.strip()


def get_answer(query: str, db, chain, chat_history=None) -> dict:
    """Queries the model with a given question and returns the answer + sources.

    chat_history, if given, is a list of {"role", "content"} dicts for prior
    turns; the query is condensed into a standalone question before
    retrieval so follow-up questions keep context (e.g. "what about in
    blue?" after asking about a t-shirt).
    """
    if chat_history:
        query = _condense_question(query, chat_history, chain.llm_chain.llm)

    matching_docs_score = db.similarity_search_with_score(query)
    matching_docs = [doc for doc, score in matching_docs_score]
    answer = chain.run(input_documents=matching_docs, question=query)

    sources = [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in matching_docs_score
    ]
    return {"answer": answer, "sources": sources}
