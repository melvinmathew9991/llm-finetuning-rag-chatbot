"""Vector store + retrieval-QA chain built on top of the selected provider.

Uses LCEL (create_history_aware_retriever + create_retrieval_chain) rather
than the deprecated load_qa_chain, per Rules.md. This replaces the
hand-rolled condense-question step from Phase 1 with LangChain's own
history-aware retriever - same behavior, less custom code to maintain.
"""

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.ingestion import load_docs, split_docs
from app.providers import get_chat_model, get_embeddings


def build_vector_db(settings: Settings):
    documents = load_docs(settings.tmp_dir)
    docs = split_docs(documents, settings.chunk_size, settings.chunk_overlap)
    embeddings = get_embeddings(settings)
    return Chroma.from_documents(docs, embeddings, persist_directory=settings.persist_dir)


_CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given a chat history and the latest user question which might "
            "reference context in the chat history, formulate a standalone "
            "question that can be understood without the chat history. Do "
            "NOT answer the question, just reformulate it if needed and "
            "otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the following retrieved "
            "context. If the context doesn't contain the answer, say you "
            "don't know.\n\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _scored_retriever(db):
    """A Runnable retriever over a Chroma store that keeps similarity scores.

    `db.as_retriever()` drops the similarity score; stashing it in each
    Document's metadata lets get_answer() still report it in `sources`,
    matching the old load_qa_chain-based behavior.
    """

    def _retrieve(query: str):
        return [
            Document(page_content=doc.page_content, metadata={**doc.metadata, "score": score})
            for doc, score in db.similarity_search_with_score(query)
        ]

    return RunnableLambda(_retrieve)


def build_qa_chain(settings: Settings, db):
    """Builds the history-aware retrieval + answer chain for a vector store."""
    llm = get_chat_model(settings)
    history_aware_retriever = create_history_aware_retriever(
        llm, _scored_retriever(db), _CONTEXTUALIZE_QUESTION_PROMPT
    )
    combine_docs_chain = create_stuff_documents_chain(llm, _ANSWER_PROMPT)
    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)


def _to_lc_messages(chat_history):
    messages = []
    for turn in chat_history or []:
        message_cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(message_cls(content=turn["content"]))
    return messages


def get_answer(query: str, chain, chat_history=None) -> dict:
    """Queries the retrieval chain and returns the answer + scored sources."""
    result = chain.invoke({"input": query, "chat_history": _to_lc_messages(chat_history)})

    sources = [
        {
            "content": doc.page_content,
            "metadata": {key: value for key, value in doc.metadata.items() if key != "score"},
            "score": doc.metadata.get("score"),
        }
        for doc in result["context"]
    ]
    return {"answer": result["answer"], "sources": sources}
