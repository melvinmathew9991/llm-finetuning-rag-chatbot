"""Streamlit RAG shopping chatbot - OpenAI API or local Ollama, picked in the sidebar."""

import os
import shutil
from dataclasses import replace
from glob import glob

import streamlit as st
from dotenv import load_dotenv

from app.config import load_settings
from app.ingestion import ensure_default_kb
from app.providers import ProviderError
from app.rag import build_qa_chain, build_vector_db, get_answer

load_dotenv()

st.title("Hi, I'm your Shopping ChatBot!")

settings = load_settings()
ensure_default_kb(settings.tmp_dir, settings.kb_source_dir)

UPLOAD_NEW = "Upload new one"
ALREADY_UPLOADED = "Already Uploaded"

provider_label = st.sidebar.radio(
    "Model provider",
    ["OpenAI API", "Local (Ollama)"],
    index=0 if settings.provider == "openai" else 1,
)
settings = replace(settings, provider="openai" if provider_label == "OpenAI API" else "ollama")


@st.cache_resource
def _load_db_and_chain(cache_key: str, settings):
    db = build_vector_db(settings)
    chain = build_qa_chain(settings)
    return db, chain


def start_chatbot(settings, cache_key: str):
    try:
        db, chain = _load_db_and_chain(cache_key, settings)
    except ProviderError as exc:
        st.error(str(exc))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            chat_history = st.session_state.messages[:-1]
            full_response = get_answer(
                st.session_state.messages[-1]["content"], db, chain, chat_history
            )
            answer = full_response["answer"]
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


content_type = st.sidebar.radio(
    "Which Knowledge base you want to use?", [ALREADY_UPLOADED, UPLOAD_NEW]
)

if content_type == UPLOAD_NEW:
    uploaded_files = st.sidebar.file_uploader("Choose a txt file", accept_multiple_files=True)

    if uploaded_files:
        if os.path.exists(settings.tmp_dir):
            shutil.rmtree(settings.tmp_dir)
        os.makedirs(settings.tmp_dir)

        for file in uploaded_files:
            with open(f"{settings.tmp_dir}/{file.name}", "wb") as temp:
                temp.write(file.getvalue())

curr_dir = [path.split(os.path.sep)[-1] for path in glob(settings.tmp_dir + "/*")]

if content_type == ALREADY_UPLOADED:
    st.sidebar.write("Current Knowledge Base")
    st.sidebar.write(curr_dir if curr_dir else "**No KB Uploaded**")

# Cache key covers both the KB contents and the provider, so switching either
# one rebuilds the vector DB / chain instead of reusing a stale cached_resource.
cache_key = f"{settings.provider}:{settings.openai_model}:{settings.ollama_model}:{','.join(sorted(curr_dir))}"

if curr_dir:
    start_chatbot(settings, cache_key)
else:
    st.header("No KB Loaded, use the left menu to start")
