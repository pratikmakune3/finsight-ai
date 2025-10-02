import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from operator import itemgetter

from prompts import QA_PROMPT
from utils import format_docs, sources_badge

load_dotenv()

st.set_page_config(page_title="Finance Research Assistant", page_icon="📈")

PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
EMBEDDINGS_PROVIDER = "hf" # huggingface

@st.cache_resource
def get_embeddings():
  return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_retriever():
    embeddings = get_embeddings()
    vs = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name="fianace-rag",
        embedding_function=embeddings
    )
    # MMR gives diversity; tweak in sidebar
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": st.session_state.k, "fetch_k": st.session_state.fetch_k})

def build_chain(retriever):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=1, groq_api_key=os.getenv("GROQ_API_KEY"))
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "k" not in st.session_state:
        st.session_state.k = 4
    if "fetch_k" not in st.session_state:
        st.session_state.fetch_k = 20
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

init_session()

st.sidebar.header("RAG Settings")
st.session_state.k = st.sidebar.slider("Top-K (returned)", min_value=1, max_value=10, value=4)
st.session_state.fetch_k = st.sidebar.slider("Fetch-K (pool size)", min_value=10, max_value=50, value=20)
rerun_ingest = st.sidebar.button("Rebuild Index (run ingest.py locally)")

st.title("📈 FinSight AI")
st.caption("Questions on Technical & Fundamental Analysis. Answers grounded in PDFs with citations.")

retriever = get_retriever()
chain = build_chain(retriever)

# Chat UI
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Ask about candlesticks, S&R, RSI, dow theory, FA basics, etc.")
if user_q:
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    # retrieve to show sources in UI too
    docs = retriever.get_relevant_documents(user_q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # build chat history from memory
            chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
            answer = chain.invoke({
                "question": user_q,
                "chat_history": chat_history,
            })
            st.markdown(answer)
            if docs:
                st.info("**Sources**: " + sources_badge(docs))
    st.session_state.messages.append(("assistant", answer))
    # update memory with this turn
    st.session_state.memory.save_context({"input": user_q}, {"output": answer})