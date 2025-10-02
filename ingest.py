# Documents → Splits → Vector DB

import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = Path("data")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
EMBEDDINGS_PROVIDER = "hf" # huggingface

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_docs():
    pdf_paths = [
        DATA_DIR / "zerodha-technical-analysis-part-1.pdf",
        DATA_DIR / "zerodha-technical-analysis-part-2.pdf",
        DATA_DIR / "zerodha-fundamental-analysis.pdf",
    ]
    docs = []
    for p in pdf_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        loader = PyPDFLoader(str(p))
        for d in loader.load():
            # enrich metadata for better citations
            d.metadata["source"] = p.name
        docs.extend(loader.load())
    return docs

def chunk_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, 
      chunk_overlap=200, 
      separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(docs)

def main():
    print("Loading PDFs...")
    docs = ingest_docs()

    print("Chunking...")
    chunks = chunk_docs(docs)

    print(f"Building embeddings with: {EMBEDDINGS_PROVIDER}")
    embeddings = get_embeddings()

    print("Writing to Chroma (persistent)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="fianace-rag"
    )
    print("Ingestion completed!")

if __name__ == "__main__":
    main()