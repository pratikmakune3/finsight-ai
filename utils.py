from typing import List
from langchain.schema import Document

def format_docs(docs: List[Document]) -> str:
    as_blocks = []
    for d in docs:
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page", "?")
        as_blocks.append(f"[source: {src} p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(as_blocks)

def sources_badge(docs: List[Document]) -> str:
    uniq = []
    for d in docs:
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page", "?")
        tag = f"{src} p.{page}"
        if tag not in uniq:
            uniq.append(tag)
    return ", ".join(uniq)
