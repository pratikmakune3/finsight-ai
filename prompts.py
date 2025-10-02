from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM = (
  "You are a meticulous equity research assistant. "
  "Answer using only the provided context. If the answer isn’t in the context, say you don’t know. "
  "Always show citations like [source: <filename> p.<page>]. Keep answers concise but accurate."
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ]
)