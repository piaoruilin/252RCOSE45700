# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import answer_question

app = FastAPI(
    title="RAG Chatbot",
    description="LangChain-based RAG chatbot with Chroma vector store and source citations.",
    version="1.0.0",
)

class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "RAG Chatbot is running. Use POST /chat with {'query': '...'}"}


@app.post("/chat")
def chat(q: Question):
    """
    POST /chat
    Body: { "query": "your question" }

    Response: { "answer": "...", "sources": [ ... ] }
    """
    result = answer_question(q.query)
    return result