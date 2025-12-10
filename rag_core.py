# rag_core.py
import os
from typing import List, Dict, Any

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# LLM & embeddings (OPENAI_API_KEY comes from env)
llm = ChatOpenAI(model="gpt-4.1-mini")  # choose any allowed OpenAI chat model
embeddings = OpenAIEmbeddings()

DATA_PDF_DIR = "data/pdfs"
DATA_TXT_DIR = "data/txt"
CHROMA_DIR = "chroma_db"


def load_documents():
    """
    Load documents from at least two different data sources:
    - PDFs in data/pdfs
    - TXT files in data/txt
    """
    pdf_loader = DirectoryLoader(
        DATA_PDF_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    txt_loader = DirectoryLoader(
        DATA_TXT_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
    )

    pdf_docs = pdf_loader.load()
    txt_docs = txt_loader.load()
    docs = pdf_docs + txt_docs
    return docs


def build_or_load_vectorstore():
    """
    Create a Chroma vector store from docs if it doesn't exist,
    otherwise load the existing one from CHROMA_DIR.
    """
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        # Build new vector store
        docs = load_documents()
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
    else:
        # Load existing vector store
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    return vectorstore


# Initialize vector store & retriever once at import time
vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# RAG chain (Retrieval + Generation), returns source documents
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)


def answer_question(question: str) -> Dict[str, Any]:
    """
    Main RAG function.
    Returns both the answer and the list of source metadata.
    """
    result = qa_chain({"query": question})
    answer = result["result"]
    docs = result["source_documents"]

    sources: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        src = meta.get("source") or meta.get("file_path") or "unknown"
        page = meta.get("page")
        sources.append(
            {
                "index": i,
                "source": src,
                "page": page,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    # Optional: simple CLI for local testing
    print("RAG CLI. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = answer_question(q)
        print("\nAssistant:\n", result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            line = f"- [{s['index']}] {s['source']}"
            if s["page"] is not None:
                line += f" (page {s['page']})"
            print(line)
        print("-" * 50)
