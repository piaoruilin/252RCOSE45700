# 🎷 All That Jazz (LangChain + FastAPI + Chroma)

(This project is for Korea University's COSE457 course with guidance from Nxtcloud)

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using:

- **LangChain**
- **OpenAI embeddings**
- **Chroma vector store**
- **FastAPI backend**
- **Custom PDF + TXT jazz documents**
- **Source citation for every answer (mandatory assignment requirement)**

It is deployed on an **AWS Cloud9 environment running on EC2**, fully meeting the requirement for a public endpoint.

---

## 📌 Features

### 🔍 Retrieval-Augmented Generation (RAG)
- Embeds and indexes **multiple data sources** (PDF + TXT)
- Retrieves the most relevant passages using ChromaDB
- Generates grounded answers with explicit source references

### 💬 FastAPI Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/docs` | Swagger UI |
| GET | `/web` | Beautiful web chat UI |
| POST | `/chat` | Main RAG endpoint |

### 🖼️ Attractive Web Chat Interface
A custom HTML/CSS/JS interface is included under `/web`.  
It features:

- Chat bubbles  
- Dark theme  
- Smooth UI  
- Source citation below each assistant message  
- Keyboard shortcuts (Enter to send, Shift+Enter for newline)

---
## 🚀 How to Run Locally / Cloud9

### 1. Clone the repository
```bash
git clone https://github.com/piaoruilin/252RCOSE45700.git
cd 252RCOSE45700
```

### 2. Create environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
``` bash
pip install -r requirements.txt
```

### 4. Set OpenAI API key
``` bash
export OPENAI_API_KEY="your_api_key_here"
```

### 5. Run the FastAPI server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

