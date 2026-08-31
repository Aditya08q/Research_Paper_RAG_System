# Research Paper RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) application for asking questions about uploaded research papers — built to make every stage of the RAG pipeline visible and understandable, not just functional.

**Stack:** FastAPI · LangChain · ChromaDB · BAAI/bge-small-en-v1.5 · Groq · PyMuPDF

## Why this project exists

RAG is often taught as a black box: "upload a doc, ask a question, get an answer." This project intentionally keeps every stage as its own small, readable module so you can see exactly what happens between upload and answer — chunking, embedding, vector search, prompt construction, and citation — each in its own file.

## Installation

**1. Clone / navigate into the project**
```bash
cd research-paper-rag
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure your Groq API key**
```bash
cp .env.example .env
```
Then edit `.env` and paste in your key from [console.groq.com/keys](https://console.groq.com/keys):
```
GROQ_API_KEY=your-groq-api-key-here
```

## Running the server

```bash
uvicorn app.main:app --reload
```

The API will be live at `http://localhost:8000`. Interactive docs (Swagger UI) are auto-generated at `http://localhost:8000/docs`.

## Uploading a paper

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/paper.pdf"
```

Response:
```json
{
  "filename": "paper.pdf",
  "pages_extracted": 12,
  "chunks_created": 47,
  "message": "File uploaded and indexed successfully."
}
```

## Asking a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What dataset did the paper use for training?"}'
```

Response:
```json
{
  "answer": "The paper used the TinyStories dataset...",
  "sources": [
    {
      "filename": "paper.pdf",
      "page_number": 3,
      "chunk_text": "...",
      "similarity_score": 0.87
    }
  ]
}
```

## Architecture

**Ingestion flow (upload):**
```
PDF → PyMuPDF (extract per-page text) → RecursiveCharacterTextSplitter
    (500 chars, 100 overlap) → bge-small-en-v1.5 (embed) → ChromaDB (store)
```

**Query flow (chat):**
```
Question → bge-small-en-v1.5 (embed) → ChromaDB (cosine similarity, top_k=5)
    → PromptTemplate (context + question) → Groq (generate) → Answer + citations
```

### Why these choices

- **Chunking (500/100 overlap):** keeps each chunk focused on roughly one idea while preventing sentences from being cut in half at chunk boundaries.
- **bge-small-en-v1.5:** a small, CPU-friendly embedding model with strong retrieval performance — no GPU required for this project.
- **Cosine similarity:** embeddings encode meaning as direction, not magnitude, so cosine similarity is more robust than raw distance for comparing chunks of different lengths.
- **Strict prompt grounding:** the prompt explicitly instructs the model to answer only from retrieved context and to say so when the answer isn't found — this is what prevents hallucination and is the core reason RAG is used instead of just calling the LLM directly.
- **No LangGraph, no Docker, no auth:** kept out intentionally to keep the codebase focused on the RAG concepts themselves rather than production infrastructure.

### Project structure


app/
├── api/            FastAPI route handlers (upload.py, chat.py)
├── core/           Config and logging setup
├── services/       The actual RAG logic (one concern per file)
│   ├── pdf_loader.py      → text extraction
│   ├── chunking.py        → text splitting
│   ├── embedding.py       → embedding model wrapper
│   ├── vector_store.py    → ChromaDB add/search
│   ├── llm_service.py     → Groq API wrapper
│   └── rag_pipeline.py    → orchestrates the query flow
├── models/         Pydantic request/response schemas
└── prompts/        The grounding prompt template
data/
├── pdfs/           Uploaded PDFs are saved here
└── chroma_db/      Persistent vector database files


## Error handling

The app handles and returns clear errors for: empty PDFs, corrupted/encrypted PDFs, embedding failures, vector database failures, and LLM/Groq API failures — each raised as a specific exception type and mapped to an appropriate HTTP status code.

## Notes

- OCR is not supported — scanned image PDFs with no embedded text layer will be rejected with a clear error.
- No authentication layer — this is a local, educational project, not production-ready as-is.
