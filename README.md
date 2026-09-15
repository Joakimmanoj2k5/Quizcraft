# QuizCraft RAG

A dark-mode Streamlit quiz generator that produces schema-validated multiple-choice quizzes.

## Features

-Baseline mode — generate a general-knowledge quiz from a typed topic.
-Grounded RAG mode — upload your own notes (PDF/TXT) and get quiz questions traceable back to your content.
-Schema-validated output — all quiz JSON is validated against strict Pydantic models so malformed LLM responses never break the app.
-Source display — each grounded question shows the source file/section it was drawn from.

## RAG Pipeline Architecture
QuizCraft's grounded mode follows a standard Retrieval-Augmented Generation flow:

Ingestion → Chunking → Embedding → Retrieval → Generation

1.Ingestion — the user uploads a .txt or .pdf file through the Streamlit UI. The file is parsed and readable text is extracted (PDF parsing via pdfplumber/PyPDF2).
2.Chunking — extracted text is split into overlapping chunks sized for semantic coherence, so retrieval can pull focused, relevant passages instead of whole documents.
3.Embedding — each chunk is converted into a vector embedding using  [EMBEDDING MODEL — TBD].
4.Retrieval — embeddings are stored in a local ChromaDB vector store. When a quiz is requested, the topic is embedded and the most semantically relevant chunks are retrieved.
5.Generation — the retrieved chunks are passed as grounded context to the Gemini API, which is prompted to generate quiz questions only from the provided context. Output is validated against Pydantic schemas before being shown to the user.
6.Quiz-taking & scoring — the validated quiz is rendered in the UI; the user answers, submits, and receives a scored result with source references for each question.

## Tech Stack
Layer          Choice
Frontend       Streamlit
Backend        Python
LLM API        Google Gemini (free tier)
Embeddings     [EMBEDDING MODEL — TBD, see note above]
Vector Store   ChromaDB (local, free)
File Parsing   pdfplumber / PyPDF2 (PDF)
Validation     Pydantic
Testing        pytest

## Setup Instructions

Requirements: Python 3.10+, a free Gemini API key.
# 1. Clone the repo
git clone https://github.com/Joakimmanoj2k5/Quizcraft.git
cd Quizcraft

# 2. Create and activate a virtual environment
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Set your Gemini API key
export GEMINI_API_KEY="your-key"

# 4. Run the app
.venv/bin/streamlit run quizcraft_app.py
Open http://localhost:8501 in your browser.
Note on cost: This project uses only free-tier services — the Gemini API free tier and a local ChromaDB instance. No paid APIs, subscriptions, or tools are required to run it.

Running Tests
.venv/bin/pytest tests/




## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
.venv/bin/streamlit run quizcraft_app.py
```

Open `http://localhost:8501`.
