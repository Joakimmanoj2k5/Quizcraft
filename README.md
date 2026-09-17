# QuizCraft RAG

A dark-mode Streamlit quiz generator that produces schema-validated multiple-choice quizzes.

## Features

- Grounded RAG mode: quiz content is limited to the supplied context chunks and includes source snippets.
- Baseline mode: generates general-knowledge quizzes from a topic.
- Gemini structured JSON generation validated by strict Pydantic models.

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
.venv/bin/streamlit run quizcraft_app.py
```

Open `http://localhost:8501`.

## Deploy on Vercel

This repo now includes a Vercel-ready FastAPI entrypoint in [app.py](app.py) so the project can run as a Python function on Vercel.

1. Push the repository to GitHub.
2. Import the repo into Vercel.
3. Add `GEMINI_API_KEY` in the Vercel project environment variables.
4. Deploy with the default settings.

Vercel will detect the FastAPI app from [app.py](app.py) and serve the quiz UI at the project root. The generated API is available at `/api/generate`.

For local testing of the Vercel path, install dependencies and run a server such as `uvicorn app:app --reload`.
