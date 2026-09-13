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
