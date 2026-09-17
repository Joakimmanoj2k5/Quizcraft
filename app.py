"""Vercel-ready FastAPI app for QuizCraft."""

from __future__ import annotations

from html import escape
from typing import Literal

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import generation
from generation import GenerationError, generate_baseline_quiz, generate_grounded_quiz


app = FastAPI(title="QuizCraft", version="1.0.0")


class QuizRequest(BaseModel):
    topic: str = Field(default="Photosynthesis", min_length=1)
    count: int = Field(default=5, ge=1, le=10)
    difficulty: str = Field(default="Medium", min_length=1)
    mode: Literal["Grounded RAG", "Baseline"] = "Grounded RAG"
    context_text: str = Field(default="")
    api_key: str | None = None


def _chunk_context(text: str) -> list[str]:
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def _render_option_cards(question: dict[str, object]) -> str:
    options = []
    correct_answer = str(question["correct_answer"])
    for option in question["options"]:
        option_text = str(option)
        option_class = "option option-answer" if option_text == correct_answer else "option"
        options.append(
            f"<div class=\"{option_class}\">{escape(option_text)}</div>"
        )

    return "".join(options)


def _render_question_cards(questions: list[dict[str, object]]) -> str:
    cards = []
    for index, item in enumerate(questions, start=1):
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">Question {index}</div>
              <h3>{escape(str(item["question"]))}</h3>
              <div class="options">{_render_option_cards(item)}</div>
              <p class="meta"><strong>Explanation:</strong> {escape(str(item["explanation"]))}</p>
              <p class="meta"><strong>Source:</strong> {escape(str(item["source_snippet"]))}</p>
            </article>
            """
        )
    return "".join(cards)


def _page_shell() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Quiz Generation App</title>
        <style>
          :root {
            color-scheme: dark;
            --bg: #07111f;
            --panel: rgba(12, 20, 36, 0.86);
            --panel-strong: rgba(18, 27, 46, 0.96);
            --line: rgba(148, 163, 184, 0.16);
            --text: #eef2ff;
            --muted: #9aa9bf;
            --accent: #7c3aed;
            --accent-2: #22d3ee;
            --success: #31c48d;
            --shadow: 0 24px 70px rgba(2, 6, 23, 0.42);
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            min-height: 100vh;
            font-family: "SF Pro Display", "Avenir Next", "Helvetica Neue", sans-serif;
            color: var(--text);
            background:
              radial-gradient(circle at top right, rgba(124, 58, 237, 0.28), transparent 28rem),
              radial-gradient(circle at 10% 18%, rgba(34, 211, 238, 0.12), transparent 20rem),
              linear-gradient(180deg, #07111f, #0b1425 44%, #050a13);
          }
          .wrap { max-width: 1180px; margin: 0 auto; padding: 40px 20px 56px; }
          .hero {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 24px;
            align-items: stretch;
          }
          .intro, .panel, .card, .status {
            border: 1px solid var(--line);
            background: var(--panel);
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
          }
          .intro {
            padding: 30px;
            border-radius: 28px;
          }
          .kicker {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(124, 58, 237, 0.18);
            color: #d8c7ff;
            font-size: 12px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
          }
          h1 {
            margin: 18px 0 14px;
            font-size: clamp(2.4rem, 6vw, 4.8rem);
            line-height: 0.95;
            letter-spacing: -0.06em;
          }
          .lead { color: var(--muted); font-size: 1.05rem; line-height: 1.7; max-width: 62ch; }
          .panel {
            padding: 24px;
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(18, 27, 46, 0.98), rgba(7, 14, 26, 0.94));
          }
          .grid { display: grid; gap: 14px; }
          .field {
            display: grid;
            gap: 8px;
          }
          label { color: #cbd5e1; font-size: 0.9rem; font-weight: 600; }
          input, select, textarea {
            width: 100%;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 16px;
            background: rgba(2, 6, 23, 0.5);
            color: var(--text);
            padding: 13px 14px;
            outline: none;
            font: inherit;
          }
          textarea { min-height: 164px; resize: vertical; line-height: 1.55; }
          input:focus, select:focus, textarea:focus { border-color: rgba(34, 211, 238, 0.65); box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.12); }
          .actions { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 4px; }
          button {
            border: 0;
            border-radius: 16px;
            padding: 13px 18px;
            background: linear-gradient(135deg, var(--accent), #9f5cff);
            color: white;
            font-weight: 700;
            cursor: pointer;
          }
          button.secondary { background: rgba(148, 163, 184, 0.12); color: var(--text); border: 1px solid var(--line); }
          .hint { color: var(--muted); font-size: 0.9rem; line-height: 1.6; }
          .status {
            margin-top: 18px;
            padding: 16px 18px;
            border-radius: 20px;
            color: var(--muted);
          }
          .status strong { color: var(--text); }
          .results { margin-top: 28px; display: grid; gap: 16px; }
          .card {
            padding: 22px;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(18, 27, 46, 0.98), rgba(10, 18, 32, 0.98));
          }
          .eyebrow {
            color: #9bd8ff;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.75rem;
            font-weight: 800;
          }
          .card h3 { margin: 10px 0 14px; font-size: 1.1rem; line-height: 1.45; }
          .options { display: grid; gap: 10px; }
          .option {
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(2, 6, 23, 0.4);
            border-radius: 14px;
            padding: 11px 12px;
          }
          .option-answer {
            border-color: rgba(49, 196, 141, 0.55);
            background: rgba(49, 196, 141, 0.12);
          }
          .meta {
            margin: 14px 0 0;
            color: var(--muted);
            line-height: 1.6;
          }
          .meta strong { color: #dbe5f4; }
          @media (max-width: 920px) {
            .hero { grid-template-columns: 1fr; }
            .wrap { padding: 18px 14px 46px; }
            .intro, .panel { border-radius: 22px; }
          }
        </style>
      </head>
      <body>
        <main class="wrap">
          <section class="hero">
            <div class="intro">
              <div class="kicker">Quiz app</div>
              <h1>Quiz Generation App</h1>
              <p class="lead">Create a quiz by choosing a topic, difficulty, and question count. Add source text for grounded questions, or use baseline mode for general questions.</p>
              <div class="status">
                Keep the form settings on the right and click Generate quiz.
              </div>
            </div>

            <div class="panel">
              <form id="quiz-form" class="grid">
                <div class="field">
                  <label for="topic">Topic</label>
                  <input id="topic" name="topic" value="Photosynthesis" />
                </div>
                <div class="field">
                  <label for="difficulty">Difficulty</label>
                  <select id="difficulty" name="difficulty">
                    <option>Easy</option>
                    <option selected>Medium</option>
                    <option>Hard</option>
                  </select>
                </div>
                <div class="field">
                  <label for="count">Question count</label>
                  <input id="count" name="count" type="number" min="1" max="10" value="5" />
                </div>
                <div class="field">
                  <label for="mode">Mode</label>
                  <select id="mode" name="mode">
                    <option selected>Grounded RAG</option>
                    <option>Baseline</option>
                  </select>
                </div>
                <div class="field">
                  <label for="api_key">Gemini API key (optional)</label>
                  <input id="api_key" name="api_key" type="password" placeholder="Uses Vercel env var if left blank" />
                </div>
                <div class="field">
                  <label for="context_text">Context chunks</label>
                  <textarea id="context_text" name="context_text">Photosynthesis takes place in chloroplasts. Plants use light energy to convert carbon dioxide and water into glucose and oxygen.

Chlorophyll is the pigment that absorbs light energy for photosynthesis.</textarea>
                </div>
                <div class="actions">
                  <button type="submit">Generate quiz</button>
                  <button type="button" id="clear-btn" class="secondary">Clear results</button>
                </div>
                <p class="hint">Tip: keep one blank line between context chunks. Grounded mode only uses the pasted context.</p>
              </form>
            </div>
          </section>

          <section id="status" class="status">Ready to generate a quiz.</section>
          <section id="results" class="results"></section>
        </main>

        <script>
          const form = document.getElementById("quiz-form");
          const status = document.getElementById("status");
          const results = document.getElementById("results");
          const clearBtn = document.getElementById("clear-btn");

          const escapeHtml = (value) => String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");

          clearBtn.addEventListener("click", () => {
            results.innerHTML = "";
            status.innerHTML = "Ready to generate a quiz.";
          });

          form.addEventListener("submit", async (event) => {
            event.preventDefault();
            status.innerHTML = "Generating quiz...";
            results.innerHTML = "";

            const payload = {
              topic: document.getElementById("topic").value,
              difficulty: document.getElementById("difficulty").value,
              count: Number(document.getElementById("count").value),
              mode: document.getElementById("mode").value,
              api_key: document.getElementById("api_key").value || null,
              context_text: document.getElementById("context_text").value,
            };

            try {
              const response = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              const data = await response.json();

              if (!response.ok) {
                status.innerHTML = `<strong>Error:</strong> ${data.detail || "Unable to generate quiz."}`;
                return;
              }

              status.innerHTML = `<strong>Success:</strong> generated ${data.questions.length} question(s).`;
              results.innerHTML = data.questions.map((question, index) => `
                <article class="card">
                  <div class="eyebrow">Question ${index + 1}</div>
                  <h3>${escapeHtml(question.question)}</h3>
                  <div class="options">
                    ${question.options.map((option) => `
                      <div class="option ${option === question.correct_answer ? 'option-answer' : ''}">${escapeHtml(option)}</div>
                    `).join("")}
                  </div>
                  <p class="meta"><strong>Explanation:</strong> ${escapeHtml(question.explanation)}</p>
                  <p class="meta"><strong>Source:</strong> ${escapeHtml(question.source_snippet)}</p>
                </article>
              `).join("");
            } catch (error) {
              status.innerHTML = `<strong>Error:</strong> ${error.message}`;
            }
          });
        </script>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(_page_shell())


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate")
def generate_quiz(request: QuizRequest) -> JSONResponse:
    try:
        client = generation.init_client(request.api_key)
        if request.mode == "Grounded RAG":
            questions = generate_grounded_quiz(
                client,
                context_chunks=_chunk_context(request.context_text),
                topic=request.topic,
                count=request.count,
                difficulty=request.difficulty,
            )
        else:
            questions = generate_baseline_quiz(
                client,
                topic=request.topic,
                count=request.count,
                difficulty=request.difficulty,
            )
    except GenerationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({"questions": questions})
