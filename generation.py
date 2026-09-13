"""Gemini-backed, schema-validated quiz generation for QuizCraft RAG."""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from schemas import QuizResponse


MODEL_NAME = "gemini-2.5-flash"


class GenerationError(RuntimeError):
    """Raised when a quiz cannot be generated or validated safely."""


def init_client(api_key: str | None = None) -> genai.Client:
    """Create a Gemini client using an explicit key or ``GEMINI_API_KEY``."""
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "A Gemini API key is required. Pass api_key or set GEMINI_API_KEY."
        )
    return genai.Client(api_key=resolved_key)


def _validate_request(topic: str, count: int, difficulty: str) -> None:
    if not isinstance(topic, str) or not topic.strip():
        raise GenerationError("topic must be a non-empty string")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise GenerationError("count must be a positive integer")
    if not isinstance(difficulty, str) or not difficulty.strip():
        raise GenerationError("difficulty must be a non-empty string")


def _decode_response(response: Any, expected_count: int) -> QuizResponse:
    """Decode SDK parsed data first, then safely fall back to JSON response text."""
    payload = getattr(response, "parsed", None)
    try:
        if isinstance(payload, QuizResponse):
            quiz = payload
        elif payload is not None:
            quiz = QuizResponse.model_validate(payload)
        else:
            raw_text = getattr(response, "text", None)
            if not raw_text or not isinstance(raw_text, str):
                raise GenerationError("Gemini returned an empty structured response")
            # Defensive fallback for providers/proxies that wrap JSON in markdown.
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
            quiz = QuizResponse.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise GenerationError(
            "Gemini returned quiz data that does not match the required schema."
        ) from exc

    if len(quiz.questions) != expected_count:
        raise GenerationError(
            f"Gemini returned {len(quiz.questions)} questions; expected exactly {expected_count}."
        )
    return quiz


def _raise_generation_error(exc: Exception) -> None:
    message = str(exc)
    normalized = message.lower()
    if any(marker in normalized for marker in ("429", "rate limit", "resource_exhausted")):
        raise GenerationError(
            "Gemini rate limit reached. Please wait briefly and try again."
        ) from exc
    raise GenerationError(f"Gemini quiz generation failed: {message}") from exc


def _generate(client: genai.Client, prompt: str, count: int) -> QuizResponse:
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuizResponse,
                temperature=0.2,
            ),
        )
    except Exception as exc:  # SDK exceptions vary by installed google-genai version.
        _raise_generation_error(exc)

    return _decode_response(response, count)


def generate_grounded_quiz(
    client: genai.Client,
    context_chunks: list[str],
    topic: str,
    count: int = 5,
    difficulty: str = "Medium",
) -> list[dict[str, Any]]:
    """Generate questions whose answers are supported only by retrieved chunks."""
    _validate_request(topic, count, difficulty)
    usable_chunks = [
        chunk.strip()
        for chunk in context_chunks
        if isinstance(chunk, str) and chunk.strip()
    ]
    if not usable_chunks:
        raise GenerationError(
            "Cannot generate a grounded quiz because no non-empty context chunks were retrieved."
        )

    context_block = "\n\n".join(
        f"[Context chunk {index}]\n{chunk}"
        for index, chunk in enumerate(usable_chunks, start=1)
    )
    prompt = f"""You are QuizCraft, a precise educational quiz generator.

Create exactly {count} {difficulty.strip()} multiple-choice questions about: {topic.strip()}

Grounding rules (mandatory):
- Use ONLY facts explicitly stated in the context below.
- If a fact is not stated in the context, do not invent, infer, or extrapolate it.
- Each question must have exactly four distinct options and correct_answer must exactly equal one option.
- Every explanation must be supported by the context.
- source_snippet must be a short, exact verbatim excerpt from the supplied context that directly proves the answer. Do not paraphrase it.

SUPPLIED CONTEXT:
{context_block}
"""
    quiz = _generate(client, prompt, count)
    if any(not item.source_snippet for item in quiz.questions):
        raise GenerationError("Grounded quiz response contained an empty source_snippet.")
    if any(
        not any(item.source_snippet in chunk for chunk in usable_chunks)
        for item in quiz.questions
    ):
        raise GenerationError(
            "Grounded quiz response contained a source_snippet absent from the retrieved context."
        )
    return quiz.model_dump(mode="json")["questions"]


def generate_baseline_quiz(
    client: genai.Client,
    topic: str,
    count: int = 5,
    difficulty: str = "Medium",
) -> list[dict[str, Any]]:
    """Generate a general-knowledge quiz without retrieved documents."""
    _validate_request(topic, count, difficulty)
    prompt = f"""You are QuizCraft, an accurate educational quiz generator.

Create exactly {count} {difficulty.strip()} general-knowledge multiple-choice questions about: {topic.strip()}

Each question must have exactly four distinct options, and correct_answer must exactly equal one option.
Provide a concise explanation. This is baseline mode with no document source: set source_snippet to exactly "General Knowledge" for every question.
"""
    quiz = _generate(client, prompt, count)
    # Enforce the baseline contract even if a model returns an unnecessary citation.
    questions = [
        item.model_copy(update={"source_snippet": "General Knowledge"})
        for item in quiz.questions
    ]
    return QuizResponse(questions=questions).model_dump(mode="json")["questions"]


if __name__ == "__main__":
    client = init_client()

    baseline = generate_baseline_quiz(client, topic="Operating Systems")
    print("Baseline quiz:\n" + json.dumps(baseline, indent=2))

    photosynthesis_notes = [
        "Photosynthesis takes place in chloroplasts. Plants use light energy to convert carbon dioxide and water into glucose and oxygen.",
        "Chlorophyll is the pigment that absorbs light energy for photosynthesis.",
    ]
    grounded = generate_grounded_quiz(
        client,
        context_chunks=photosynthesis_notes,
        topic="Photosynthesis",
    )
    print("\nGrounded quiz:\n" + json.dumps(grounded, indent=2))
