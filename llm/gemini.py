import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai


load_dotenv()


QUIZ_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
        },
        "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard"],
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "correct_answer": {
                        "type": "string",
                    },
                    "explanation": {
                        "type": "string",
                    },
                    "source_chunk_id": {
                        "type": "string",
                    },
                },
                "required": [
                    "question",
                    "options",
                    "correct_answer",
                    "explanation",
                    "source_chunk_id",
                ],
            },
        },
    },
    "required": [
        "topic",
        "difficulty",
        "questions",
    ],
}


class GeminiClient:
    """Wrapper around the Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.6-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or pass api_key explicitly."
            )

        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """Generate plain text from Gemini."""

        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text

    def generate_quiz(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a quiz using Gemini's structured JSON output.
        """

        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QUIZ_RESPONSE_SCHEMA,
            },
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        try:
            result = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Gemini returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                "Gemini quiz response must be a JSON object."
            )

        return result