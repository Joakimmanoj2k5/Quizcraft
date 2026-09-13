"""Pydantic schemas used for structured QuizCraft quiz generation."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator


NonEmptyText = Annotated[StrictStr, Field(min_length=1)]


class QuestionItem(BaseModel):
    """One four-option multiple-choice question produced by the LLM."""

    model_config = ConfigDict(strict=True)

    question: NonEmptyText
    options: Annotated[list[NonEmptyText], Field(min_length=4, max_length=4)]
    correct_answer: NonEmptyText
    explanation: NonEmptyText
    source_snippet: StrictStr

    @model_validator(mode="after")
    def correct_answer_is_an_option(self) -> "QuestionItem":
        """Require an exact, rather than case-insensitive, option match."""
        if self.correct_answer not in self.options:
            raise ValueError("correct_answer must exactly match one entry in options")
        if len(set(self.options)) != 4:
            raise ValueError("options must contain four distinct choices")
        return self


class QuizResponse(BaseModel):
    """The complete structured response expected from Gemini."""

    model_config = ConfigDict(strict=True)

    questions: list[QuestionItem]
