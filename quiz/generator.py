from typing import List, Dict

from llm.gemini import GeminiClient
from llm.prompts import build_quiz_prompt
from llm.validator import validate_quiz_response
from llm.schemas import Quiz


class QuizGenerator:
    """Generate grounded quizzes from retrieved RAG context."""

    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    def generate(
        self,
        topic: str,
        context_chunks: List[Dict],
        question_count: int = 5,
        difficulty: str = "medium",
    ) -> Quiz:
        """
        Generate and validate a quiz using only retrieved context.

        Args:
            topic: Topic the quiz should cover.
            context_chunks: Retrieved RAG chunks.
            question_count: Requested number of questions.
            difficulty: easy, medium, or hard.

        Returns:
            A validated Quiz object.

        Raises:
            ValueError: If inputs or generated output are invalid.
        """
        if not context_chunks:
            raise ValueError("context_chunks must not be empty.")

        prompt = build_quiz_prompt(
            topic=topic,
            context_chunks=context_chunks,
            question_count=question_count,
            difficulty=difficulty,
        )

        response_data = self.gemini_client.generate_quiz(prompt)

        valid_chunk_ids = [
            chunk["chunk_id"]
            for chunk in context_chunks
            if "chunk_id" in chunk
        ]

        return validate_quiz_response(
            data=response_data,
            valid_chunk_ids=valid_chunk_ids,
        )