from typing import Any, Dict, List

from .schemas import Quiz, QuizQuestion


VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_quiz_response(
    data: Dict[str, Any],
    valid_chunk_ids: List[str],
) -> Quiz:
    """
    Validate a Gemini-generated quiz response and convert it
    into our internal Quiz dataclass.

    Raises ValueError when the response is invalid.
    """

    if not isinstance(data, dict):
        raise ValueError("Quiz response must be a JSON object.")

    topic = data.get("topic")
    difficulty = data.get("difficulty")
    questions = data.get("questions")

    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Quiz topic must be a non-empty string.")

    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            "Quiz difficulty must be one of: easy, medium, hard."
        )

    if not isinstance(questions, list):
        raise ValueError("Quiz questions must be a list.")

    if not questions:
        raise ValueError("Quiz must contain at least one question.")

    if not isinstance(valid_chunk_ids, list):
        raise ValueError("valid_chunk_ids must be a list.")

    validated_questions = []

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question {index} must be a JSON object."
            )

        question = item.get("question")
        options = item.get("options")
        correct_answer = item.get("correct_answer")
        explanation = item.get("explanation")
        source_chunk_id = item.get("source_chunk_id")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Question {index} must have non-empty question text."
            )

        if not isinstance(options, list):
            raise ValueError(
                f"Question {index} options must be a list."
            )

        if len(options) != 4:
            raise ValueError(
                f"Question {index} must contain exactly 4 options."
            )

        if any(
            not isinstance(option, str) or not option.strip()
            for option in options
        ):
            raise ValueError(
                f"Question {index} contains an empty or invalid option."
            )

        if len(set(options)) != 4:
            raise ValueError(
                f"Question {index} options must be distinct."
            )

        if not isinstance(correct_answer, str) or not correct_answer.strip():
            raise ValueError(
                f"Question {index} must have a correct answer."
            )

        if correct_answer not in options:
            raise ValueError(
                f"Question {index} correct_answer must exactly "
                "match one of the options."
            )

        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(
                f"Question {index} must have a non-empty explanation."
            )

        if not isinstance(source_chunk_id, str) or not source_chunk_id.strip():
            raise ValueError(
                f"Question {index} must have a source_chunk_id."
            )

        if source_chunk_id not in valid_chunk_ids:
            raise ValueError(
                f"Question {index} references an unknown source chunk: "
                f"{source_chunk_id}"
            )

        validated_questions.append(
            QuizQuestion(
                question=question.strip(),
                options=[option.strip() for option in options],
                correct_answer=correct_answer.strip(),
                explanation=explanation.strip(),
                source_chunk_id=source_chunk_id.strip(),
            )
        )

    return Quiz(
        questions=validated_questions,
        topic=topic.strip(),
        difficulty=difficulty,
    )