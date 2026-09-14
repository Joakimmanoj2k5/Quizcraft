from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QuizQuestion:
    """Represents a single generated multiple-choice question."""

    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    source_chunk_id: Optional[str] = None


@dataclass
class Quiz:
    """Represents a complete generated quiz."""

    questions: List[QuizQuestion]
    topic: str
    difficulty: str