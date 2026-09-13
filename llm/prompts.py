from typing import List, Dict


def build_quiz_prompt(
    topic: str,
    context_chunks: List[Dict],
    question_count: int = 5,
    difficulty: str = "medium",
) -> str:
    """
    Build a grounded quiz-generation prompt.

    The model must use ONLY the supplied context chunks.
    """

    if not topic or not topic.strip():
        raise ValueError("topic must not be empty.")

    if not context_chunks:
        raise ValueError("context_chunks must not be empty.")

    if question_count <= 0:
        raise ValueError("question_count must be greater than 0.")

    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError(
            "difficulty must be one of: easy, medium, hard."
        )

    context_parts = []

    for chunk in context_chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        source = chunk.get("source", "unknown")
        start_page = chunk.get("start_page")
        end_page = chunk.get("end_page")
        text = chunk.get("text", "")

        page_info = ""
        if start_page is not None:
            if end_page is not None and end_page != start_page:
                page_info = f"Pages {start_page}-{end_page}"
            else:
                page_info = f"Page {start_page}"

        context_parts.append(
            f"""
SOURCE CHUNK:
Chunk ID: {chunk_id}
Source: {source}
{page_info}

CONTENT:
{text}
""".strip()
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""
You are a quiz-generation assistant.

Your task is to generate a multiple-choice quiz about:
TOPIC: {topic}

DIFFICULTY: {difficulty}
NUMBER OF QUESTIONS: {question_count}

IMPORTANT GROUNDING RULES:
1. Use ONLY the information contained in the provided source context.
2. Do NOT use your general knowledge.
3. Do NOT invent, assume, or add facts that are not explicitly supported by the context.
4. Every question must be answerable using the provided context.
5. Every correct answer must be directly supported by the context.
6. Every explanation must be based only on the context.
7. For every question, provide the source chunk ID that supports the question.
8. If the context does not contain enough information to create the requested number of questions, create only questions that can be fully supported by the context.
9. Do not make up information just to reach the requested question count.

OUTPUT FORMAT:
Return ONLY valid JSON.
Do not include Markdown.
Do not include ```json.
Do not include any introductory or explanatory text.

The JSON must have exactly this structure:

{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "questions": [
    {{
      "question": "Question text",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correct_answer": "The exact correct option text",
      "explanation": "Explanation supported by the source context",
      "source_chunk_id": "ID of the supporting source chunk"
    }}
  ]
}}

SOURCE CONTEXT:
{context}
""".strip()