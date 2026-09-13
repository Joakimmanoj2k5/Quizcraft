from typing import Dict, Any, Optional
from llm.schemas import Quiz

def calculate_score(quiz: Quiz, answers: Dict[int, Optional[str]]) -> Dict[str, Any]:
    """
    Calculates the score of a quiz based on user answers.
    
    Args:
        quiz: The Quiz dataclass containing the questions and correct answers.
        answers: A dictionary mapping 1-based question index to the user's selected answer.
                 Missing or None values are treated as incorrect (unanswered).
                 
    Returns:
        A dictionary containing the score, total, percentage, and per-question detailed results.
        Note: The correct answers are extracted from the quiz object and are never exposed
        or evaluated prior to calling this function.
    """
    if not quiz or not quiz.questions:
        return {
            "score": 0,
            "total": 0,
            "percentage": 0.0,
            "per_question_results": []
        }

    total = len(quiz.questions)
    score = 0
    per_question_results = []

    for index, question_item in enumerate(quiz.questions, start=1):
        selected_answer = answers.get(index)
        is_correct = False
        
        if selected_answer is not None and selected_answer == question_item.correct_answer:
            is_correct = True
            score += 1
            
        per_question_results.append({
            "question": question_item.question,
            "selected_answer": selected_answer,
            "correct_answer": question_item.correct_answer,
            "is_correct": is_correct,
            "explanation": question_item.explanation,
            "source_chunk_id": question_item.source_chunk_id
        })

    percentage = (score / total) * 100 if total > 0 else 0.0

    return {
        "score": score,
        "total": total,
        "percentage": percentage,
        "per_question_results": per_question_results
    }
