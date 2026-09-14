import pytest
from llm.schemas import Quiz, QuizQuestion
from quiz.scorer import calculate_score

@pytest.fixture
def sample_quiz():
    questions = [
        QuizQuestion(
            question="Q1",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation="Exp 1",
            source_chunk_id="chunk1"
        ),
        QuizQuestion(
            question="Q2",
            options=["W", "X", "Y", "Z"],
            correct_answer="Z",
            explanation="Exp 2",
            source_chunk_id="chunk2"
        ),
        QuizQuestion(
            question="Q3",
            options=["1", "2", "3", "4"],
            correct_answer="2",
            explanation="Exp 3",
            source_chunk_id="chunk3"
        )
    ]
    return Quiz(questions=questions, topic="Test", difficulty="easy")

def test_all_answers_correct(sample_quiz):
    answers = {1: "A", 2: "Z", 3: "2"}
    result = calculate_score(sample_quiz, answers)
    
    assert result["score"] == 3
    assert result["total"] == 3
    assert result["percentage"] == 100.0
    
    for res in result["per_question_results"]:
        assert res["is_correct"] is True

def test_all_answers_incorrect(sample_quiz):
    answers = {1: "B", 2: "W", 3: "1"}
    result = calculate_score(sample_quiz, answers)
    
    assert result["score"] == 0
    assert result["total"] == 3
    assert result["percentage"] == 0.0
    
    for res in result["per_question_results"]:
        assert res["is_correct"] is False

def test_mixed_correct_incorrect_answers(sample_quiz):
    answers = {1: "A", 2: "W", 3: "2"}
    result = calculate_score(sample_quiz, answers)
    
    assert result["score"] == 2
    assert result["total"] == 3
    assert result["percentage"] == (2/3) * 100
    
    assert result["per_question_results"][0]["is_correct"] is True
    assert result["per_question_results"][1]["is_correct"] is False
    assert result["per_question_results"][2]["is_correct"] is True

def test_missing_unanswered_answers(sample_quiz):
    answers = {1: "A"}  # Missing 2 and 3
    result = calculate_score(sample_quiz, answers)
    
    assert result["score"] == 1
    assert result["total"] == 3
    
    assert result["per_question_results"][0]["is_correct"] is True
    assert result["per_question_results"][1]["is_correct"] is False
    assert result["per_question_results"][1]["selected_answer"] is None
    assert result["per_question_results"][2]["is_correct"] is False
    assert result["per_question_results"][2]["selected_answer"] is None

def test_empty_quiz():
    empty_quiz = Quiz(questions=[], topic="Empty", difficulty="hard")
    result = calculate_score(empty_quiz, {})
    
    assert result["score"] == 0
    assert result["total"] == 0
    assert result["percentage"] == 0.0
    assert result["per_question_results"] == []

def test_none_quiz():
    result = calculate_score(None, {})
    assert result["score"] == 0
    assert result["total"] == 0
    assert result["percentage"] == 0.0
    assert result["per_question_results"] == []
