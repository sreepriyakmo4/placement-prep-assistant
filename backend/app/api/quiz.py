"""
Quiz generation and submission endpoints.
POST /quiz/generate/{doc_id}  - Generate 15 MCQs from document
POST /quiz/submit/{doc_id}    - Submit answers and get score + weak topics
GET  /quiz/history/{doc_id}   - Get quiz attempt history for a document
"""

import json
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.base import get_db
from app.db.models import User, Document, Chunk, QuizQuestion, QuizAttempt, QuizAnswer
from app.api.deps import get_current_user
from app.core.config import settings
from groq import Groq

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quiz", tags=["quiz"])

_client = Groq(api_key=settings.GROQ_API_KEY)

# ── Schemas ──────────────────────────────────────────────────────────────

class GeneratedQuestion(BaseModel):
    question: str
    options: List[str]  # ["A) Can be null", "B) Not null and unique", ...]
    correct_answer: str  # "B"
    topic: str  # "Constraints"
    explanation: str


class QuizGenerateResponse(BaseModel):
    questions: List[GeneratedQuestion]
    doc_id: int
    total: int


class UserAnswerSchema(BaseModel):
    question_id: int
    selected_answer: str  # "A", "B", "C", or "D"


class QuizSubmitRequest(BaseModel):
    answers: List[UserAnswerSchema]


class TopicStats(BaseModel):
    topic: str
    correct: int
    total: int
    percentage: float
    is_weak: bool  # < 60% is weak


class WrongQuestion(BaseModel):
    question: str
    user_answer: str
    correct_answer: str
    explanation: str
    topic: str


class QuizResult(BaseModel):
    score: int
    total: int
    percentage: float
    strong_topics: List[TopicStats]  # >= 70%
    weak_topics: List[TopicStats]    # < 60%
    wrong_questions: List[WrongQuestion]


class QuizAttemptOut(BaseModel):
    id: int
    score: int
    total_questions: int
    percentage: float
    created_at: datetime

    class Config:
        from_attributes = True


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/generate/{doc_id}", response_model=QuizGenerateResponse)
def generate_quiz(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate 15 MCQ questions from document chunks.
    Uses Groq to create questions based on the document content.
    Stores questions in DB for reuse and result tracking.
    """
    
    # Verify document ownership
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Fetch top chunks from document (most relevant content)
    chunks = db.query(Chunk).filter(
        Chunk.document_id == doc_id
    ).order_by(Chunk.chunk_index).limit(20).all()
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Document has no content to quiz on")
    
    # Combine chunk content
    context = "\n\n".join([f"[{c.heading or 'Section'}]\n{c.content}" for c in chunks])
    
    # Prompt Groq to generate 15 MCQs
    prompt = f"""You are an expert MCQ question generator for placement preparation.

Based on the following study material, generate exactly 15 multiple-choice questions.

REQUIREMENTS:
- Each question must test a different concept or topic
- Questions should be challenging but answerable from the material
- Each question must have 4 options (A, B, C, D) with exactly one correct answer
- Include an explanation for why the correct answer is right
- Tag each question with its topic (e.g., "Constraints", "Joins", "Indexing")
- Return ONLY valid JSON, no markdown or extra text

FORMAT - Return this exact JSON structure:
[
  {{
    "question": "What is a PRIMARY KEY?",
    "options": ["A) Can be null", "B) Not null and unique", "C) Can repeat values", "D) Optional field"],
    "correct_answer": "B",
    "topic": "Constraints",
    "explanation": "A primary key must be not null, unique, and only one per table"
  }},
  ...15 total...
]

STUDY MATERIAL:
{context}

Generate the 15 questions now:"""

    try:
        response = _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON API. Respond with ONLY valid JSON, no markdown or extra text."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=0.7,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Clean up markdown if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        questions_data = json.loads(result_text.strip())
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Groq response as JSON: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate questions")
    except Exception as e:
        logger.error(f"Groq quiz generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate questions")
    
    # Save questions to DB
    for q_data in questions_data[:15]:  # Ensure exactly 15
        db_question = QuizQuestion(
            document_id=doc_id,
            question=q_data["question"],
            options=json.dumps(q_data["options"]),
            correct_answer=q_data["correct_answer"],
            topic=q_data.get("topic", "General"),
            explanation=q_data.get("explanation", ""),
        )
        db.add(db_question)
    
    db.commit()
    
    # Return generated questions
    parsed_questions = []
    for q_data in questions_data[:15]:
        parsed_questions.append(GeneratedQuestion(
            question=q_data["question"],
            options=q_data["options"],
            correct_answer=q_data["correct_answer"],
            topic=q_data.get("topic", "General"),
            explanation=q_data.get("explanation", ""),
        ))
    
    return QuizGenerateResponse(
        questions=parsed_questions,
        doc_id=doc_id,
        total=len(parsed_questions),
    )


@router.post("/submit/{doc_id}", response_model=QuizResult)
def submit_quiz(
    doc_id: int,
    body: QuizSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit quiz answers and get score + weak topics analysis.
    Stores attempt and answers in DB for future reference.
    """
    
    # Verify document ownership
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Fetch questions from DB
    questions = db.query(QuizQuestion).filter(
        QuizQuestion.document_id == doc_id
    ).all()
    
    if not questions:
        raise HTTPException(status_code=400, detail="Quiz not generated yet")
    
    # Map question_id to question data
    question_map = {q.id: q for q in questions}
    
    # Score the answers
    score = 0
    wrong_questions = []
    topic_stats = {}
    
    for idx, user_ans in enumerate(body.answers):
        if idx >= len(questions):
            continue
            
        question = questions[idx]
        is_correct = user_ans.selected_answer == question.correct_answer
        if is_correct:
            score += 1
        else:
            wrong_questions.append(WrongQuestion(
                question=question.question,
                user_answer=user_ans.selected_answer,
                correct_answer=question.correct_answer,
                explanation=question.explanation or "See study material",
                topic=question.topic,
            ))
        
        # Track topic stats
        if question.topic not in topic_stats:
            topic_stats[question.topic] = {"correct": 0, "total": 0}
        topic_stats[question.topic]["total"] += 1
        if is_correct:
            topic_stats[question.topic]["correct"] += 1
    
    # Save attempt to DB
    attempt = QuizAttempt(
        user_id=current_user.id,
        document_id=doc_id,
        score=score,
        total_questions=len(questions),
    )
    db.add(attempt)
    db.flush()  # Get attempt ID
    
    # Save individual answers
    for idx, user_ans in enumerate(body.answers):
        if idx >= len(questions):
            continue
            
        question = questions[idx]
        is_correct = user_ans.selected_answer == question.correct_answer
        db_answer = QuizAnswer(
            attempt_id=attempt.id,
            question_id=question.id,
            user_answer=user_ans.selected_answer,
            is_correct=is_correct,
            topic=question.topic,
        )
        db.add(db_answer)
    
    db.commit()
    
    # Build topic analysis
    strong_topics = []
    weak_topics = []
    
    for topic, stats in topic_stats.items():
        percentage = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        topic_stat = TopicStats(
            topic=topic,
            correct=stats["correct"],
            total=stats["total"],
            percentage=round(percentage, 1),
            is_weak=percentage < 60,
        )
        
        if percentage >= 70:
            strong_topics.append(topic_stat)
        elif percentage < 60:
            weak_topics.append(topic_stat)
    
    # Sort by score
    strong_topics.sort(key=lambda x: x.percentage, reverse=True)
    weak_topics.sort(key=lambda x: x.percentage)
    
    percentage = round((score / len(questions) * 100), 1) if questions else 0
    
    return QuizResult(
        score=score,
        total=len(questions),
        percentage=percentage,
        strong_topics=strong_topics,
        weak_topics=weak_topics,
        wrong_questions=wrong_questions,
    )


@router.get("/history/{doc_id}", response_model=List[QuizAttemptOut])
def get_quiz_history(
    doc_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all quiz attempts for a document (for user only).
    Shows how user's performance has improved over time.
    """
    
    attempts = db.query(QuizAttempt).filter(
        QuizAttempt.user_id == current_user.id,
        QuizAttempt.document_id == doc_id,
    ).order_by(QuizAttempt.created_at.desc()).all()
    
    result = []
    for attempt in attempts:
        percentage = (attempt.score / attempt.total_questions * 100) if attempt.total_questions > 0 else 0
        result.append(QuizAttemptOut(
            id=attempt.id,
            score=attempt.score,
            total_questions=attempt.total_questions,
            percentage=round(percentage, 1),
            created_at=attempt.created_at,
        ))
    
    return result