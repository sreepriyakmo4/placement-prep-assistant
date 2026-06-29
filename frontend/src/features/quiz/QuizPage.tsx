import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { quizApi, documentsApi } from '../../lib/api'
import { useAuth } from '../auth/AuthContext'
import ReactMarkdown from 'react-markdown'
import {
  ChevronLeft, Loader2, CheckCircle, XCircle, BarChart3,
  TrendingUp, TrendingDown, AlertCircle
} from 'lucide-react'
import './QuizPage.css'

interface GeneratedQuestion {
  question: string
  options: string[]
  correct_answer: string
  topic: string
  explanation: string
}

interface QuizState {
  questions: GeneratedQuestion[]
  currentIndex: number
  selectedAnswers: Record<number, string>
  submitted: boolean
}

interface QuizResult {
  score: number
  total: number
  percentage: number
  strong_topics: Array<{
    topic: string
    correct: number
    total: number
    percentage: number
  }>
  weak_topics: Array<{
    topic: string
    correct: number
    total: number
    percentage: number
  }>
  wrong_questions: Array<{
    question: string
    user_answer: string
    correct_answer: string
    explanation: string
    topic: string
  }>
}

interface UserAnswer {
  question_id: number
  selected_answer: string
}

export default function QuizPage() {
  const { docId } = useParams()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [quizState, setQuizState] = useState<QuizState>({
    questions: [],
    currentIndex: 0,
    selectedAnswers: {},
    submitted: false,
  })
  const [result, setResult] = useState<QuizResult | null>(null)
  const [generating, setGenerating] = useState(true)
  const [error, setError] = useState('')

  // Fetch document info
  const { data: docs } = useQuery({
    queryKey: ['documents'],
    queryFn: documentsApi.list,
  })

  const doc = docs?.find((d: any) => d.id === parseInt(docId || '0'))

  // Generate quiz
  useEffect(() => {
    const generateQuiz = async () => {
      if (!docId) return
      try {
        const data = await quizApi.generateQuiz(parseInt(docId))
        setQuizState(prev => ({
          ...prev,
          questions: data.questions,
        }))
      } catch (err: any) {
        setError(err.message || 'Failed to generate quiz')
      } finally {
        setGenerating(false)
      }
    }

    generateQuiz()
  }, [docId])

  const handleSelectAnswer = (letter: string) => {
    if (quizState.submitted) return

    setQuizState(prev => ({
      ...prev,
      selectedAnswers: {
        ...prev.selectedAnswers,
        [prev.currentIndex]: letter,
      },
    }))
  }

  const handleNext = () => {
    if (quizState.currentIndex < quizState.questions.length - 1) {
      setQuizState(prev => ({
        ...prev,
        currentIndex: prev.currentIndex + 1,
      }))
    }
  }

  const handlePrev = () => {
    if (quizState.currentIndex > 0) {
      setQuizState(prev => ({
        ...prev,
        currentIndex: prev.currentIndex - 1,
      }))
    }
  }

  const handleSubmit = async () => {
    try {
      // Build answers array
      const answers: UserAnswer[] = quizState.questions
        .map((q, idx) => ({
          question_id: idx,
          selected_answer: quizState.selectedAnswers[idx] || 'A',
        }))

      const resultData = await quizApi.submitQuiz(parseInt(docId || '0'), answers)
      setResult(resultData)
      setQuizState(prev => ({ ...prev, submitted: true }))
    } catch (err: any) {
      setError(err.message || 'Failed to submit quiz')
    }
  }

  if (generating) {
    return (
      <div className="quiz-container flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-accent animate-spin mx-auto mb-3" />
          <p className="text-gray-300">Generating questions...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="quiz-container flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-10 h-10 text-red-400 mx-auto mb-3" />
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={() => navigate('/chat')}
            className="px-4 py-2 bg-accent hover:bg-accent-dim text-white rounded-lg"
          >
            Go Back
          </button>
        </div>
      </div>
    )
  }

  if (!quizState.questions.length) {
    return (
      <div className="quiz-container flex items-center justify-center">
        <p className="text-gray-400">No questions available</p>
      </div>
    )
  }

  // Results screen
  if (quizState.submitted && result) {
    return <QuizResultsScreen result={result} docId={docId} />
  }

  // Quiz screen
  const currentQuestion = quizState.questions[quizState.currentIndex]
  const selectedAnswer = quizState.selectedAnswers[quizState.currentIndex]
  const totalQuestions = quizState.questions.length
  const progressPercent =
    ((quizState.currentIndex + 1) / totalQuestions) * 100
  const allAnswered = Object.keys(quizState.selectedAnswers).length === totalQuestions

  return (
    <div className="quiz-container">
      {/* Header */}
      <div className="quiz-header">
        <button
          onClick={() => navigate('/chat')}
          className="btn-icon"
          title="Exit quiz"
        >
          <ChevronLeft size={20} />
        </button>
        <div className="quiz-title">
          <h1>{doc?.filename || 'Quiz'}</h1>
          <p className="text-gray-500">
            Question {quizState.currentIndex + 1} of {totalQuestions}
          </p>
        </div>
        <div className="quiz-progress-badge">
          {Math.round(progressPercent)}%
        </div>
      </div>

      {/* Progress bar */}
      <div className="quiz-progress-bar">
        <div
          className="quiz-progress-fill"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Main content */}
      <div className="quiz-content">
        {/* Question card */}
        <div className="quiz-card">
          <div className="question-header">
            <span className="topic-badge">{currentQuestion.topic}</span>
            <span className="question-number">
              Q{quizState.currentIndex + 1}/{totalQuestions}
            </span>
          </div>

          <div className="question-text">
            {currentQuestion.question}
          </div>

          {/* Answer options */}
          <div className="options-grid">
            {['A', 'B', 'C', 'D'].map((letter, idx) => {
              const optionText = currentQuestion.options[idx] || ''
              const isSelected = selectedAnswer === letter
              const cleanText = optionText.replace(/^[A-D]\)\s*/, '')

              return (
                <button
                  key={letter}
                  onClick={() => handleSelectAnswer(letter)}
                  className={`option-button ${isSelected ? 'selected' : ''}`}
                >
                  <div className="option-circle">
                    {letter}
                  </div>
                  <span className="option-text">{cleanText}</span>
                </button>
              )
            })}
          </div>

          {/* Navigation */}
          <div className="quiz-nav">
            <button
              onClick={handlePrev}
              disabled={quizState.currentIndex === 0}
              className="btn-nav"
            >
              ← Previous
            </button>

            {quizState.currentIndex === totalQuestions - 1 ? (
              <button
                onClick={handleSubmit}
                disabled={!allAnswered}
                className={`btn-submit ${allAnswered ? '' : 'disabled'}`}
              >
                {allAnswered ? '✓ Submit Quiz' : 'Answer all questions first'}
              </button>
            ) : (
              <button
                onClick={handleNext}
                className="btn-nav"
              >
                Next →
              </button>
            )}
          </div>
        </div>

        {/* Question map sidebar */}
        <div className="quiz-map">
          <div className="map-title">Questions</div>
          <div className="question-dots">
            {quizState.questions.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setQuizState(prev => ({ ...prev, currentIndex: idx }))}
                className={`dot ${
                  idx === quizState.currentIndex
                    ? 'current'
                    : quizState.selectedAnswers[idx]
                    ? 'answered'
                    : 'unanswered'
                }`}
                title={`Question ${idx + 1}`}
              >
                {idx + 1}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// Results and Review Screen
function QuizResultsScreen({
  result,
  docId,
}: {
  result: QuizResult
  docId?: string
}) {
  const navigate = useNavigate()
  const [reviewMode, setReviewMode] = useState(false)

  const performanceColor =
    result.percentage >= 70
      ? 'text-emerald-400'
      : result.percentage >= 50
      ? 'text-amber-400'
      : 'text-red-400'

  return (
    <div className="quiz-container results-screen">
      <button
        onClick={() => navigate('/chat')}
        className="btn-close"
      >
        <ChevronLeft size={20} />
      </button>

      {!reviewMode ? (
        // Score screen
        <div className="results-card">
          <div className="score-circle">
            <div className={`score-number ${performanceColor}`}>
              {result.score}/{result.total}
            </div>
            <div className="score-percent">{result.percentage}%</div>
          </div>

          <div className="score-message">
            {result.percentage >= 80 && (
              <>
                <h2>🎉 Excellent!</h2>
                <p>You've mastered this material. Great job!</p>
              </>
            )}
            {result.percentage >= 70 && result.percentage < 80 && (
              <>
                <h2>👏 Good Job!</h2>
                <p>You have a solid understanding. Review weak areas.</p>
              </>
            )}
            {result.percentage >= 50 && result.percentage < 70 && (
              <>
                <h2>⚠️ Keep Studying</h2>
                <p>You need more practice. Focus on weak topics.</p>
              </>
            )}
            {result.percentage < 50 && (
              <>
                <h2>💪 More Practice Needed</h2>
                <p>Review the material and try again.</p>
              </>
            )}
          </div>

          {/* Topic breakdown */}
          <div className="topics-breakdown">
            <h3>Topic Performance</h3>

            {result.strong_topics.length > 0 && (
              <div className="topic-section strong">
                <div className="section-title">
                  <TrendingUp size={16} /> Strong Topics
                </div>
                {result.strong_topics.map(topic => (
                  <div key={topic.topic} className="topic-item">
                    <div className="topic-name">{topic.topic}</div>
                    <div className="topic-stats">
                      <span className="score">{topic.correct}/{topic.total}</span>
                      <span className="percentage">{topic.percentage}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {result.weak_topics.length > 0 && (
              <div className="topic-section weak">
                <div className="section-title">
                  <TrendingDown size={16} /> Areas to Improve
                </div>
                {result.weak_topics.map(topic => (
                  <div key={topic.topic} className="topic-item">
                    <div className="topic-name">{topic.topic}</div>
                    <div className="topic-stats">
                      <span className="score">{topic.correct}/{topic.total}</span>
                      <span className="percentage">{topic.percentage}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="results-actions">
            <button
              onClick={() => setReviewMode(true)}
              className="btn-review"
            >
              📋 Review Wrong Answers
            </button>
            <button
              onClick={() => navigate('/chat')}
              className="btn-back"
            >
              Back to Chat
            </button>
          </div>
        </div>
      ) : (
        // Review screen
        <div className="review-card">
          <div className="review-header">
            <button
              onClick={() => setReviewMode(false)}
              className="btn-back-review"
            >
              ← Back to Score
            </button>
            <h2>Review Wrong Answers</h2>
          </div>

          <div className="wrong-answers">
            {result.wrong_questions.length === 0 ? (
              <div className="no-wrong">
                <CheckCircle size={32} className="text-emerald-400" />
                <p>Perfect! You got all questions correct! 🎉</p>
              </div>
            ) : (
              result.wrong_questions.map((wq, idx) => (
                <div key={idx} className="wrong-answer-item">
                  <div className="wa-header">
                    <div className="wa-number">Q{idx + 1}</div>
                    <span className="wa-topic">{wq.topic}</span>
                  </div>

                  <div className="wa-question">{wq.question}</div>

                  <div className="wa-feedback">
                    <div className="feedback-row wrong">
                      <XCircle size={16} className="text-red-400" />
                      <div>
                        <div className="label">Your answer</div>
                        <div className="value">{wq.user_answer}</div>
                      </div>
                    </div>

                    <div className="feedback-row correct">
                      <CheckCircle size={16} className="text-emerald-400" />
                      <div>
                        <div className="label">Correct answer</div>
                        <div className="value">{wq.correct_answer}</div>
                      </div>
                    </div>
                  </div>

                  <div className="wa-explanation">
                    <div className="label">Why?</div>
                    <div className="markdown-body text-sm">
                      <ReactMarkdown>{wq.explanation}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="review-footer">
            <button
              onClick={() => navigate('/chat')}
              className="btn-done"
            >
              Done Reviewing
            </button>
          </div>
        </div>
      )}
    </div>
  )
}