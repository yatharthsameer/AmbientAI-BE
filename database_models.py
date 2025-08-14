"""
Database models for the Nurse Conversation Processing API.
Defines SQLAlchemy ORM models for storing conversation data and processing results.
"""
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Text, Integer, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now()
    )


class ConversationUpload(Base):
    """Model for storing conversation upload information."""
    
    __tablename__ = "conversation_uploads"
    
    # Basic info
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Processing status
    status: Mapped[str] = mapped_column(
        String(20), 
        default="pending",  # pending, processing, completed, failed
        nullable=False
    )
    
    # Task tracking
    transcription_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    qa_extraction_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Processing metadata
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    transcription: Mapped[Optional["ConversationTranscription"]] = relationship(
        "ConversationTranscription", 
        back_populates="upload",
        cascade="all, delete-orphan"
    )
    qa_results: Mapped[List["QuestionAnswer"]] = relationship(
        "QuestionAnswer", 
        back_populates="upload",
        cascade="all, delete-orphan"
    )


class ConversationTranscription(Base):
    """Model for storing transcription results."""
    
    __tablename__ = "conversation_transcriptions"
    
    # Foreign key
    upload_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation_uploads.id"),
        nullable=False
    )
    
    # Transcription data
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    segments: Mapped[dict] = mapped_column(JSON, nullable=False)  # Whisper segments with timestamps
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    
    # Processing info
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    processing_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Quality metrics
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Relationship
    upload: Mapped["ConversationUpload"] = relationship(
        "ConversationUpload", 
        back_populates="transcription"
    )


class QuestionAnswer(Base):
    """Model for storing question-answer extraction results."""
    
    __tablename__ = "question_answers"
    
    # Foreign key
    upload_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation_uploads.id"),
        nullable=False
    )
    
    # Question info
    question_id: Mapped[str] = mapped_column(String(100), nullable=False)  # from predefined questions
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Answer info
    answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Context and timestamps
    context_start_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    context_end_char: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    timestamp_start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # in seconds
    timestamp_end: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # in seconds
    
    # Source context
    context_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Processing info
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Status - whether answer was found or uncertain
    is_confident: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_manual_review_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Relationship
    upload: Mapped["ConversationUpload"] = relationship(
        "ConversationUpload", 
        back_populates="qa_results"
    )


class ProcessingJob(Base):
    """Model for tracking long-running processing jobs."""
    
    __tablename__ = "processing_jobs"
    
    # Job info
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # transcription, qa_extraction
    task_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    
    # Related entities
    upload_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("conversation_uploads.id"),
        nullable=True
    )
    
    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",  # pending, running, completed, failed, cancelled
        nullable=False
    )
    
    # Progress tracking
    progress_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_step: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    total_steps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Result and error info
    result_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Worker info
    worker_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class ConversationScore(Base):
    """Model for storing overall conversation analysis scores."""
    
    __tablename__ = "conversation_scores"
    
    # Foreign key
    upload_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation_uploads.id"),
        nullable=False,
        unique=True
    )
    
    # Overall scores
    completeness_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1
    information_density_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1
    
    # Category-wise scores
    patient_info_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    medical_history_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    assessment_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    treatment_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    
    # Quality metrics
    questions_answered: Mapped[int] = mapped_column(Integer, nullable=False)
    questions_total: Mapped[int] = mapped_column(Integer, nullable=False)
    high_confidence_answers: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Calculated metrics
    answers_requiring_review: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    transcription_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps for caching
    scores_calculated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(),
        nullable=False
    )