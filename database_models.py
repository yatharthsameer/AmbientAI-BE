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


class RealTimeSession(Base):
    """Model for storing real-time WebSocket transcription sessions."""

    __tablename__ = "realtime_sessions"

    # Session info
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    client_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Session status
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",  # active, paused, completed, terminated
        nullable=False,
    )

    # Audio configuration
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    chunk_duration: Mapped[float] = mapped_column(
        Float, default=2.0, nullable=False
    )  # seconds

    # Processing settings
    whisper_model: Mapped[str] = mapped_column(
        String(50), default="base", nullable=False
    )
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # Session metrics
    total_audio_duration: Mapped[float] = mapped_column(
        Float, default=0.0, nullable=False
    )
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Full conversation data
    full_transcript: Mapped[str] = mapped_column(Text, default="", nullable=False)
    final_segments: Mapped[dict] = mapped_column(JSON, default=list, nullable=False)

    # Connection info
    connected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    disconnected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    transcript_chunks: Mapped[List["RealTimeTranscriptChunk"]] = relationship(
        "RealTimeTranscriptChunk",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class RealTimeTranscriptChunk(Base):
    """Model for storing individual real-time transcript chunks."""

    __tablename__ = "realtime_transcript_chunks"

    # Foreign key
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("realtime_sessions.id"), nullable=False
    )

    # Chunk info
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # seconds from session start
    end_time: Mapped[float] = mapped_column(Float, nullable=False)

    # Transcript data
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Processing info
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Audio chunk metadata
    audio_duration: Mapped[float] = mapped_column(Float, nullable=False)
    audio_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Status
    is_final: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_corrected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationship
    session: Mapped["RealTimeSession"] = relationship(
        "RealTimeSession", back_populates="transcript_chunks"
    )


class RealTimeSessionMetrics(Base):
    """Model for storing performance metrics of real-time sessions."""

    __tablename__ = "realtime_session_metrics"

    # Foreign key
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("realtime_sessions.id"), nullable=False, unique=True
    )

    # Performance metrics
    avg_processing_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    max_processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    min_processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Audio quality metrics
    avg_confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    successful_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_chunks: Mapped[int] = mapped_column(Integer, nullable=False)

    # Network metrics
    avg_chunk_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    total_data_received_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    connection_drops: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Real-time performance
    real_time_factor: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # processing_time / audio_duration
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Quality assessment
    transcript_completeness: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1
    estimated_accuracy: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # 0-1
