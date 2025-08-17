"""
Pydantic schemas for the Nurse Conversation Processing API.
Defines request and response models for API endpoints.
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversationCategory(str, Enum):
    """Question category enumeration."""
    PATIENT_INFO = "patient_info"
    MEDICAL_HISTORY = "medical_history"
    MEDICATIONS = "medications"
    ALLERGIES = "allergies"
    ASSESSMENT = "assessment"
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    DISCHARGE = "discharge"
    FOLLOW_UP = "follow_up"
    PROCEDURES = "procedures"
    REHABILITATION = "rehabilitation"
    PATIENT_FEEDBACK = "patient_feedback"


class SessionStatus(str, Enum):
    """Real-time session status enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: datetime


# Request schemas
class ConversationUploadRequest(BaseModel):
    """Schema for conversation upload request."""
    process_immediately: bool = Field(
        default=True,
        description="Whether to start processing immediately after upload"
    )
    custom_questions: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Custom questions to ask in addition to predefined ones"
    )


class TextOnlyProcessingRequest(BaseModel):
    """Schema for processing text-only input (no audio file)."""
    text: str = Field(
        ...,
        description="Raw text content to process",
        min_length=1,
        max_length=50000
    )
    custom_questions: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Custom questions to ask in addition to predefined ones"
    )


# Response schemas
class TranscriptionSegment(BaseSchema):
    """Schema for a transcription segment with timestamp."""
    id: int
    seek: int
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class ConversationTranscriptionResponse(BaseSchema):
    """Schema for transcription response."""
    id: uuid.UUID
    upload_id: uuid.UUID
    full_text: str
    segments: List[TranscriptionSegment]
    language: Optional[str]
    model_used: str
    processing_time_seconds: float
    confidence_score: Optional[float]
    created_at: datetime


class QuestionAnswerResponse(BaseSchema):
    """Schema for question-answer response."""
    id: uuid.UUID
    upload_id: uuid.UUID
    question_id: str
    question_text: str
    category: ConversationCategory
    answer_text: Optional[str]
    confidence_score: float
    context_start_char: Optional[int]
    context_end_char: Optional[int]
    timestamp_start: Optional[float] = Field(
        None, description="Start timestamp in seconds"
    )
    timestamp_end: Optional[float] = Field(
        None, description="End timestamp in seconds"
    )
    context_snippet: Optional[str]
    model_used: str
    is_confident: bool
    is_manual_review_required: bool
    created_at: datetime


class ConversationScoreResponse(BaseSchema):
    """Schema for conversation score response."""
    id: uuid.UUID
    upload_id: uuid.UUID
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    information_density_score: float = Field(..., ge=0.0, le=1.0)
    patient_info_score: float = Field(..., ge=0.0, le=1.0)
    medical_history_score: float = Field(..., ge=0.0, le=1.0)
    assessment_score: float = Field(..., ge=0.0, le=1.0)
    treatment_score: float = Field(..., ge=0.0, le=1.0)
    questions_answered: int = Field(..., ge=0)
    questions_total: int = Field(..., ge=0)
    high_confidence_answers: int = Field(..., ge=0)
    answers_requiring_review: int = Field(..., ge=0)
    transcription_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    scores_calculated_at: datetime


class ConversationUploadResponse(BaseSchema, TimestampMixin):
    """Schema for conversation upload response."""
    id: uuid.UUID
    original_filename: str
    file_size: int
    content_type: str
    duration_seconds: Optional[float]
    status: ProcessingStatus
    transcription_task_id: Optional[str]
    qa_extraction_task_id: Optional[str]
    error_message: Optional[str]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]


class ProcessingJobResponse(BaseSchema, TimestampMixin):
    """Schema for processing job response."""
    id: uuid.UUID
    job_type: str
    task_id: str
    upload_id: Optional[uuid.UUID]
    status: JobStatus
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    current_step: Optional[str]
    total_steps: Optional[int]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    worker_name: Optional[str]


class CompleteConversationResponse(BaseSchema):
    """Schema for complete conversation processing response."""
    upload: ConversationUploadResponse
    transcription: Optional[ConversationTranscriptionResponse]
    qa_results: List[QuestionAnswerResponse]
    score: Optional[ConversationScoreResponse]
    processing_jobs: List[ProcessingJobResponse]


# Utility response schemas
class UploadStatusResponse(BaseSchema):
    """Schema for upload status check response."""
    upload_id: uuid.UUID
    status: ProcessingStatus
    progress_percentage: float
    current_step: Optional[str]
    error_message: Optional[str]
    estimated_completion_time: Optional[datetime]


class TaskProgressResponse(BaseSchema):
    """Schema for task progress response."""
    task_id: str
    status: JobStatus
    progress: float = Field(..., ge=0.0, le=100.0)
    current_step: Optional[str]
    message: Optional[str]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class QuestionListResponse(BaseSchema):
    """Schema for listing available questions."""
    questions: List[Dict[str, str]] = Field(
        ...,
        description="List of predefined questions with id, question, and category"
    )


class HealthCheckResponse(BaseSchema):
    """Schema for health check response."""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]  # service_name: status


# Error response schemas
class ErrorResponse(BaseSchema):
    """Schema for error responses."""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None


class ValidationErrorResponse(BaseSchema):
    """Schema for validation error responses."""
    error: str = "validation_error"
    message: str
    details: List[Dict[str, Any]]
    timestamp: datetime


# Pagination schemas
class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")


class PaginatedResponse(BaseModel):
    """Generic paginated response schema."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    
    @classmethod
    def create(cls, items: List[Any], total: int, page: int, size: int):
        """Create paginated response."""
        pages = (total + size - 1) // size
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages
        )


# Summary and analytics schemas
class ConversationSummary(BaseSchema):
    """Schema for conversation summary statistics."""
    total_uploads: int
    completed_uploads: int
    failed_uploads: int
    average_processing_time: float
    average_confidence_score: float
    total_questions_processed: int
    high_confidence_answers: int
    most_common_categories: List[Dict[str, Any]]


class SystemMetrics(BaseSchema):
    """Schema for system metrics."""
    active_workers: int
    queue_length: int
    processed_today: int
    average_response_time: float
    error_rate: float
    uptime_seconds: int


# Real-time WebSocket schemas
class RealTimeSessionResponse(BaseSchema):
    """Schema for real-time session information."""

    id: uuid.UUID
    session_id: str
    client_id: Optional[str] = None
    status: SessionStatus
    sample_rate: int
    channels: int
    chunk_duration: float
    whisper_model: str
    language: Optional[str] = None
    total_audio_duration: float
    chunks_processed: int
    full_transcript: str
    connected_at: datetime
    last_activity_at: datetime
    disconnected_at: Optional[datetime] = None


class RealTimeTranscriptChunkResponse(BaseSchema):
    """Schema for real-time transcript chunk."""

    id: uuid.UUID
    chunk_index: int
    start_time: float
    end_time: float
    text: str
    confidence_score: Optional[float] = None
    model_used: str
    processing_time_ms: int
    audio_duration: float
    audio_size_bytes: int
    is_final: bool
    is_corrected: bool
    created_at: datetime


class RealTimeSessionMetricsResponse(BaseSchema):
    """Schema for real-time session performance metrics."""

    session_id: uuid.UUID
    avg_processing_time_ms: float
    max_processing_time_ms: int
    min_processing_time_ms: int
    avg_confidence_score: float
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    avg_chunk_size_bytes: int
    total_data_received_bytes: int
    connection_drops: int
    real_time_factor: float
    latency_ms: int
    transcript_completeness: float
    estimated_accuracy: Optional[float] = None


class WebSocketConnectionConfig(BaseSchema):
    """Schema for WebSocket connection configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 2.0
    whisper_model: str = "base"
    language: Optional[str] = None


class WebSocketMessage(BaseSchema):
    """Base schema for WebSocket messages."""

    type: str
    timestamp: Optional[datetime] = None


class WebSocketConnectedMessage(WebSocketMessage):
    """Schema for WebSocket connection confirmation."""

    type: str = "connected"
    session_id: str
    message: str
    config: WebSocketConnectionConfig


class WebSocketTranscriptMessage(WebSocketMessage):
    """Schema for WebSocket transcript message."""

    type: str = "transcript"
    text: str
    confidence: float
    chunk_index: int
    start_time: float
    end_time: float
    processing_time_ms: int
    model_used: str
    is_final: bool = True


class WebSocketErrorMessage(WebSocketMessage):
    """Schema for WebSocket error message."""

    type: str = "error"
    message: str
    error_code: Optional[str] = None


class WebSocketSessionInfoMessage(WebSocketMessage):
    """Schema for WebSocket session information message."""

    type: str = "session_info"
    session_id: str
    connected_at: datetime
    total_chunks: int
    full_transcript: str
    metrics: Dict[str, Any]


class WebSocketControlMessage(BaseSchema):
    """Schema for WebSocket control messages from client."""

    type: str  # ping, session_info, end_session
    data: Optional[Dict[str, Any]] = None


class ActiveWebSocketSessionsResponse(BaseSchema):
    """Schema for active WebSocket sessions response."""

    active_sessions: int
    max_connections: int
    sessions: List[Dict[str, Any]]
