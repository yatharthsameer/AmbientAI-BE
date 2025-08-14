"""
Core configuration module for the Nurse Conversation Processing API.
Handles environment variables and application settings.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    url: str
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


class RedisSettings(BaseModel):
    """Redis configuration settings."""
    url: str
    max_connections: int = 10
    socket_keepalive: bool = True
    socket_keepalive_options: dict = {}


class CelerySettings(BaseModel):
    """Celery configuration settings."""
    broker_url: str
    result_backend: str
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = ["json"]
    timezone: str = "UTC"
    enable_utc: bool = True
    task_track_started: bool = True
    task_time_limit: int = 3600  # 1 hour
    worker_concurrency: int = 4


class FileUploadSettings(BaseModel):
    """File upload configuration settings."""
    upload_dir: Path
    max_file_size: int
    allowed_audio_formats: List[str]


class ASRSettings(BaseModel):
    """Automatic Speech Recognition settings."""
    whisper_model: str = "base"
    device: str = "cpu"
    language: Optional[str] = None
    task: str = "transcribe"  # or "translate"


class QASettings(BaseModel):
    """Question Answering settings."""
    model_name: str = "distilbert-base-cased-distilled-squad"
    max_question_length: int = 512
    max_context_length: int = 4096
    confidence_threshold: float = 0.5


class GeminiSettings(BaseModel):
    """Google Gemini AI settings."""
    api_key: str
    model_name: str = "gemini-2.0-flash"
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.8
    top_k: int = 40


class OpenAISettings(BaseModel):
    """OpenAI API settings."""
    api_key: str
    model_name: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.1


class SecuritySettings(BaseModel):
    """Security configuration settings."""
    secret_key: str
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "INFO"
    
    # Database
    database_url: str
    
    # Redis
    redis_url: str
    
    # Celery
    celery_broker_url: str
    celery_result_backend: str
    
    # File Upload
    upload_dir: str = "./uploads"
    max_file_size: int = 104857600  # 100MB
    allowed_audio_formats: str = "mp3,wav,m4a,ogg,flac"
    
    # ASR
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    
    # Q&A
    qa_model: str = "distilbert-base-cased-distilled-squad"
    max_question_length: int = 512
    max_context_length: int = 4096
    
    # AI Services
    gemini_api_key: str = ""
    openai_api_key: str = ""
    
    # Security
    secret_key: str
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def database_settings(self) -> DatabaseSettings:
        return DatabaseSettings(url=self.database_url)
    
    @property
    def redis_settings(self) -> RedisSettings:
        return RedisSettings(url=self.redis_url)
    
    @property
    def celery_settings(self) -> CelerySettings:
        return CelerySettings(
            broker_url=self.celery_broker_url,
            result_backend=self.celery_result_backend
        )
    
    @property
    def file_upload_settings(self) -> FileUploadSettings:
        return FileUploadSettings(
            upload_dir=Path(self.upload_dir),
            max_file_size=self.max_file_size,
            allowed_audio_formats=self.allowed_audio_formats.split(",")
        )
    
    @property
    def asr_settings(self) -> ASRSettings:
        return ASRSettings(
            whisper_model=self.whisper_model,
            device=self.whisper_device
        )
    
    @property
    def qa_settings(self) -> QASettings:
        return QASettings(model_name=self.qa_model)
    
    @property
    def gemini_settings(self) -> GeminiSettings:
        return GeminiSettings(api_key=self.gemini_api_key)
    
    @property
    def openai_settings(self) -> OpenAISettings:
        return OpenAISettings(api_key=self.openai_api_key)
    
    @property
    def security_settings(self) -> SecuritySettings:
        return SecuritySettings(
            secret_key=self.secret_key,
            access_token_expire_minutes=self.access_token_expire_minutes
        )


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()


# Predefined questions for nurse conversation analysis
PREDEFINED_QUESTIONS = [
    {
        "id": "patient_name",
        "question": "What is the patient's name?",
        "category": "patient_info"
    },
    {
        "id": "patient_age",
        "question": "What is the patient's age?",
        "category": "patient_info"
    },
    {
        "id": "chief_complaint",
        "question": "What is the patient's chief complaint or main problem?",
        "category": "medical_history"
    },
    {
        "id": "symptoms",
        "question": "What symptoms is the patient experiencing?",
        "category": "medical_history"
    },
    {
        "id": "medications",
        "question": "What medications is the patient currently taking?",
        "category": "medications"
    },
    {
        "id": "allergies",
        "question": "Does the patient have any allergies?",
        "category": "allergies"
    },
    {
        "id": "vital_signs",
        "question": "What are the patient's vital signs?",
        "category": "assessment"
    },
    {
        "id": "pain_level",
        "question": "What is the patient's pain level on a scale of 1 to 10?",
        "category": "assessment"
    },
    {
        "id": "diagnosis",
        "question": "What is the patient's diagnosis?",
        "category": "diagnosis"
    },
    {
        "id": "treatment_plan",
        "question": "What is the treatment plan for the patient?",
        "category": "treatment"
    },
    {
        "id": "discharge_instructions",
        "question": "What are the discharge instructions for the patient?",
        "category": "discharge"
    },
    {
        "id": "follow_up",
        "question": "When should the patient follow up?",
        "category": "follow_up"
    }
]