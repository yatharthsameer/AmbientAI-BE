"""
Main FastAPI application for the Nurse Conversation Processing API.
"""
import uuid
import time
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Depends,
    BackgroundTasks,
    status,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import sys

from config import get_settings
from database import get_database_session, init_database, cleanup_database, check_database_health
from celery_app import celery_app, get_task_status
from schemas import (
    ConversationUploadResponse,
    CompleteConversationResponse,
    TextOnlyProcessingRequest,
    HealthCheckResponse,
    TaskProgressResponse,
    QuestionListResponse,
    ErrorResponse,
    ConversationTranscriptionResponse,
    QuestionAnswerResponse,
)
from database_models import ConversationUpload, ConversationTranscription, QuestionAnswer
from transcription import TranscriptionService
from qa_extraction import QAExtractionService
from tasks import process_conversation_complete, process_text_only
from helpers import save_upload_file, get_file_duration
from services.qa_service import QAService, AIProvider
from websocket_transcription import ws_transcription_manager

# Request models for test endpoints
from pydantic import BaseModel

class GeminiTestRequest(BaseModel):
    prompt: str
    provider: AIProvider = AIProvider.GEMINI

class MedicalExtractionRequest(BaseModel):
    conversation_text: str
    provider: AIProvider = AIProvider.GEMINI


# Initialize settings
settings = get_settings()

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=settings.log_level
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Nurse Conversation Processing API...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # Test Celery connection
        celery_app.control.ping(timeout=5)
        logger.info("Celery connection established")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    logger.info("API startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Nurse Conversation Processing API...")
    await cleanup_database()
    logger.info("API shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Nurse Conversation Processing API",
    description="A scalable backend for processing nurse conversations with automatic transcription and Q&A extraction",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from datetime import timezone
    error_response = ErrorResponse(
        error="http_error",
        message=str(exc.detail),
        timestamp=datetime.now(timezone.utc)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    from datetime import timezone
    error_response = ErrorResponse(
        error="internal_server_error",
        message="An internal server error occurred",
        timestamp=datetime.now(timezone.utc)
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode='json')
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Check API and service health."""
    try:
        # Check database
        db_health = await check_database_health()
        
        # Check Celery
        celery_health = True
        try:
            celery_app.control.ping(timeout=5)
        except Exception:
            celery_health = False
        
        services = {
            "database": "healthy" if db_health["connected"] else "unhealthy",
            "celery": "healthy" if celery_health else "unhealthy"
        }
        
        overall_status = "healthy" if all(s == "healthy" for s in services.values()) else "unhealthy"
        
        from datetime import timezone
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            services=services
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


# Upload audio file endpoint
@app.post("/api/v1/uploads", response_model=ConversationUploadResponse)
async def upload_conversation(
    file: UploadFile = File(...),
    process_immediately: bool = True,
    db: AsyncSession = Depends(get_database_session),
    background_tasks: BackgroundTasks = None
):
    """Upload an audio file for processing."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file type
        allowed_formats = settings.file_upload_settings.allowed_audio_formats
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in allowed_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_formats)}"
            )
        
        # Check file size
        if file.size and file.size > settings.file_upload_settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail="File too large"
            )
        
        # Save file
        file_path = await save_upload_file(file, settings.file_upload_settings.upload_dir)
        
        # Get file duration
        duration = get_file_duration(file_path)
        
        # Create database record
        upload = ConversationUpload(
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=file.size or 0,
            content_type=file.content_type or "audio/unknown",
            duration_seconds=duration,
            status="pending"
        )
        
        db.add(upload)
        await db.commit()
        await db.refresh(upload)
        
        # Start processing if requested
        if process_immediately:
            task = process_conversation_complete.delay(str(upload.id))
            upload.transcription_task_id = task.id
            await db.commit()
        
        logger.info(f"File uploaded successfully: {upload.id}")
        
        return ConversationUploadResponse.from_orm(upload)
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Process text-only endpoint
@app.post("/api/v1/text-processing", response_model=dict)
async def process_text_only_endpoint(
    request: TextOnlyProcessingRequest,
    db: AsyncSession = Depends(get_database_session)
):
    """Process text-only input without audio file."""
    try:
        # Create database record for text processing
        upload = ConversationUpload(
            original_filename="text_input.txt",
            file_path="",
            file_size=len(request.text.encode('utf-8')),
            content_type="text/plain",
            duration_seconds=0,
            status="pending"
        )
        
        db.add(upload)
        await db.commit()
        await db.refresh(upload)
        
        # Start text processing
        task = process_text_only.delay(str(upload.id), request.text, request.custom_questions)
        upload.qa_extraction_task_id = task.id
        await db.commit()
        
        return {
            "upload_id": str(upload.id),
            "task_id": task.id,
            "status": "processing",
            "message": "Text processing started"
        }
    
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get upload results
@app.get("/api/v1/uploads/{upload_id}", response_model=CompleteConversationResponse)
async def get_upload_results(
    upload_id: str,
    db: AsyncSession = Depends(get_database_session)
):
    """Get complete results for an upload."""
    try:
        # Validate UUID
        try:
            uuid.UUID(upload_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid upload ID")
        
        # Get upload record with relationships loaded
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        # Use selectinload to eagerly load relationships
        stmt = select(ConversationUpload).options(
            selectinload(ConversationUpload.transcription),
            selectinload(ConversationUpload.qa_results)
        ).where(ConversationUpload.id == uuid.UUID(upload_id))
        
        result = await db.execute(stmt)
        upload = result.scalar_one_or_none()
        
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        # Get transcription
        transcription = None
        if upload.transcription:
            transcription = ConversationTranscriptionResponse.model_validate(upload.transcription)
        
        # Get Q&A results
        qa_results = []
        for qa in upload.qa_results:
            qa_results.append(QuestionAnswerResponse.model_validate(qa))
        
        # Get score if available
        score = None
        # Implementation depends on your scoring system
        
        # Convert the response to a dict and handle datetime serialization
        response = CompleteConversationResponse(
            upload=ConversationUploadResponse.model_validate(upload),
            transcription=transcription,
            qa_results=qa_results,
            score=score,
            processing_jobs=[]  # Implementation depends on your job tracking
        )
        
        # Convert to dict and handle datetime serialization
        response_dict = response.model_dump()
        
        # Convert datetime objects to ISO format strings
        def convert_datetime(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, datetime):
                        obj[key] = value.isoformat()
                    elif isinstance(value, dict):
                        convert_datetime(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                convert_datetime(item)
            return obj
        
        response_dict = convert_datetime(response_dict)
        return response_dict
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get task status
@app.get("/api/v1/tasks/{task_id}", response_model=TaskProgressResponse)
async def get_task_status_endpoint(task_id: str):
    """Get the status of a processing task."""
    try:
        result = get_task_status(task_id)

        state = result.get("state", "PENDING")
        state_map = {
            "PENDING": "pending",
            "PROGRESS": "running",
            "STARTED": "running",
            "SUCCESS": "completed",
            "FAILURE": "failed",
            "REVOKED": "cancelled",
            "NOT_FOUND": "failed",
        }
        mapped_status = state_map.get(state, "pending")

        if state == "NOT_FOUND":
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Task ID not found")

        return TaskProgressResponse(
            task_id=task_id,
            status=mapped_status,
            progress=float(result.get("progress", 0)),
            current_step=result.get("message", ""),
            message=result.get("message"),
            result=result.get("result"),
            error=result.get("error")
        )
    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get predefined questions
@app.get("/api/v1/questions", response_model=QuestionListResponse)
async def get_predefined_questions():
    """Get list of predefined questions."""
    from config import PREDEFINED_QUESTIONS
    
    return QuestionListResponse(
        questions=PREDEFINED_QUESTIONS
    )


# List uploads with pagination
@app.get("/api/v1/uploads", response_model=dict)
async def list_uploads(
    page: int = 1,
    size: int = 20,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_database_session)
):
    """List uploads with pagination and filtering."""
    try:
        # Implementation depends on your pagination needs
        # This is a basic example
        from sqlalchemy import select, func
        
        query = select(ConversationUpload)
        if status:
            query = query.where(ConversationUpload.status == status)
        
        # Get total count
        count_query = select(func.count(ConversationUpload.id))
        if status:
            count_query = count_query.where(ConversationUpload.status == status)
        
        total = await db.scalar(count_query)
        
        # Get paginated results
        offset = (page - 1) * size
        query = query.offset(offset).limit(size).order_by(ConversationUpload.created_at.desc())
        
        uploads = await db.scalars(query)
        
        return {
            "items": [ConversationUploadResponse.model_validate(upload) for upload in uploads],
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size
        }
    
    except Exception as e:
        logger.error(f"Error listing uploads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cancel task
@app.post("/api/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    try:
        from celery_app import cancel_task
        
        success = cancel_task(task_id)
        if success:
            return {"message": "Task cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel task")
    
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoints (add authentication in production)
@app.get("/api/v1/admin/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        from celery_app import get_worker_stats
        
        worker_stats = get_worker_stats()
        
        return {
            "workers": worker_stats,
            "timestamp": time.time()
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/test/gemini", response_model=dict)
async def test_gemini(request: GeminiTestRequest):
    """
    Test Gemini/OpenAI API integration.
    
    Args:
        request: Request containing prompt and provider
    
    Returns:
        AI generated response
    """
    try:
        settings = get_settings()
        
        # Initialize QA service
        qa_service = QAService(
            gemini_settings=settings.gemini_settings if settings.gemini_settings.api_key else None,
            openai_settings=settings.openai_settings if settings.openai_settings.api_key else None,
            default_provider=request.provider
        )
        
        if not qa_service.is_provider_available(request.provider):
            raise HTTPException(
                status_code=400,
                detail=f"Provider {request.provider} not available. Check API keys."
            )
        
        # Generate content
        result = await qa_service.generate_content(
            prompt=request.prompt,
            system_prompt="You are a helpful AI assistant. Provide clear, accurate responses.",
            provider=request.provider
        )
        
        return {
            "success": True,
            "provider": result["provider"],
            "model": result["model"],
            "response": result["text"],
            "usage": result["usage"],
            "finish_reason": result["finish_reason"]
        }
        
    except Exception as e:
        logger.error(f"Gemini test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"AI service test failed: {str(e)}"
        )

@app.post("/api/v1/test/medical-extraction", response_model=dict)
async def test_medical_extraction(request: MedicalExtractionRequest):
    """
    Test medical information extraction using AI services.
    
    Args:
        request: Request containing conversation text and provider
    
    Returns:
        Extracted medical information
    """
    try:
        settings = get_settings()

        # Initialize QA service
        qa_service = QAService(
            gemini_settings=settings.gemini_settings if settings.gemini_settings.api_key else None,
            openai_settings=settings.openai_settings if settings.openai_settings.api_key else None,
            default_provider=request.provider
        )

        if not qa_service.is_provider_available(request.provider):
            raise HTTPException(
                status_code=400,
                detail=f"Provider {request.provider} not available. Check API keys."
            )

        # Sample medical questions
        sample_questions = [
            {"id": "patient_name", "question": "What is the patient's name?"},
            {"id": "chief_complaint", "question": "What is the patient's chief complaint?"},
            {"id": "symptoms", "question": "What symptoms is the patient experiencing?"},
            {"id": "medications", "question": "What medications is the patient taking?"}
        ]

        # Extract medical information
        result = await qa_service.extract_medical_info(
            conversation_text=request.conversation_text,
            questions=sample_questions,
            provider=request.provider
        )

        return {
            "success": True,
            "provider": request.provider,
            "extracted_info": result,
            "questions_processed": len(sample_questions)
        }

    except Exception as e:
        logger.error(f"Medical extraction test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Medical extraction test failed: {str(e)}"
        )


# WebSocket endpoints for real-time transcription


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription.

    Protocol:
    - Send binary audio data (PCM 16-bit, 16kHz, mono)
    - Receive JSON messages with transcription results
    - Send text messages for control (ping, session_info, end_session)
    """
    session_id = str(uuid.uuid4())

    try:
        # Connect to WebSocket transcription manager
        connection_success = await ws_transcription_manager.connect(
            websocket, session_id
        )

        if not connection_success:
            logger.warning(f"WebSocket connection rejected for session {session_id}")
            return

        logger.info(f"WebSocket session started: {session_id}")

        while True:
            try:
                # Check WebSocket state before receiving
                if websocket.client_state.name == "DISCONNECTED":
                    logger.info(
                        f"WebSocket client disconnected (state check): {session_id}"
                    )
                    break

                # Check if we should receive binary or text data
                message = await websocket.receive()

                # Check for disconnect message first
                if message.get("type") == "websocket.disconnect":
                    logger.info(
                        f"WebSocket disconnect message received for {session_id}"
                    )
                    break

                if "bytes" in message:
                    # Handle binary audio data
                    audio_data = message["bytes"]
                    await ws_transcription_manager.process_audio_data(
                        session_id, audio_data
                    )

                elif "text" in message:
                    # Handle text control messages
                    try:
                        import json

                        text_message = json.loads(message["text"])
                        await ws_transcription_manager.handle_text_message(
                            session_id, text_message
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Invalid JSON message from {session_id}: {message['text']}"
                        )
                        await ws_transcription_manager.send_message(
                            session_id,
                            {"type": "error", "message": "Invalid JSON format"},
                        )

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {session_id}")
                break

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing WebSocket message for {session_id}: {error_msg}"
                )

                # Check if this is a disconnect-related error (break the loop immediately)
                if "disconnect" in error_msg.lower() or "receive" in error_msg.lower():
                    logger.info(
                        f"WebSocket connection closed for {session_id}: {error_msg}"
                    )
                    break

                # For other errors, try to send error message
                try:
                    await ws_transcription_manager.send_message(
                        session_id,
                        {
                            "type": "error",
                            "message": f"Message processing failed: {error_msg}",
                        },
                    )
                except:
                    # If we can't send the error message, the connection is likely broken
                    logger.warning(
                        f"Could not send error message to {session_id}, connection likely broken"
                    )
                    break

    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")

    finally:
        # Clean up connection
        await ws_transcription_manager.disconnect(session_id, "Connection closed")


@app.websocket("/ws/transcribe/{session_id}")
async def websocket_transcribe_session(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for resuming or connecting to a specific session.

    Args:
        session_id: Existing session ID to resume or new custom session ID
    """
    try:
        # Validate session ID format
        if not session_id or len(session_id.strip()) == 0:
            await websocket.close(code=1008, reason="Invalid session ID")
            return

        # Connect to WebSocket transcription manager
        connection_success = await ws_transcription_manager.connect(
            websocket, session_id
        )

        if not connection_success:
            logger.warning(f"WebSocket connection rejected for session {session_id}")
            return

        logger.info(f"WebSocket session connected with custom ID: {session_id}")

        while True:
            try:
                # Check WebSocket state before receiving
                if websocket.client_state.name == "DISCONNECTED":
                    logger.info(
                        f"WebSocket client disconnected (state check): {session_id}"
                    )
                    break

                message = await websocket.receive()

                # Check for disconnect message first
                if message.get("type") == "websocket.disconnect":
                    logger.info(
                        f"WebSocket disconnect message received for {session_id}"
                    )
                    break

                if "bytes" in message:
                    # Handle binary audio data
                    audio_data = message["bytes"]
                    await ws_transcription_manager.process_audio_data(
                        session_id, audio_data
                    )

                elif "text" in message:
                    # Handle text control messages
                    try:
                        import json

                        text_message = json.loads(message["text"])
                        await ws_transcription_manager.handle_text_message(
                            session_id, text_message
                        )
                    except json.JSONDecodeError:
                        await ws_transcription_manager.send_message(
                            session_id,
                            {"type": "error", "message": "Invalid JSON format"},
                        )

            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {session_id}")
                break

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error processing WebSocket message for {session_id}: {error_msg}"
                )

                # Check if this is a disconnect-related error (break the loop immediately)
                if "disconnect" in error_msg.lower() or "receive" in error_msg.lower():
                    logger.info(
                        f"WebSocket connection closed for {session_id}: {error_msg}"
                    )
                    break

                # For other errors, try to send error message
                try:
                    await ws_transcription_manager.send_message(
                        session_id,
                        {
                            "type": "error",
                            "message": f"Message processing failed: {error_msg}",
                        },
                    )
                except:
                    # If we can't send the error message, the connection is likely broken
                    logger.warning(
                        f"Could not send error message to {session_id}, connection likely broken"
                    )
                    break

    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")

    finally:
        # Clean up connection
        await ws_transcription_manager.disconnect(session_id, "Connection closed")


# WebSocket management endpoints


@app.get("/api/v1/websocket/sessions")
async def get_active_websocket_sessions():
    """Get information about all active WebSocket sessions."""
    try:
        sessions = ws_transcription_manager.get_active_sessions()

        return {
            "active_sessions": len(sessions),
            "max_connections": ws_transcription_manager.max_connections,
            "sessions": sessions,
        }

    except Exception as e:
        logger.error(f"Error getting WebSocket sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/websocket/sessions/{session_id}/end")
async def end_websocket_session(session_id: str):
    """
    Forcefully end a WebSocket session.

    Args:
        session_id: Session ID to terminate
    """
    try:
        if session_id in ws_transcription_manager.active_connections:
            await ws_transcription_manager.disconnect(session_id, "Terminated by API")
            return {"message": f"Session {session_id} terminated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    except Exception as e:
        logger.error(f"Error ending WebSocket session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/websocket/broadcast")
async def broadcast_to_websocket_sessions(message: dict):
    """
    Broadcast a message to all active WebSocket sessions.

    Args:
        message: Message dictionary to broadcast
    """
    try:
        await ws_transcription_manager.broadcast_message(message)

        return {
            "message": "Broadcast sent successfully",
            "recipients": len(ws_transcription_manager.active_connections),
        }

    except Exception as e:
        logger.error(f"Error broadcasting WebSocket message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # Use 1 worker when running directly
        log_level=settings.log_level.lower()
    )
