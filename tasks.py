"""
Celery tasks for the Nurse Conversation Processing API.
Handles background processing of audio files and conversation analysis.
"""
import uuid
import time
import asyncio

from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from celery import current_task
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from sqlalchemy import select
from datetime import timedelta

# Note: nest_asyncio is not compatible with uvloop, so we'll handle async operations differently

from celery_app import celery_app, CallbackTask, get_retry_kwargs
from database import db_manager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config import get_settings, PREDEFINED_QUESTIONS
from database_models import (
    ConversationUpload, 
    ConversationTranscription, 
    QuestionAnswer,
    ConversationScore,
    ProcessingJob
)
from transcription import TranscriptionService
from qa_extraction import QAExtractionService
from schemas import ProcessingStatus, JobStatus


def run_async(coro):
    """Run async coroutine in a fresh event loop to avoid cross-loop issues."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


# Get settings
settings = get_settings()


@celery_app.task(
    bind=True,
    base=CallbackTask,
    **get_retry_kwargs(max_retries=3, countdown=60)
)
def transcribe_audio(self, upload_id: str, model_name: str = None):
    """
    Transcribe audio file using OpenAI Whisper.
    
    Args:
        upload_id: UUID of the conversation upload
        model_name: Optional Whisper model name
    """
    logger.info(f"Starting transcription for upload {upload_id}")
    
    try:
        self.update_progress(0, 100, "Loading audio file...")
        
        # Get upload record from database (sync version for Celery)
        async def get_upload():
            async with db_manager.get_session() as session:
                result = await session.execute(
                    select(ConversationUpload).where(ConversationUpload.id == uuid.UUID(upload_id))
                )
                return result.scalar_one_or_none()
        
        upload = run_async(get_upload())
        if not upload:
            raise ValueError(f"Upload {upload_id} not found")
        
        # Update status to processing
        async def update_upload_status(status: str, error: str = None):
            async with db_manager.get_session() as session:
                upload.status = status
                if status == "processing":
                    upload.processing_started_at = datetime.now()
                elif status in ["completed", "failed"]:
                    upload.processing_completed_at = datetime.now()
                if error:
                    upload.error_message = error
                await session.commit()
        
        run_async(update_upload_status("processing"))
        
        # Initialize transcription service
        transcription_service = TranscriptionService()
        
        self.update_progress(20, 100, "Transcribing audio...")
        
        # Transcribe audio
        transcription_result = transcription_service.transcribe_audio(
            upload.file_path,
            model_name=model_name,
            beam_size=5  # Add beam_size for proper Whisper configuration
        )
        
        self.update_progress(60, 100, "Saving transcription...")
        
        # Save transcription
        async def save_transcription():
            async with db_manager.get_session() as session:
                transcription = ConversationTranscription(
                    upload_id=uuid.UUID(upload_id),
                    full_text=transcription_result["text"],
                    segments=transcription_result["segments"],
                    language=transcription_result.get("language"),
                    model_used=transcription_result["model_used"],
                    processing_time_seconds=transcription_result["processing_time_seconds"],
                    confidence_score=transcription_result.get("confidence_score")
                )
                
                session.add(transcription)
                await session.commit()
                return transcription
        
        transcription = run_async(save_transcription())
        
        # Update status to completed
        run_async(update_upload_status("completed"))
        
        self.update_progress(100, 100, "Transcription completed successfully")
        
        logger.info(f"Transcription completed for upload {upload_id}")
        
        # Trigger workflow callback for next step
        try:
            workflow_callback.delay(upload_id, "transcription", transcription_result)
        except Exception as callback_error:
            logger.warning(f"Failed to trigger workflow callback: {callback_error}")
        
        return transcription_result
        
    except Exception as e:
        logger.error(f"Transcription failed for upload {upload_id}: {e}")
        run_async(update_upload_status("failed", str(e)))
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    **get_retry_kwargs(max_retries=3, countdown=60)
)
def extract_qa_answers(
    self, 
    upload_id: str, 
    transcription_text: str = None,
    segments: List[Dict] = None,
    custom_questions: List[Dict] = None
):
    """
    Extract Q&A answers from transcription.
    
    Args:
        upload_id: UUID of the conversation upload
        transcription_text: Full transcription text
        segments: Transcription segments with timestamps
        custom_questions: Optional custom questions to add
    """
    logger.info(f"Starting Q&A extraction for upload {upload_id}")
    
    try:
        self.update_progress(0, 100, "Initializing Q&A service...")
        
        # Get transcription if not provided
        if not transcription_text:
            async def get_transcription():
                async with db_manager.get_session() as session:
                    result = await session.execute(
                        select(ConversationTranscription).where(
                            ConversationTranscription.upload_id == uuid.UUID(upload_id)
                        )
                    )
                    transcription = result.scalar_one_or_none()
                    if transcription:
                        return transcription.full_text, transcription.segments
                    return None, None
            
            transcription_text, segments = run_async(get_transcription())
            
            if not transcription_text:
                raise ValueError(f"No transcription found for upload {upload_id}")
        
        # Initialize Q&A service
        qa_service = QAExtractionService()
        
        # Prepare questions
        self.update_progress(20, 100, "Preparing questions...")
        questions = PREDEFINED_QUESTIONS.copy()
        if custom_questions:
            questions.extend(custom_questions)
        
        # Extract answers
        self.update_progress(30, 100, "Extracting answers...")
        answers = run_async(qa_service.extract_multiple_answers(
            questions=questions,
            context=transcription_text,
            segments=segments
        ))
        
        # Save answers to database
        self.update_progress(80, 100, "Saving Q&A results...")
        
        async def save_answers():
            async with db_manager.get_session() as session:
                saved_answers = []
                
                for answer_data in answers:
                    # Determine context character positions
                    context_start = None
                    context_end = None
                    if answer_data.get("answer") and transcription_text:
                        answer_pos = transcription_text.lower().find(answer_data["answer"].lower())
                        if answer_pos != -1:
                            context_start = answer_pos
                            context_end = answer_pos + len(answer_data["answer"])
                    
                    # Create Q&A record
                    qa_record = QuestionAnswer(
                        upload_id=uuid.UUID(upload_id),
                        question_id=answer_data["question_id"],
                        question_text=answer_data["question_text"],
                        category=answer_data["category"],
                        answer_text=answer_data["answer"],
                        confidence_score=answer_data["confidence_score"],
                        context_start_char=context_start,
                        context_end_char=context_end,
                        timestamp_start=answer_data.get("timestamp_start"),
                        timestamp_end=answer_data.get("timestamp_end"),
                        context_snippet=answer_data.get("context_snippet"),
                        model_used=answer_data.get("model_used", "unknown"),
                        processing_time_seconds=answer_data.get("processing_time_seconds"),
                        is_confident=answer_data["is_confident"],
                        is_manual_review_required=not answer_data["is_confident"]
                    )
                    
                    session.add(qa_record)
                    saved_answers.append(qa_record)
                
                await session.commit()
                # Attach RAG and processing metadata via in-memory echo (not persisted fields)
                return saved_answers
        
        saved_answers = run_async(save_answers())
        
        self.update_progress(100, 100, "Q&A extraction completed successfully")
        
        logger.info(f"Q&A extraction completed for upload {upload_id} - {len(saved_answers)} answers")
        
        # Trigger workflow callback for next step
        try:
            workflow_callback.delay(upload_id, "qa_extraction", {
                "answers_count": len(saved_answers),
                "confident_answers": len([a for a in saved_answers if a.is_confident]),
                "review_required": len([a for a in saved_answers if a.is_manual_review_required])
            })
        except Exception as callback_error:
            logger.warning(f"Failed to trigger workflow callback: {callback_error}")
        
        return {
            "answers_count": len(saved_answers),
            "confident_answers": len([a for a in saved_answers if a.is_confident]),
            "review_required": len([a for a in saved_answers if a.is_manual_review_required])
        }
        
    except Exception as e:
        logger.error(f"Q&A extraction failed for upload {upload_id}: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    **get_retry_kwargs(max_retries=2, countdown=30)
)
def calculate_conversation_score(self, upload_id: str):
    """
    Calculate overall conversation analysis score.
    
    Args:
        upload_id: UUID of the conversation upload
    """
    logger.info(f"Calculating conversation score for upload {upload_id}")
    
    try:
        async def calculate_score():
            async with db_manager.get_session() as session:
                # Get Q&A results
                result = await session.execute(
                    select(QuestionAnswer).where(
                        QuestionAnswer.upload_id == uuid.UUID(upload_id)
                    )
                )
                qa_results = result.scalars().all()
                
                if not qa_results:
                    logger.warning(f"No Q&A results found for upload {upload_id}")
                    return None
                
                # Calculate scores
                total_questions = len(qa_results)
                answered_questions = len([qa for qa in qa_results if qa.answer_text])
                high_confidence = len([qa for qa in qa_results if qa.confidence_score > 0.7])
                
                completeness_score = answered_questions / total_questions if total_questions > 0 else 0
                confidence_score = sum(qa.confidence_score for qa in qa_results) / total_questions if total_questions > 0 else 0
                
                # Create score record
                score_record = ConversationScore(
                    upload_id=uuid.UUID(upload_id),
                    completeness_score=completeness_score,
                    confidence_score=confidence_score,
                    information_density_score=min(completeness_score * confidence_score, 1.0),
                    patient_info_score=0.8,  # Placeholder - implement actual calculation
                    medical_history_score=0.7,  # Placeholder
                    assessment_score=0.6,  # Placeholder
                    treatment_score=0.5,  # Placeholder
                    questions_answered=answered_questions,
                    questions_total=total_questions,
                    high_confidence_answers=high_confidence,
                    answers_requiring_review=total_questions - high_confidence
                )
                
                session.add(score_record)
                await session.commit()
                return score_record
        
        score = run_async(calculate_score())
        
        if score is None:
            logger.warning(f"No score calculated for upload {upload_id}")
            return None
            
        logger.info(f"Conversation score calculated for upload {upload_id}")
        
        # Trigger workflow callback for completion
        try:
            workflow_callback.delay(upload_id, "scoring", {
                "completeness_score": score.completeness_score,
                "confidence_score": score.confidence_score,
                "questions_answered": score.questions_answered,
                "questions_total": score.questions_total
            })
        except Exception as callback_error:
            logger.warning(f"Failed to trigger workflow callback: {callback_error}")
        
        return {
            "completeness_score": score.completeness_score,
            "confidence_score": score.confidence_score,
            "questions_answered": score.questions_answered,
            "questions_total": score.questions_total
        }
        
    except Exception as e:
        logger.error(f"Score calculation failed for upload {upload_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def process_conversation_complete(self, upload_id: str):
    """
    Complete conversation processing pipeline: transcription -> Q&A -> scoring.
    
    Args:
        upload_id: UUID of the conversation upload
    """
    logger.info(f"Starting complete conversation processing for upload {upload_id}")
    
    try:
        self.update_progress(0, 100, "Starting transcription...")
        
        # Step 1: Start transcription (non-blocking)
        transcription_result = transcribe_audio.delay(upload_id)
        
        # Store the task ID for tracking
        self.update_progress(20, 100, "Transcription started...")
        
        # Return immediately with task IDs for tracking
        return {
            "status": "started",
            "message": "Processing pipeline initiated",
            "tasks": {
                "transcription": transcription_result.id,
                "next_step": "qa_extraction",
                "upload_id": upload_id
            }
        }
        
    except Exception as e:
        logger.error(f"Complete conversation processing failed for upload {upload_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def process_conversation_step(self, upload_id: str, step: str, previous_result: dict = None):
    """
    Process individual steps in the conversation pipeline.
    
    Args:
        upload_id: UUID of the conversation upload
        step: Current step to process
        previous_result: Result from previous step
    """
    logger.info(f"Processing step '{step}' for upload {upload_id}")
    
    try:
        if step == "transcription":
            self.update_progress(40, 100, "Starting Q&A extraction...")
            # Start Q&A extraction
            qa_result = extract_qa_answers.delay(upload_id)
            return {
                "status": "in_progress",
                "step": "qa_extraction",
                "task_id": qa_result.id,
                "upload_id": upload_id
            }
            
        elif step == "qa_extraction":
            self.update_progress(80, 100, "Starting Q&A extraction...")
            # Actually run Q&A extraction instead of just scheduling it
            qa_result = extract_qa_answers.delay(upload_id)
            return {
                "status": "in_progress",
                "step": "qa_extraction",
                "task_id": qa_result.id,
                "upload_id": upload_id
            }
            
        elif step == "scoring":
            self.update_progress(100, 100, "Processing completed successfully")
            logger.info(f"Complete conversation processing finished for upload {upload_id}")
            return {
                "status": "completed",
                "message": "All processing steps completed",
                "upload_id": upload_id
            }
            
        else:
            raise ValueError(f"Unknown step: {step}")
            
    except Exception as e:
        logger.error(f"Step '{step}' processing failed for upload {upload_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def workflow_callback(self, upload_id: str, step: str, task_result: dict = None):
    """
    Callback task to handle workflow progression after each step completes.
    
    Args:
        upload_id: UUID of the conversation upload
        step: Completed step
        task_result: Result from the completed step
    """
    logger.info(f"Workflow callback for step '{step}' of upload {upload_id}")
    
    try:
        # Update progress based on completed step
        if step == "transcription":
            self.update_progress(30, 100, "Transcription completed, starting Q&A...")
            # Start next step
            process_conversation_step.delay(upload_id, "qa_extraction")
            
        elif step == "qa_extraction":
            self.update_progress(70, 100, "Q&A completed, starting scoring...")
            # Start next step
            process_conversation_step.delay(upload_id, "scoring")
            
        elif step == "scoring":
            self.update_progress(100, 100, "All processing completed!")
            logger.info(f"Workflow completed for upload {upload_id}")
            
        return {
            "status": "callback_processed",
            "step": step,
            "upload_id": upload_id,
            "next_action": "workflow_progression"
        }
        
    except Exception as e:
        logger.error(f"Workflow callback failed for step '{step}' of upload {upload_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def process_text_only(self, upload_id: str, text: str, custom_questions: List[Dict] = None):
    """
    Process text-only input without audio transcription.
    
    Args:
        upload_id: UUID of the conversation upload
        text: Raw text to process
        custom_questions: Optional custom questions
    """
    logger.info(f"Starting text-only processing for upload {upload_id}")
    
    try:
        self.update_progress(0, 100, "Processing text input...")
        
        # Create fake transcription from text
        transcription_service = TranscriptionService()
        transcription_result = transcription_service.transcribe_text_only(text)
        
        # Save transcription using helper that binds to the Celery process loop
        async def save_transcription():
            async with db_manager.get_session() as session:
                transcription = ConversationTranscription(
                    upload_id=uuid.UUID(upload_id),
                    full_text=transcription_result["text"],
                    segments=transcription_result["segments"],
                    language=transcription_result.get("language"),
                    model_used=transcription_result["model_used"],
                    processing_time_seconds=transcription_result["processing_time_seconds"],
                    confidence_score=transcription_result.get("confidence_score")
                )
                session.add(transcription)
                await session.commit()
                return transcription

        transcription = run_async(save_transcription())
        
        self.update_progress(30, 100, "Extracting Q&A answers...")
        
        # Extract Q&A answers (non-blocking)
        qa_result = extract_qa_answers.delay(
            upload_id, 
            text, 
            transcription_result["segments"],
            custom_questions
        )
        
        self.update_progress(80, 100, "Starting score calculation...")
        
        # Calculate scores (non-blocking)
        score_result = calculate_conversation_score.delay(upload_id)
        
        self.update_progress(100, 100, "Text processing tasks initiated")
        
        logger.info(f"Text-only processing tasks initiated for upload {upload_id}")
        
        return {
            "status": "started",
            "message": "Text processing tasks initiated",
            "tasks": {
                "qa_extraction": qa_result.id,
                "scoring": score_result.id,
                "upload_id": upload_id
            }
        }
        
    except Exception as e:
        logger.error(f"Text-only processing failed for upload {upload_id}: {e}")
        raise


# Maintenance tasks

@celery_app.task
def cleanup_failed_tasks():
    """Clean up failed tasks and orphaned records."""
    logger.info("Running cleanup of failed tasks...")
    
    try:
        import asyncio
        
        async def cleanup():
            async with db_manager.get_session() as session:
                # Clean up uploads that have been processing for too long
                from sqlalchemy import select, and_
                from datetime import datetime, timedelta
                
                cutoff_time = datetime.now() - timedelta(hours=2)
                
                query = select(ConversationUpload).where(
                    and_(
                        ConversationUpload.status == "processing",
                        ConversationUpload.processing_started_at < cutoff_time
                    )
                )
                
                stale_uploads = await session.scalars(query)
                count = 0
                
                for upload in stale_uploads:
                    upload.status = "failed"
                    upload.error_message = "Processing timeout - cleaned up by maintenance task"
                    upload.processing_completed_at = datetime.now()
                    count += 1
                
                if count > 0:
                    await session.commit()
                    logger.info(f"Cleaned up {count} stale uploads")
                
                return count
        
        cleaned_count = run_async(cleanup())
        return {"cleaned_uploads": cleaned_count}
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise


@celery_app.task
def update_system_metrics():
    """Update system performance metrics."""
    logger.info("Updating system metrics...")
    
    try:
        from celery_app import get_worker_stats, get_queue_length
        
        # Get worker statistics
        worker_stats = get_worker_stats()
        
        # Get queue lengths
        queue_lengths = {
            "default": get_queue_length("default"),
            "transcription": get_queue_length("transcription"),
            "qa_extraction": get_queue_length("qa_extraction"),
            "processing": get_queue_length("processing"),
            "scoring": get_queue_length("scoring")
        }
        
        metrics = {
            "timestamp": time.time(),
            "workers": worker_stats,
            "queues": queue_lengths
        }
        
        # Store metrics (implement based on your monitoring needs)
        logger.info(f"System metrics updated: {metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics update failed: {e}")
        raise