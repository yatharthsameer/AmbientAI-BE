"""
WebSocket-based real-time transcription service for the Nurse Conversation Processing API.
Handles streaming audio transcription with real-time processing and response.
"""
import asyncio
import json
import numpy as np
import uuid
import time
import tempfile
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
import librosa
import soundfile as sf
from pydub import AudioSegment
import io

from config import get_settings, WebSocketSettings
from database import db_manager
from database_models import RealTimeSession, RealTimeTranscriptChunk, RealTimeSessionMetrics
from transcription import TranscriptionService


class AudioChunkProcessor:
    """Processes audio chunks for real-time transcription."""

    def __init__(self, session_config: WebSocketSettings):
        self.config = session_config
        self.audio_buffer: List[np.ndarray] = []
        self.buffer_duration = 0.0
        self.chunk_counter = 0
        self.session_start_time = time.time()
        self.last_chunk_time = 0.0

        # Audio processing parameters
        self.sample_rate = session_config.sample_rate
        self.channels = session_config.channels
        self.chunk_duration = session_config.chunk_duration

        # Deduplication tracking
        self.recent_transcripts = []  # Store recent transcripts for deduplication
        self.max_recent_transcripts = 5  # Keep last 5 transcripts for comparison
        self.previous_chunks: List[Dict[str, Any]] = (
            []
        )  # Store previous chunks for timestamp dedup
        self.overlap_duration = session_config.overlap_duration
        self.min_chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.sample_rate * self.overlap_duration)

        # Initialize transcription service
        self.transcription_service = TranscriptionService()

        # Performance tracking
        self.processing_times = []
        self.confidence_scores = []

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process incoming audio chunk and return transcription result.
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit)
            
        Returns:
            Dictionary with transcription result or None if not ready
        """
        start_time = time.time()

        try:
            # Convert bytes to numpy array (assume PCM 16-bit)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Normalize to float32 [-1.0, 1.0]
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Add to buffer
            self.audio_buffer.append(audio_float)
            self.buffer_duration += len(audio_float) / self.sample_rate

            # Bound the live buffer to avoid runaway memory
            max_samples = int(
                self.sample_rate * (self.chunk_duration + self.overlap_duration) * 2
            )
            total_samples = sum(len(ch) for ch in self.audio_buffer)
            if total_samples > max_samples:
                # Keep only the last max_samples worth of audio
                keep = max_samples
                merged = np.concatenate(self.audio_buffer)
                self.audio_buffer = [merged[-keep:]]
                self.buffer_duration = len(self.audio_buffer[0]) / self.sample_rate

            # Check if we have enough data for processing
            if self._should_process_buffer():
                result = await self._transcribe_buffer()
                if result:
                    # Calculate processing metrics
                    processing_time = (time.time() - start_time) * 1000  # ms
                    self.processing_times.append(processing_time)

                    result.update({
                        "chunk_index": self.chunk_counter,
                        "processing_time_ms": processing_time,
                        "session_time": time.time() - self.session_start_time,
                        "audio_duration": self.chunk_duration,
                        "audio_size_bytes": len(audio_data)
                    })

                    self.chunk_counter += 1
                    return result

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                "type": "error",
                "message": f"Audio processing failed: {str(e)}",
                "chunk_index": self.chunk_counter,
                "session_time": time.time() - self.session_start_time
            }

        return None

    def _should_process_buffer(self) -> bool:
        """Check if buffer has enough data for processing."""
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        return total_samples >= self.min_chunk_size

    async def _transcribe_buffer(self) -> Optional[Dict[str, Any]]:
        """Transcribe the current audio buffer."""
        try:
            # Concatenate buffer chunks
            if not self.audio_buffer:
                return None

            combined_audio = np.concatenate(self.audio_buffer)

            # Extract chunk for processing (with overlap)
            chunk_samples = combined_audio[:self.min_chunk_size]

            # Keep overlap for next processing
            if len(combined_audio) > self.min_chunk_size:
                overlap_samples = combined_audio[self.min_chunk_size - self.overlap_size:]
                self.audio_buffer = [overlap_samples]
                self.buffer_duration = len(overlap_samples) / self.sample_rate
            else:
                self.audio_buffer = []
                self.buffer_duration = 0.0

            # Transcribe the chunk
            transcript_result = await self._transcribe_audio_chunk(chunk_samples)

            if transcript_result and transcript_result.get("text", "").strip():
                # Timestamp-based overlap trimming
                deduped_text = self._deduplicate_by_timestamps(
                    transcript_result,
                    self.previous_chunks,
                    overlap_duration=self.overlap_duration,
                ).strip()

                if not deduped_text:
                    return None

                transcript_result["text"] = deduped_text
                transcript_text = deduped_text

                # Record confidence for metrics
                conf = transcript_result.get("confidence_score")
                if conf is not None:
                    if not hasattr(self, "confidence_scores"):
                        self.confidence_scores = []
                    self.confidence_scores.append(conf)

                # Advance timestamp correctly
                hop = self.chunk_duration - self.overlap_duration
                current_start = self.last_chunk_time
                current_end = self.last_chunk_time + self.chunk_duration
                self.last_chunk_time += max(hop, 0.0)

                result = {
                    "type": "transcript",
                    "text": transcript_text,
                    "confidence": conf or 0.0,
                    "model_used": transcript_result.get("model_used", "unknown"),
                    "start_time": current_start,
                    "end_time": current_end,
                    "is_final": True,
                    "segments": transcript_result.get(
                        "segments", []
                    ),  # Store segments for timestamp dedup
                }

                # Store this chunk for future timestamp-based deduplication
                self.previous_chunks.append(result)
                # Keep only last few chunks to avoid memory growth
                if len(self.previous_chunks) > 3:
                    self.previous_chunks = self.previous_chunks[-3:]

                return result

            return None

        except Exception as e:
            logger.error(f"Buffer transcription failed: {e}")
            return None

    async def _transcribe_audio_chunk(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Transcribe a single audio chunk using Whisper - direct array processing."""
        try:
            # Check if audio chunk has sufficient energy (not silence) - lowered threshold since VAD is enabled
            audio_energy = np.sqrt(np.mean(audio_data ** 2))
            if audio_energy < 0.005:  # Lowered from 0.01 to avoid dropping quiet speech
                logger.debug("Skipping silent audio chunk")
                return None

            # Direct array transcription - no temp files
            result = self.transcription_service.transcribe_array(
                audio_data,
                sr=self.sample_rate,
                model_name=self.config.whisper_model_realtime,
                language="en",  # Force English only
                temperature=0.0,  # Deterministic results
                task="transcribe",
                beam_size=2,  # Reduced from 3 to 2 for lower CPU latency
                patience=1.0,
                word_timestamps=True,  # Enable for clean stitching
                vad_filter=True,  # Enable model's own VAD
                vad_parameters={
                    "min_silence_duration_ms": 600,
                    "speech_pad_ms": 150,
                },
                no_speech_threshold=0.6,  # Reduce garbage
                compression_ratio_threshold=2.3,
                logprob_threshold=-0.5,
                condition_on_previous_text=False,  # Disable for live to reduce drift
            )
            return result

        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return None

    def _deduplicate_by_timestamps(
        self,
        current_result: Dict[str, Any],
        previous_chunks: List[Dict[str, Any]],
        overlap_duration: float = 2.0,
    ) -> str:
        """
        Deduplicate transcription results using word timestamps.
        Removes words from current result that overlap with previous chunks.
        """
        if not previous_chunks or not current_result.get("segments"):
            return current_result.get("text", "")

        current_segments = current_result.get("segments", [])
        if not current_segments:
            return current_result.get("text", "")

        # Get the last chunk's end time to determine overlap region
        last_chunk = previous_chunks[-1] if previous_chunks else None
        if not last_chunk or not last_chunk.get("segments"):
            return current_result.get("text", "")

        # Find the overlap threshold - words starting before this time should be removed
        last_chunk_segments = last_chunk.get("segments", [])
        last_chunk_end = 0.0
        for seg in last_chunk_segments:
            seg_end = float(seg.get("end", 0.0))
            if seg_end > last_chunk_end:
                last_chunk_end = seg_end

        overlap_threshold = max(0.0, last_chunk_end - overlap_duration)

        # Filter words from current segments that don't overlap
        filtered_text_parts = []
        for segment in current_segments:
            words = segment.get("words", [])
            if not words:
                # If no word timestamps, use segment timestamp
                seg_start = float(segment.get("start", 0.0))
                if seg_start >= overlap_threshold:
                    filtered_text_parts.append(segment.get("text", "").strip())
            else:
                # Filter words by timestamp
                kept_words = []
                for word in words:
                    word_start = float(word.get("start", 0.0))
                    if word_start >= overlap_threshold:
                        kept_words.append(word.get("word", "").strip())

                if kept_words:
                    filtered_text_parts.append(" ".join(kept_words))

        return " ".join(filtered_text_parts).strip()

    def _is_duplicate_transcript(self, new_text: str) -> bool:
        """Check if the new transcript is a duplicate of recent ones."""
        if not new_text or not new_text.strip():
            return True

        new_text_clean = new_text.strip().lower()

        # Check against recent transcripts
        for recent_text in self.recent_transcripts:
            recent_clean = recent_text.strip().lower()

            # Check for exact match
            if new_text_clean == recent_clean:
                return True

            # Check for substantial overlap (>80% similarity)
            if len(new_text_clean) > 5 and len(recent_clean) > 5:
                # Simple similarity check - count common words
                new_words = set(new_text_clean.split())
                recent_words = set(recent_clean.split())

                if new_words and recent_words:
                    overlap = len(new_words.intersection(recent_words))
                    similarity = overlap / max(len(new_words), len(recent_words))

                    if similarity > 0.8:  # 80% word overlap
                        return True

        return False

    def _deduplicate_by_timestamps(
        self,
        current_result: Dict[str, Any],
        previous_results: List[Dict[str, Any]],
        overlap_duration: float = 2.0,
    ) -> str:
        """
        Deduplicate overlapping transcripts using word-level timestamps.

        Args:
            current_result: Current transcription result with word timestamps
            previous_results: List of previous results for comparison
            overlap_duration: Expected overlap duration in seconds

        Returns:
            Deduplicated text from current result
        """
        if not current_result.get("segments") or not previous_results:
            return current_result.get("text", "")

        # Get word-level timestamps from current result
        current_words = []
        for segment in current_result.get("segments", []):
            for word in segment.get("words", []):
                current_words.append(
                    {
                        "word": word.get("word", "").strip(),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                    }
                )

        if not current_words:
            return current_result.get("text", "")

        # Find words that fall outside the overlap region
        overlap_threshold = overlap_duration
        deduplicated_words = []

        for word in current_words:
            # Keep words that start after the overlap threshold
            if word["start"] >= overlap_threshold:
                deduplicated_words.append(word["word"])

        # If no words survive deduplication, return the full text
        # (this can happen with very short segments)
        if not deduplicated_words:
            return current_result.get("text", "")

        return " ".join(deduplicated_words).strip()

    def _add_to_recent_transcripts(self, text: str):
        """Add transcript to recent list for deduplication."""
        if text and text.strip():
            self.recent_transcripts.append(text.strip())
            # Keep only the most recent transcripts
            if len(self.recent_transcripts) > self.max_recent_transcripts:
                self.recent_transcripts.pop(0)

    def get_session_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the current session."""
        if not self.processing_times:
            return {}

        return {
            "chunks_processed": self.chunk_counter,
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times),
            "max_processing_time_ms": max(self.processing_times),
            "min_processing_time_ms": min(self.processing_times),
            "total_session_duration": time.time() - self.session_start_time,
            "avg_confidence_score": sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            "real_time_factor": (sum(self.processing_times) / 1000) / max(1, time.time() - self.session_start_time)
        }


class WebSocketTranscriptionManager:
    """Manages WebSocket connections and real-time transcription sessions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.ws_settings = self.settings.websocket_settings
        
        # Active connections and processors
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_processors: Dict[str, AudioChunkProcessor] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
        
        # Connection management
        self.connection_count = 0
        self.max_connections = self.ws_settings.max_connections
        
        logger.info(f"WebSocket transcription manager initialized with max connections: {self.max_connections}")
    
    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """
        Accept and initialize a WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            session_id: Unique session identifier
            
        Returns:
            True if connection successful, False if rejected
        """
        try:
            # Check connection limits
            if self.connection_count >= self.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                return False
            
            # Accept the connection
            await websocket.accept()
            
            # Initialize session
            self.active_connections[session_id] = websocket
            self.session_processors[session_id] = AudioChunkProcessor(self.ws_settings)
            self.session_data[session_id] = {
                "connected_at": datetime.now(),
                "last_activity": datetime.now(),
                "total_chunks": 0,
                "full_transcript": "",
                "chunks": []
            }
            
            self.connection_count += 1
            
            # Create database session record
            await self._create_session_record(session_id)
            
            logger.info(f"WebSocket connected: {session_id} (total connections: {self.connection_count})")
            
            # Send connection confirmation
            await self.send_message(session_id, {
                "type": "connected",
                "session_id": session_id,
                "message": "Ready for audio streaming",
                "config": {
                    "sample_rate": self.ws_settings.sample_rate,
                    "channels": self.ws_settings.channels,
                    "chunk_duration": self.ws_settings.chunk_duration,
                    "whisper_model": self.ws_settings.whisper_model_realtime
                }
            })
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed for {session_id}: {e}")
            await self.disconnect(session_id)
            return False
    
    async def disconnect(self, session_id: str, reason: str = "Normal closure"):
        """
        Clean up WebSocket connection and session data.
        
        Args:
            session_id: Session identifier
            reason: Reason for disconnection
        """
        try:
            # Update session data
            if session_id in self.session_data:
                session_info = self.session_data[session_id]
                
                # Save final session data to database
                await self._finalize_session_record(session_id, session_info)
                
                # Clean up session data
                del self.session_data[session_id]
            
            # Clean up processor
            if session_id in self.session_processors:
                del self.session_processors[session_id]
            
            # Clean up WebSocket connection
            if session_id in self.active_connections:
                try:
                    await self.active_connections[session_id].close()
                except:
                    pass  # Connection might already be closed
                del self.active_connections[session_id]
                self.connection_count -= 1
            
            logger.info(f"WebSocket disconnected: {session_id} - {reason} (remaining: {self.connection_count})")
            
        except Exception as e:
            logger.error(f"Error during disconnection cleanup for {session_id}: {e}")
    
    async def process_audio_data(self, session_id: str, audio_data: bytes):
        """
        Process incoming audio data for a session.
        
        Args:
            session_id: Session identifier
            audio_data: Raw audio bytes
        """
        if session_id not in self.session_processors:
            logger.warning(f"Audio data received for unknown session: {session_id}")
            return
        
        try:
            # Update activity timestamp
            self.session_data[session_id]["last_activity"] = datetime.now()
            self.session_data[session_id]["total_chunks"] += 1
            
            # Process audio chunk
            processor = self.session_processors[session_id]
            result = await processor.process_audio_chunk(audio_data)
            
            if result:
                # Store transcript chunk
                if result.get("type") == "transcript":
                    self.session_data[session_id]["chunks"].append(result)
                    self.session_data[session_id]["full_transcript"] += " " + result.get("text", "")
                    
                    # Save to database
                    await self._save_transcript_chunk(session_id, result)
                
                # Send result to client
                await self.send_message(session_id, result)
                
        except Exception as e:
            logger.error(f"Error processing audio data for {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Audio processing failed: {str(e)}"
            })
    
    async def handle_text_message(self, session_id: str, message: Dict[str, Any]):
        """
        Handle text messages from the client.
        
        Args:
            session_id: Session identifier
            message: Parsed JSON message
        """
        try:
            message_type = message.get("type")
            
            if message_type == "ping":
                await self.send_message(session_id, {"type": "pong"})
                
            elif message_type == "session_info":
                # Send session information
                session_info = self.session_data.get(session_id, {})
                processor = self.session_processors.get(session_id)
                
                response = {
                    "type": "session_info",
                    "session_id": session_id,
                    "connected_at": session_info.get("connected_at", "").isoformat() if session_info.get("connected_at") else "",
                    "total_chunks": session_info.get("total_chunks", 0),
                    "full_transcript": session_info.get("full_transcript", ""),
                    "metrics": processor.get_session_metrics() if processor else {}
                }
                await self.send_message(session_id, response)
                
            elif message_type == "end_session":
                # Finalize session and trigger Q&A processing
                await self._finalize_session(session_id)
                await self.send_message(session_id, {
                    "type": "session_ended",
                    "message": "Session finalized, processing Q&A extraction"
                })
                
            else:
                logger.warning(f"Unknown message type from {session_id}: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling text message from {session_id}: {e}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific WebSocket client.
        
        Args:
            session_id: Session identifier
            message: Message dictionary to send
        """
        if session_id not in self.active_connections:
            return
        
        try:
            websocket = self.active_connections[session_id]
            await websocket.send_text(json.dumps(message, default=str))
            
        except Exception as e:
            logger.error(f"Error sending message to {session_id}: {e}")
            await self.disconnect(session_id, "Send error")
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected_sessions = []
        
        for session_id in list(self.active_connections.keys()):
            try:
                await self.send_message(session_id, message)
            except:
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            await self.disconnect(session_id, "Broadcast error")
    
    async def _create_session_record(self, session_id: str):
        """Create database record for new session."""
        try:
            async with db_manager.get_session() as db_session:
                session_record = RealTimeSession(
                    session_id=session_id,
                    status="active",
                    sample_rate=self.ws_settings.sample_rate,
                    channels=self.ws_settings.channels,
                    chunk_duration=self.ws_settings.chunk_duration,
                    whisper_model=self.ws_settings.whisper_model_realtime,
                    connected_at=datetime.now(),
                    last_activity_at=datetime.now()
                )
                
                db_session.add(session_record)
                await db_session.commit()
                
        except Exception as e:
            logger.error(f"Failed to create session record for {session_id}: {e}")
    
    async def _save_transcript_chunk(self, session_id: str, chunk_data: Dict[str, Any]):
        """Save transcript chunk to database."""
        try:
            async with db_manager.get_session() as db_session:
                # Get session record
                stmt = select(RealTimeSession).where(RealTimeSession.session_id == session_id)
                result = await db_session.execute(stmt)
                session_record = result.scalar_one_or_none()
                
                if session_record:
                    chunk_record = RealTimeTranscriptChunk(
                        session_id=session_record.id,
                        chunk_index=chunk_data.get("chunk_index", 0),
                        start_time=chunk_data.get("start_time", 0.0),
                        end_time=chunk_data.get("end_time", 0.0),
                        text=chunk_data.get("text", ""),
                        confidence_score=chunk_data.get("confidence", 0.0),
                        model_used=chunk_data.get("model_used", "unknown"),
                        processing_time_ms=int(chunk_data.get("processing_time_ms", 0)),
                        audio_duration=chunk_data.get("audio_duration", 0.0),
                        audio_size_bytes=chunk_data.get("audio_size_bytes", 0),
                        is_final=True
                    )
                    
                    db_session.add(chunk_record)
                    await db_session.commit()
                    
        except Exception as e:
            logger.error(f"Failed to save transcript chunk for {session_id}: {e}")
    
    async def _finalize_session_record(self, session_id: str, session_info: Dict[str, Any]):
        """Update session record with final data."""
        try:
            async with db_manager.get_session() as db_session:
                stmt = select(RealTimeSession).where(RealTimeSession.session_id == session_id)
                result = await db_session.execute(stmt)
                session_record = result.scalar_one_or_none()
                
                if session_record:
                    # Update session record
                    session_record.status = "completed"
                    session_record.disconnected_at = datetime.now()
                    session_record.full_transcript = session_info.get("full_transcript", "")
                    session_record.chunks_processed = session_info.get("total_chunks", 0)
                    
                    # Calculate session metrics
                    processor = self.session_processors.get(session_id)
                    if processor:
                        metrics = processor.get_session_metrics()
                        
                        # Create metrics record
                        session_metrics = RealTimeSessionMetrics(
                            session_id=session_record.id,
                            avg_processing_time_ms=metrics.get("avg_processing_time_ms", 0.0),
                            max_processing_time_ms=int(metrics.get("max_processing_time_ms", 0)),
                            min_processing_time_ms=int(metrics.get("min_processing_time_ms", 0)),
                            avg_confidence_score=metrics.get("avg_confidence_score", 0.0),
                            total_chunks=metrics.get("chunks_processed", 0),
                            successful_chunks=metrics.get("chunks_processed", 0),
                            failed_chunks=0,
                            avg_chunk_size_bytes=1024,  # Estimate
                            total_data_received_bytes=session_info.get("total_chunks", 0) * 1024,
                            connection_drops=0,
                            real_time_factor=metrics.get("real_time_factor", 0.0),
                            latency_ms=int(metrics.get("avg_processing_time_ms", 0)),
                            transcript_completeness=1.0,  # Assume complete for now
                            estimated_accuracy=metrics.get("avg_confidence_score", 0.0)
                        )
                        
                        db_session.add(session_metrics)
                    
                    await db_session.commit()
                    
        except Exception as e:
            logger.error(f"Failed to finalize session record for {session_id}: {e}")
    
    async def _finalize_session(self, session_id: str):
        """Finalize session and trigger Q&A processing."""
        try:
            session_info = self.session_data.get(session_id, {})
            full_transcript = session_info.get("full_transcript", "").strip()
            
            if full_transcript:
                # Import here to avoid circular imports
                from tasks import process_text_only
                
                # Create a conversation upload record for the transcript
                from database_models import ConversationUpload
                
                async with db_manager.get_session() as db_session:
                    upload = ConversationUpload(
                        original_filename=f"realtime_session_{session_id}.txt",
                        file_path="",
                        file_size=len(full_transcript.encode('utf-8')),
                        content_type="text/plain",
                        duration_seconds=session_info.get("total_chunks", 0) * self.ws_settings.chunk_duration,
                        status="pending"
                    )
                    
                    db_session.add(upload)
                    await db_session.commit()
                    await db_session.refresh(upload)
                    
                    # Start Q&A processing
                    task = process_text_only.delay(str(upload.id), full_transcript, [])
                    upload.qa_extraction_task_id = task.id
                    await db_session.commit()
                    
                    logger.info(f"Started Q&A processing for session {session_id} with upload ID {upload.id}")
                    
        except Exception as e:
            logger.error(f"Failed to finalize session {session_id}: {e}")
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        sessions = []
        
        for session_id, session_info in self.session_data.items():
            processor = self.session_processors.get(session_id)
            metrics = processor.get_session_metrics() if processor else {}
            
            sessions.append({
                "session_id": session_id,
                "connected_at": session_info.get("connected_at", ""),
                "last_activity": session_info.get("last_activity", ""),
                "total_chunks": session_info.get("total_chunks", 0),
                "transcript_length": len(session_info.get("full_transcript", "")),
                "metrics": metrics
            })
        
        return sessions


# Global WebSocket manager instance
ws_transcription_manager = WebSocketTranscriptionManager()
