#!/usr/bin/env python3
"""
Minimal WebSocket Transcription Server

A simplified server for testing WAV file transcription without database dependencies.
This server provides only the WebSocket transcription functionality.
"""

import asyncio
import json
import uuid
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from starlette.websockets import WebSocketState

# Import only the transcription service and config
from transcription import TranscriptionService
from config import get_settings


class MinimalTranscriptionManager:
    """Minimal WebSocket transcription manager without database."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.transcription_service = TranscriptionService()
        self.session_transcripts: Dict[str, list[str]] = {}
        self.saved_sessions: set[str] = set()

        # Load configuration
        settings = get_settings()
        ws_config = settings.websocket_settings

        self.sample_rate = ws_config.sample_rate
        self.chunk_duration = ws_config.chunk_duration
        self.heartbeat_interval = ws_config.heartbeat_interval
        self.chunk_timeout = ws_config.chunk_timeout
        self.whisper_model = ws_config.whisper_model_realtime

        print(f"📊 Loaded WebSocket Configuration:")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Chunk Duration: {self.chunk_duration} seconds")
        print(f"   Chunk Timeout: {self.chunk_timeout} seconds")
        print(f"   Heartbeat Interval: {self.heartbeat_interval} seconds")

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_transcripts[session_id] = []

        # Send connection confirmation
        await self.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "sample_rate": self.sample_rate,
                "chunk_duration": self.chunk_duration
            }
        })

        print(f"✅ WebSocket connected: {session_id}")
        return True

    async def disconnect(self, session_id: str):
        """Disconnect a WebSocket client."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        # Persist transcript on disconnect if not already saved
        await self._persist_transcript(session_id)
        print(f"🔌 WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a WebSocket client."""
        if session_id not in self.active_connections:
            return
        ws = self.active_connections[session_id]
        if ws.application_state != WebSocketState.CONNECTED:
            return
        try:
            await ws.send_text(json.dumps(message))
        except Exception as e:
            print(f"❌ Failed to send message to {session_id}: {e}")

    async def process_audio_data(self, session_id: str, audio_data: bytes):
        """Process incoming audio data."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0

            # Check if audio has sufficient energy
            audio_energy = np.sqrt(np.mean(audio_array ** 2))
            if audio_energy < 0.01:
                print(f"🔇 Skipping silent chunk for {session_id}")
                return

            # Create temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

                # Write audio to temporary file
                sf.write(temp_path, audio_array, self.sample_rate)

                try:
                    # Transcribe using Whisper in a background thread to avoid blocking the event loop
                    result = await asyncio.to_thread(
                        self.transcription_service.transcribe_audio,
                        temp_path,
                        model_name=self.whisper_model,
                        language="en",
                        temperature=0.0,
                        task="transcribe",
                    )

                    if result and result.get("segments"):
                        # Extract transcript text
                        segments = result.get("segments", [])
                        if segments:
                            text = " ".join(
                                [seg.get("text", "").strip() for seg in segments]
                            )
                            if text.strip():
                                # Accumulate transcript server-side only
                                if session_id in self.session_transcripts:
                                    self.session_transcripts[session_id].append(
                                        text.strip()
                                    )
                                print(f'📝 Transcribed (buffered): "{text.strip()}"')

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except FileNotFoundError:
                        pass

        except Exception as e:
            print(f"❌ Audio processing error for {session_id}: {e}")
            await self.send_message(session_id, {
                "type": "error",
                "message": f"Audio processing failed: {str(e)}"
            })

    async def handle_text_message(self, session_id: str, message: Dict[str, Any]):
        """Handle text control messages."""
        msg_type = message.get("type")

        if msg_type == "ping":
            await self.send_message(session_id, {"type": "pong"})
        elif msg_type == "pong":
            # No-op
            return
        elif msg_type == "end_session":
            # Acknowledge and close connection gracefully
            await self.send_message(
                session_id,
                {"type": "session_ended", "timestamp": datetime.now().isoformat()},
            )
            await self._persist_transcript(session_id)
            if session_id in self.active_connections:
                try:
                    await self.active_connections[session_id].close()
                except Exception:
                    pass
        elif msg_type == "config":
            # Handle configuration updates
            await self.send_message(session_id, {
                "type": "config_updated",
                "config": {
                    "sample_rate": self.sample_rate,
                    "chunk_duration": self.chunk_duration
                }
            })
        else:
            print(f"⚠️ Unknown message type: {msg_type}")

    async def _persist_transcript(self, session_id: str):
        """Persist the full transcript for a session to logs/ as a txt file."""
        if session_id in self.saved_sessions:
            return
        texts = self.session_transcripts.get(session_id, [])
        if not texts:
            return
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = os.path.join("logs", f"transcript_{timestamp}_{session_id}.txt")
        content = [
            "Transcript Recording",
            "===================",
            f"Date: {datetime.now().isoformat(timespec='seconds')}",
            f"Session ID: {session_id}",
            "",
            "Transcript:",
            "-----------",
            " ".join(texts).strip(),
            "",
            "---",
            "Generated by MinimalTranscriptionServer",
        ]
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            self.saved_sessions.add(session_id)
            print(f"💾 Transcript saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save transcript for {session_id}: {e}")


# Create FastAPI app
app = FastAPI(title="Minimal WebSocket Transcription Server")

# Create transcription manager
transcription_manager = MinimalTranscriptionManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "minimal_transcription_server"
    })


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for transcription."""
    session_id = str(uuid.uuid4())

    try:
        # Connect
        await transcription_manager.connect(websocket, session_id)

        while True:
            try:
                # Check WebSocket state
                if websocket.client_state.name == "DISCONNECTED":
                    print(f"🔌 Client disconnected (state check): {session_id}")
                    break

                # Receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(), timeout=transcription_manager.chunk_timeout
                    )
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await transcription_manager.send_message(
                        session_id, {"type": "ping"}
                    )
                    continue

                # Check for disconnect message
                if message.get("type") == "websocket.disconnect":
                    print(f"🔌 Disconnect message received: {session_id}")
                    break

                if "bytes" in message:
                    # Handle audio data
                    audio_data = message["bytes"]
                    await transcription_manager.process_audio_data(session_id, audio_data)

                elif "text" in message:
                    # Handle text messages
                    try:
                        text_message = json.loads(message["text"])
                        await transcription_manager.handle_text_message(session_id, text_message)
                    except json.JSONDecodeError:
                        print(f"⚠️ Invalid JSON from {session_id}")

            except WebSocketDisconnect:
                print(f"🔌 WebSocket client disconnected: {session_id}")
                break

            except Exception as e:
                error_msg = str(e)

                # Check for disconnect-related errors
                if ("disconnect" in error_msg.lower() or 
                    "receive" in error_msg.lower() or
                    ("connection" in error_msg.lower() and "closed" in error_msg.lower())):
                    print(f"🔌 Connection closed: {session_id}")
                    break

                print(f"❌ Error processing message: {error_msg}")
                try:
                    await transcription_manager.send_message(session_id, {
                        "type": "error",
                        "message": f"Processing error: {error_msg}"
                    })
                except:
                    print(f"⚠️ Could not send error message to {session_id}")
                    break

    except Exception as e:
        print(f"❌ WebSocket error for {session_id}: {e}")

    finally:
        await transcription_manager.disconnect(session_id)


if __name__ == "__main__":
    print("🚀 Starting Minimal WebSocket Transcription Server")
    print("=" * 60)
    print("📡 WebSocket endpoint: ws://localhost:8001/ws/transcribe")
    print("🏥 Health check: http://localhost:8001/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
