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

# Import only the transcription service
from transcription import TranscriptionService


class MinimalTranscriptionManager:
    """Minimal WebSocket transcription manager without database."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.transcription_service = TranscriptionService()
        self.sample_rate = 16000
        self.chunk_duration = 3.0
        
    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
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
        print(f"🔌 WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a WebSocket client."""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
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
                    # Transcribe using Whisper
                    result = self.transcription_service.transcribe_audio(
                        temp_path,
                        model_name="base",
                        language="en",
                        temperature=0.0,
                        task="transcribe"
                    )
                    

                    if result and result.get("segments"):
                        # Extract transcript text
                        segments = result.get("segments", [])
                        if segments:
                            text = " ".join([seg.get("text", "").strip() for seg in segments])
                            # Use the top-level confidence_score from the transcription service
                            confidence = result.get("confidence_score", 0.0)
                            
                            if text.strip():
                                # Send transcript
                                await self.send_message(session_id, {
                                    "type": "transcript",
                                    "text": text.strip(),
                                    "confidence": float(confidence),
                                    "timestamp": datetime.now().isoformat(),
                                    "is_final": True,
                                    "language": result.get("language", "en")
                                })
                                
                                print(f"📝 Transcribed: \"{text.strip()}\" (conf: {confidence:.3f})")
                    
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
                
                # Receive message
                message = await websocket.receive()
                
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
