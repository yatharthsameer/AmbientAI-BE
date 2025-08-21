#!/usr/bin/env python3
"""
Simple Live Transcription Server

Just does:
1. Live transcription chunks → append to txt file
2. Final complete transcription at the end

Uses medium.en model for good quality/speed balance.
"""

import asyncio
import json
import uuid
import os
from datetime import datetime
from typing import Dict, Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from starlette.websockets import WebSocketState


class SimpleTranscriptionManager:
    """Simple live transcription manager."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_audio_buffers: Dict[str, bytearray] = {}
        self.session_full_audio: Dict[str, bytearray] = {}  # Keep full audio for final transcription
        
        # Simple settings
        self.sample_rate = 16000
        self.chunk_seconds = 60.0  # Process every 60 seconds of audio for better context
        self.model_name = "medium.en"
        
        # Lazy load transcription service
        self._transcription_service = None
        
        print("🚀 Simple Live Transcription Server")
        print(f"   Model: {self.model_name}")
        print(f"   Live chunk size: {self.chunk_seconds}s (better context for accuracy)")

    def _get_transcription_service(self):
        """Lazy load transcription service to avoid startup overhead."""
        if self._transcription_service is None:
            from transcription import TranscriptionService
            self._transcription_service = TranscriptionService()
        return self._transcription_service

    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        """Connect a WebSocket client."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_audio_buffers[session_id] = bytearray()
        self.session_full_audio[session_id] = bytearray()
        
        # Create session log file
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/live_{session_id}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Live Transcription Session: {session_id}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")

        await self.send_message(session_id, {
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })

        print(f"✅ Connected: {session_id}")
        return True

    async def disconnect(self, session_id: str):
        """Disconnect and create final transcription."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        # Create final transcription from full audio
        await self._create_final_transcription(session_id)
        
        # Cleanup
        self.session_audio_buffers.pop(session_id, None)
        self.session_full_audio.pop(session_id, None)
        
        print(f"🔌 Disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict):
        """Send message to client."""
        if session_id not in self.active_connections:
            return
        ws = self.active_connections[session_id]
        if ws.application_state != WebSocketState.CONNECTED:
            return
        try:
            await ws.send_text(json.dumps(message))
        except Exception as e:
            print(f"❌ Failed to send message: {e}")

    async def process_audio_data(self, session_id: str, audio_data: bytes):
        """Process incoming audio data."""
        try:
            # Add to buffers
            self.session_audio_buffers[session_id].extend(audio_data)
            self.session_full_audio[session_id].extend(audio_data)
            
            # Check if we have enough audio to process
            buffer = self.session_audio_buffers[session_id]
            samples = len(buffer) // 2  # int16
            duration = samples / self.sample_rate
            
            if duration >= self.chunk_seconds:
                # Process this chunk (60s for better context and accuracy)
                chunk_bytes = bytes(buffer)
                self.session_audio_buffers[session_id] = bytearray()  # Clear buffer
                
                # Transcribe in background
                asyncio.create_task(self._transcribe_chunk(session_id, chunk_bytes))
                
        except Exception as e:
            print(f"❌ Audio processing error: {e}")

    async def _transcribe_chunk(self, session_id: str, audio_bytes: bytes):
        """Transcribe a chunk of audio and append to live file."""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
            if len(audio_array) == 0:
                return
                
            # Check audio energy
            energy = np.sqrt(np.mean(audio_array**2))
            if energy < 0.001:  # Skip silent chunks
                return
            
            # Transcribe using the service
            service = self._get_transcription_service()
            result = service.transcribe_array(
                audio_array,
                sr=self.sample_rate,
                model_name=self.model_name,
                language="en",
                task="transcribe",
                temperature=0.0,
                beam_size=1,  # Fast greedy decoding
                word_timestamps=False,
                initial_prompt="Medical conversation between nurse and patient discussing health assessment, medications, and care planning."
            )
            
            text = result.get("text", "").strip()
            if text:
                # Append to live file (no timestamps for simplicity)
                log_path = f"logs/live_{session_id}.txt"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{text}\n\n")
                
                print(f"📝 Live: {text[:80]}...")
                
                # Send to client
                await self.send_message(session_id, {
                    "type": "live_transcript",
                    "text": text
                })
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")

    async def _create_final_transcription(self, session_id: str):
        """Create final high-quality transcription from full audio."""
        try:
            full_audio = self.session_full_audio.get(session_id)
            if not full_audio or len(full_audio) < self.sample_rate * 2:  # Less than 1 second
                print("⚠️ Not enough audio for final transcription")
                return
            
            print(f"🎯 Creating final transcription for {session_id}...")
            
            # Convert to numpy array
            audio_array = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32767.0
            
            # High-quality transcription
            service = self._get_transcription_service()
            result = service.transcribe_array(
                audio_array,
                sr=self.sample_rate,
                model_name=self.model_name,
                language="en",
                task="transcribe",
                temperature=0.0,
                beam_size=5,  # Higher quality
                word_timestamps=True,
                initial_prompt="Medical conversation between nurse and patient discussing health assessment, medications, and care planning."
            )
            
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            
            if text:
                # Save final transcription
                final_path = f"logs/final_{session_id}.txt"
                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(f"Final Transcription Session: {session_id}\n")
                    f.write(f"Completed: {datetime.now().isoformat()}\n")
                    f.write(f"Duration: {len(audio_array) / self.sample_rate:.1f} seconds\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(text)
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("Segments with timestamps:\n\n")
                    
                    for seg in segments:
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        seg_text = seg.get("text", "").strip()
                        f.write(f"[{start:.1f}s - {end:.1f}s] {seg_text}\n")
                
                print(f"💾 Final transcription saved: {final_path}")
                
        except Exception as e:
            print(f"❌ Final transcription error: {e}")

    async def handle_text_message(self, session_id: str, message: Dict):
        """Handle text control messages."""
        msg_type = message.get("type")
        
        if msg_type == "ping":
            await self.send_message(session_id, {"type": "pong"})
        elif msg_type == "end_session":
            await self.send_message(session_id, {
                "type": "session_ended", 
                "timestamp": datetime.now().isoformat()
            })


# Create FastAPI app
app = FastAPI(title="Simple Live Transcription Server")
transcription_manager = SimpleTranscriptionManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "simple_transcription_server"
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
                # Receive message
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
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
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        await transcription_manager.disconnect(session_id)


if __name__ == "__main__":
    print("🚀 Starting Simple Live Transcription Server")
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
