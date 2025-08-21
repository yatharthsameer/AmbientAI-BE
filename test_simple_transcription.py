#!/usr/bin/env python3
"""
Simple test client for the live transcription server.
"""

import asyncio
import json
import os
import sys
import websockets
from pydub import AudioSegment


AUDIO_PATH = "audio_for_test.mp3"
WS_URL = "ws://127.0.0.1:8001/ws/transcribe"
SAMPLE_RATE = 16000
FRAME_SEC = 1.0  # Send 1 second chunks (server processes every 60s for better accuracy)


def load_audio_as_pcm16(audio_path: str) -> bytes:
    """Load audio file and convert to 16kHz mono PCM16."""
    seg = AudioSegment.from_file(audio_path)
    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return seg.raw_data


def chunk_audio(pcm_bytes: bytes, frame_sec: float = FRAME_SEC):
    """Split audio into chunks."""
    frame_size = int(SAMPLE_RATE * 2 * frame_sec)  # int16 mono
    for i in range(0, len(pcm_bytes), frame_size):
        yield pcm_bytes[i:i + frame_size]


async def run_test():
    """Run the transcription test."""
    if not os.path.exists(AUDIO_PATH):
        print(f"❌ Audio file not found: {AUDIO_PATH}")
        return

    # Load audio
    print(f"🎵 Loading audio: {AUDIO_PATH}")
    pcm = load_audio_as_pcm16(AUDIO_PATH)
    duration = len(pcm) / (SAMPLE_RATE * 2)
    print(f"   Duration: {duration:.1f}s")

    # Connect to server
    print(f"📡 Connecting to {WS_URL}")
    async with websockets.connect(WS_URL, ping_interval=None, ping_timeout=None) as ws:
        
        # Listen for messages
        async def listen():
            try:
                async for message in ws:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "connected":
                        print(f"✅ Connected: {data.get('session_id')}")
                    elif msg_type == "live_transcript":
                        text = data.get("text")
                        print(f"📝 {text}")
                    elif msg_type == "session_ended":
                        print("🛑 Session ended")
                        break
            except Exception as e:
                print(f"❌ Listen error: {e}")

        # Start listener
        listen_task = asyncio.create_task(listen())

        # Wait for connection
        await asyncio.sleep(0.5)

        # Stream audio
        print("📤 Streaming audio...")
        for i, chunk in enumerate(chunk_audio(pcm)):
            if not chunk:
                continue
            await ws.send(chunk)
            # Realtime simulation
            await asyncio.sleep(FRAME_SEC)
            
            if i % 60 == 0 and i > 0:  # Progress every 60 seconds (when transcription happens)
                print(f"   Sent {i * FRAME_SEC:.1f}s... (transcription should appear soon)")

        print("📤 Finished streaming. Ending session...")
        await ws.send(json.dumps({"type": "end_session"}))

        # Wait a bit for final processing
        await asyncio.sleep(2.0)
        
        # Close
        await ws.close()
        await listen_task

    print("✅ Test completed!")
    print("📁 Check logs/ directory for:")
    print("   - live_*.txt (live transcription)")
    print("   - final_*.txt (final high-quality transcription)")


if __name__ == "__main__":
    asyncio.run(run_test())
