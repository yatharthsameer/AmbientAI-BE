#!/usr/bin/env python3
"""
Simple Frontend Transcription Test

This script tests the WebSocket transcription by directly streaming audio
to the backend while monitoring the frontend manually.

This is a simpler alternative that doesn't require browser automation.
You manually interact with the frontend while this script streams audio.

Requirements:
- pip install pydub websockets
"""

import asyncio
import json
import wave
import sys
from pathlib import Path
from typing import Optional

try:
    import websockets
    from pydub import AudioSegment
except ImportError as e:
    print(f"❌ Missing required packages. Please install:")
    print("pip install pydub websockets")
    sys.exit(1)

class SimpleAudioTester:
    """Simple audio file to WebSocket streamer"""
    
    def __init__(self, audio_file_path: str, websocket_url: str = "ws://localhost:8001/ws/transcribe"):
        self.audio_file_path = Path(audio_file_path)
        self.websocket_url = websocket_url
        self.websocket = None
        self.session_id = None
        self.transcript_lines = []
        
    async def run_test(self):
        """Run the test by streaming audio file to WebSocket"""
        print(f"🎵 Testing with audio file: {self.audio_file_path}")
        
        if not self.audio_file_path.exists():
            print(f"❌ Audio file not found: {self.audio_file_path}")
            return
        
        try:
            # Connect to WebSocket
            print(f"🔌 Connecting to: {self.websocket_url}")
            self.websocket = await websockets.connect(self.websocket_url)
            print("✅ WebSocket connected")
            
            # Start listening for messages
            listen_task = asyncio.create_task(self._listen_for_messages())
            
            # Wait a moment for connection to stabilize
            await asyncio.sleep(1)
            
            # Stream the audio file
            await self._stream_audio_file()
            
            # Wait a bit more for final transcription
            await asyncio.sleep(3)
            
            # Send end session
            await self._send_end_session()
            
            # Wait for final messages
            await asyncio.sleep(2)
            
            # Persist transcript to a file
            await self._save_transcript_to_file()

            print("✅ Test completed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
    
    async def _listen_for_messages(self):
        """Listen for messages from WebSocket"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'session_info':
                        self.session_id = data.get('session_id')
                        print(f"🆔 Session started: {self.session_id}")
                    
                    elif msg_type == 'transcript':
                        text = data.get('text', '')
                        is_final = data.get('is_final', False)
                        confidence = data.get('confidence', 0)
                        
                        status = "FINAL" if is_final else "partial"
                        print(f"📝 [{status}] \"{text}\" (confidence: {confidence:.2f})")
                        if is_final and text:
                            self.transcript_lines.append(text)
                    
                    elif msg_type == 'error':
                        print(f"❌ Error: {data.get('message', 'Unknown error')}")
                    
                    elif msg_type == 'status':
                        print(f"📊 Status: {data.get('message', 'Unknown status')}")
                    
                    elif msg_type == 'ping':
                        # Respond to ping with pong
                        await self.websocket.send(json.dumps({"type": "pong"}))
                        print("🏓 Ping received, sent pong")
                    
                    elif msg_type == 'pong':
                        print("🏓 Pong received")
                    
                    else:
                        print(f"📨 Received: {data}")
                        
                except json.JSONDecodeError:
                    print(f"📨 Raw message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"⚠️ Message listening error: {e}")
    
    async def _stream_audio_file(self):
        """Convert and stream audio file in large chunks for better Whisper context"""
        print(f"🎵 Loading and converting audio...")
        
        # Load and convert audio to required format
        audio = AudioSegment.from_file(str(self.audio_file_path))
        
        # Convert to 16kHz, mono, 16-bit PCM
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Get raw PCM data
        raw_audio = audio.raw_data
        
        duration = len(audio) / 1000.0  # duration in seconds
        print(f"🎵 Audio: {len(raw_audio)} bytes, {duration:.1f}s duration")
        
        # Use large chunks for better Whisper context (40 seconds each)
        chunk_duration_seconds = 40  # 40 seconds per chunk
        sample_rate = 16000
        bytes_per_sample = 2  # 16-bit = 2 bytes
        
        # Calculate chunk size in bytes (40 seconds * 16000 samples/sec * 2 bytes/sample)
        chunk_size = chunk_duration_seconds * sample_rate * bytes_per_sample
        
        total_chunks = len(raw_audio) // chunk_size + (1 if len(raw_audio) % chunk_size > 0 else 0)
        
        print(f"📡 Streaming {total_chunks} large chunks ({chunk_duration_seconds}s each)...")
        print(f"📏 Chunk size: {chunk_size:,} bytes ({chunk_size / (1024*1024):.1f} MB each)")
        
        for i in range(0, len(raw_audio), chunk_size):
            chunk = raw_audio[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            actual_duration = len(chunk) / (sample_rate * bytes_per_sample)
            
            # Send chunk
            await self.websocket.send(chunk)
            print(f"📡 Sent chunk {chunk_num}/{total_chunks} ({len(chunk):,} bytes, {actual_duration:.1f}s)")
            
            # Wait a bit between chunks (but not real-time since we want faster processing)
            await asyncio.sleep(2)  # Just 2 seconds between chunks for faster testing
        
        print("✅ Audio streaming completed")
    
    async def _send_end_session(self):
        """Send end session message"""
        end_message = {"type": "end_session"}
        await self.websocket.send(json.dumps(end_message))
        print("🛑 Sent end session message")

    async def _save_transcript_to_file(self):
        """Save collected transcript lines to a text file in the repo root"""
        if not self.transcript_lines:
            print("📄 No transcript collected to save")
            return
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = Path("transcript_output.txt")
        content = [
            "Transcript Recording",
            "===================",
            f"Date: {datetime.now().isoformat(timespec='seconds')}",
            f"Session ID: {self.session_id or 'Unknown'}",
            "",
            "Transcript:",
            "-----------",
            " ".join(self.transcript_lines).strip(),
            "",
            "---",
            "Generated by SimpleAudioTester",
        ]
        output_path.write_text("\n".join(content), encoding="utf-8")
        print(f"💾 Transcript saved to: {output_path.resolve()}")

def print_instructions():
    """Print test instructions"""
    print("""
🤖 Simple Frontend Transcription Test
=====================================

This script will stream your audio file to the WebSocket transcription service.

INSTRUCTIONS:
1. Make sure your backend server is running on port 8001
2. Make sure your frontend is running on port 5173
3. Open your browser to http://localhost:5173
4. Navigate to the Conversation page
5. Click "Start Recording" in the browser
6. Run this script - it will stream the audio file
7. Watch both the console output here AND the browser console
8. Click "Stop Recording" in the browser when done
9. Check if a transcript file was downloaded

This tests the complete flow: Frontend UI → WebSocket → Backend → Transcription
""")

async def main():
    """Main function"""
    print_instructions()
    
    # Check for audio file
    audio_files = ["audio_for_test.mp3", "harvard.wav"]
    audio_file = None
    
    for file in audio_files:
        if Path(file).exists():
            audio_file = file
            break
    
    if not audio_file:
        print(f"❌ No audio file found. Looking for: {audio_files}")
        return
    
    print(f"🎵 Using audio file: {audio_file}")
    
    # Ask user to confirm setup
    input("\n🚀 Press Enter when you have the frontend open and clicked 'Start Recording'...")
    
    # Run the test
    tester = SimpleAudioTester(audio_file)
    await tester.run_test()
    
    print("\n✅ Test completed!")
    print("📋 Check the browser console and Downloads folder for results.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
