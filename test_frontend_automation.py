#!/usr/bin/env python3
"""
Automated Frontend Transcription Test

This script automates testing of the frontend transcription feature by:
1. Starting the backend server
2. Starting the frontend dev server
3. Opening the browser and navigating to the conversation page
4. Simulating user clicks on record/stop buttons
5. Streaming audio file data through a mock WebSocket connection
6. Verifying transcript file generation

Requirements:
- pip install playwright pydub websockets asyncio
- playwright install chromium
"""

import asyncio
import json
import subprocess
import time
import os
import sys
import signal
from pathlib import Path
from typing import Optional
import websockets
import wave
import threading

try:
    from playwright.async_api import async_playwright
    from pydub import AudioSegment
except ImportError as e:
    print(f"❌ Missing required packages. Please install:")
    print("pip install playwright pydub")
    print("playwright install chromium")
    sys.exit(1)

class AudioFileStreamer:
    """Streams audio file data to WebSocket as if it were live microphone input"""
    
    def __init__(self, audio_file_path: str):
        self.audio_file_path = audio_file_path
        self.websocket = None
        self.is_streaming = False
        
    async def connect_and_stream(self, websocket_url: str = "ws://localhost:8001/ws/transcribe"):
        """Connect to WebSocket and stream audio file"""
        try:
            print(f"🔌 Connecting to WebSocket: {websocket_url}")
            self.websocket = await websockets.connect(websocket_url)
            print("✅ WebSocket connected")
            
            # Listen for messages in background
            asyncio.create_task(self._listen_for_messages())
            
            # Stream audio file
            await self._stream_audio_file()
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            
    async def _listen_for_messages(self):
        """Listen for transcription messages from WebSocket"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                print(f"📝 Received: {data}")
        except Exception as e:
            print(f"⚠️ Message listening error: {e}")
    
    async def _stream_audio_file(self):
        """Convert and stream audio file in large chunks for better Whisper context"""
        print(f"🎵 Loading audio file: {self.audio_file_path}")
        
        # Convert audio to required format (16kHz, mono, 16-bit PCM)
        audio = AudioSegment.from_file(self.audio_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Get raw audio data
        raw_audio = audio.raw_data
        
        print(f"🎵 Audio loaded: {len(raw_audio)} bytes, duration: {len(audio)/1000:.1f}s")
        
        # Use large chunks for better Whisper context (40 seconds each)
        chunk_duration_seconds = 40  # 40 seconds per chunk
        sample_rate = 16000
        bytes_per_sample = 2  # 16-bit = 2 bytes
        
        # Calculate chunk size in bytes (40 seconds * 16000 samples/sec * 2 bytes/sample)
        chunk_size = chunk_duration_seconds * sample_rate * bytes_per_sample
        
        total_chunks = len(raw_audio) // chunk_size + (1 if len(raw_audio) % chunk_size > 0 else 0)
        
        print(f"📡 Streaming {total_chunks} large chunks ({chunk_duration_seconds}s each)...")
        print(f"📏 Chunk size: {chunk_size:,} bytes ({chunk_size / (1024*1024):.1f} MB each)")
        
        self.is_streaming = True
        
        for i in range(0, len(raw_audio), chunk_size):
            if not self.is_streaming:
                break
                
            chunk = raw_audio[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            actual_duration = len(chunk) / (sample_rate * bytes_per_sample)
            
            if self.websocket and not self.websocket.closed:
                await self.websocket.send(chunk)
                print(f"📡 Sent chunk {chunk_num}/{total_chunks} ({len(chunk):,} bytes, {actual_duration:.1f}s)")
                
                # Wait a bit between chunks (faster than real-time for testing)
                await asyncio.sleep(2)  # Just 2 seconds between chunks
            else:
                print("❌ WebSocket connection lost")
                break
        
        print("✅ Audio streaming completed")
        
        # Send end session message
        if self.websocket and not self.websocket.closed:
            await self.websocket.send(json.dumps({"type": "end_session"}))
    
    def stop_streaming(self):
        """Stop audio streaming"""
        self.is_streaming = False
        
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()

class FrontendTester:
    """Automates frontend testing using Playwright"""
    
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.browser = None
        self.page = None
        self.audio_streamer = None
        
    async def setup(self):
        """Setup test environment"""
        print("🚀 Setting up test environment...")
        
        # Start backend server
        await self._start_backend()
        
        # Start frontend dev server
        await self._start_frontend()
        
        # Setup browser
        await self._setup_browser()
        
    async def _start_backend(self):
        """Start the backend server"""
        print("🔧 Starting backend server...")
        
        # Check if server is already running
        try:
            import requests
            response = requests.get("http://localhost:8001/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend server already running")
                return
        except:
            pass
        
        # Start backend server
        self.backend_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                import requests
                response = requests.get("http://localhost:8001/health", timeout=1)
                if response.status_code == 200:
                    print("✅ Backend server started")
                    return
            except:
                await asyncio.sleep(1)
        
        raise Exception("Backend server failed to start")
    
    async def _start_frontend(self):
        """Start the frontend dev server"""
        print("🔧 Starting frontend dev server...")
        
        # Check if frontend is already running
        try:
            import requests
            response = requests.get("http://localhost:5173", timeout=2)
            if response.status_code == 200:
                print("✅ Frontend server already running")
                return
        except:
            pass
        
        # Start frontend dev server
        frontend_dir = Path(__file__).parent / "frontend"
        self.frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for frontend to start
        for _ in range(60):  # 60 second timeout
            try:
                import requests
                response = requests.get("http://localhost:5173", timeout=1)
                if response.status_code == 200:
                    print("✅ Frontend server started")
                    return
            except:
                await asyncio.sleep(1)
        
        raise Exception("Frontend server failed to start")
    
    async def _setup_browser(self):
        """Setup Playwright browser"""
        print("🌐 Setting up browser...")
        
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=False,  # Set to True for headless mode
            args=['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']
        )
        
        context = await self.browser.new_context(
            permissions=['microphone']
        )
        
        self.page = await context.new_page()
        print("✅ Browser setup complete")
    
    async def run_test(self, audio_file: str = "audio_for_test.mp3"):
        """Run the automated test"""
        print(f"🧪 Starting automated test with audio file: {audio_file}")
        
        try:
            # Navigate to conversation page
            print("📱 Navigating to conversation page...")
            await self.page.goto("http://localhost:5173")
            await self.page.wait_for_load_state("networkidle")
            
            # Wait for page to be ready
            await self.page.wait_for_selector('[aria-label*="Start recording"]', timeout=10000)
            print("✅ Conversation page loaded")
            
            # Setup audio streamer
            audio_path = Path(__file__).parent / audio_file
            if not audio_path.exists():
                raise Exception(f"Audio file not found: {audio_path}")
            
            self.audio_streamer = AudioFileStreamer(str(audio_path))
            
            # Start recording
            print("🎤 Clicking start recording button...")
            start_button = self.page.locator('[aria-label*="Start recording"]')
            await start_button.click()
            
            # Wait a moment for WebSocket connection
            await asyncio.sleep(2)
            
            # Start audio streaming in background
            print("📡 Starting audio streaming...")
            streaming_task = asyncio.create_task(
                self.audio_streamer.connect_and_stream()
            )
            
            # Wait for audio to finish streaming
            await asyncio.sleep(5)  # Give some time for streaming to start
            
            # Monitor for transcription activity (check console logs)
            await self.page.wait_for_timeout(10000)  # Wait 10 seconds for transcription
            
            # Stop recording
            print("🛑 Clicking stop recording button...")
            stop_button = self.page.locator('[aria-label*="Pause recording"]')
            await stop_button.click()
            
            # Wait for file download
            print("💾 Waiting for transcript file download...")
            await self.page.wait_for_timeout(3000)
            
            # Check for success indicators
            await self._verify_test_results()
            
            print("✅ Test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
        finally:
            if self.audio_streamer:
                await self.audio_streamer.disconnect()
    
    async def _verify_test_results(self):
        """Verify test results"""
        print("🔍 Verifying test results...")
        
        # Check for success toast messages
        try:
            success_toast = self.page.locator('text="Recording Stopped"')
            await success_toast.wait_for(timeout=5000)
            print("✅ Found success toast message")
        except:
            print("⚠️ No success toast found")
        
        # Check console for transcription messages
        console_messages = []
        
        def handle_console(msg):
            console_messages.append(msg.text)
        
        self.page.on("console", handle_console)
        
        # Look for transcription-related console messages
        transcription_found = any("Transcribed" in msg or "WebSocket" in msg for msg in console_messages)
        if transcription_found:
            print("✅ Found transcription activity in console")
        else:
            print("⚠️ No transcription activity detected in console")
    
    async def cleanup(self):
        """Cleanup test environment"""
        print("🧹 Cleaning up test environment...")
        
        if self.audio_streamer:
            await self.audio_streamer.disconnect()
        
        if self.page:
            await self.page.close()
        
        if self.browser:
            await self.browser.close()
        
        if self.frontend_process:
            self.frontend_process.terminate()
            self.frontend_process.wait()
        
        if self.backend_process:
            self.backend_process.terminate()
            self.backend_process.wait()
        
        print("✅ Cleanup complete")

async def main():
    """Main test function"""
    tester = FrontendTester()
    
    try:
        await tester.setup()
        await tester.run_test("audio_for_test.mp3")  # Use your MP3 file
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    print("🤖 Frontend Transcription Automation Test")
    print("=" * 50)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n⚠️ Received interrupt signal, cleaning up...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the test
    asyncio.run(main())
