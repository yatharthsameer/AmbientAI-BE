"""
WebSocket test client for real-time transcription.
This script demonstrates how to connect to the WebSocket transcription service.
"""
import asyncio
import websockets
import json
import numpy as np
import time
import struct
import argparse
from pathlib import Path


class WebSocketTranscriptionClient:
    """Test client for WebSocket transcription service."""
    
    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.port = port
        self.websocket = None
        self.session_id = None
        
        # Audio parameters (must match server settings)
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 0.5  # Send smaller chunks for testing
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
    async def connect(self, session_id=None):
        """Connect to the WebSocket transcription service."""
        try:
            if session_id:
                uri = f"ws://{self.host}:{self.port}/ws/transcribe/{session_id}"
                self.session_id = session_id
            else:
                uri = f"ws://{self.host}:{self.port}/ws/transcribe"
                self.session_id = "auto-generated"
            
            print(f"Connecting to {uri}...")
            self.websocket = await websockets.connect(uri)
            print(f"Connected successfully! Session ID: {self.session_id}")
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket service."""
        if self.websocket:
            try:
                # Send end session message
                await self.send_control_message({"type": "end_session"})
                await self.websocket.close()
                print("Disconnected successfully")
            except Exception as e:
                print(f"Disconnect error: {e}")
    
    async def send_control_message(self, message):
        """Send a control message to the server."""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                print(f"Error sending control message: {e}")
    
    async def send_audio_chunk(self, audio_data):
        """Send audio data to the server."""
        if self.websocket:
            try:
                # Convert audio to PCM 16-bit format
                if audio_data.dtype != np.int16:
                    # Convert float to int16
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data
                
                # Send as binary data
                await self.websocket.send(audio_int16.tobytes())
                
            except Exception as e:
                print(f"Error sending audio chunk: {e}")
    
    async def listen_for_responses(self):
        """Listen for responses from the server."""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                try:
                    response = json.loads(message)
                    await self.handle_response(response)
                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
        except Exception as e:
            print(f"Error listening for responses: {e}")
    
    async def handle_response(self, response):
        """Handle response messages from the server."""
        message_type = response.get("type", "unknown")
        
        if message_type == "connected":
            print(f"✅ Connected - {response.get('message', '')}")
            config = response.get("config", {})
            print(f"   Server config: {config}")
            
        elif message_type == "transcript":
            text = response.get("text", "")
            confidence = response.get("confidence", 0.0)
            chunk_index = response.get("chunk_index", 0)
            processing_time = response.get("processing_time_ms", 0)
            
            print(f"🎤 [{chunk_index}] {text} (confidence: {confidence:.2f}, {processing_time:.0f}ms)")
            
        elif message_type == "error":
            error_msg = response.get("message", "Unknown error")
            print(f"❌ Error: {error_msg}")
            
        elif message_type == "session_info":
            print(f"📊 Session Info:")
            print(f"   Total chunks: {response.get('total_chunks', 0)}")
            print(f"   Full transcript: {response.get('full_transcript', '')[:100]}...")
            metrics = response.get("metrics", {})
            if metrics:
                print(f"   Avg processing time: {metrics.get('avg_processing_time_ms', 0):.1f}ms")
                print(f"   Real-time factor: {metrics.get('real_time_factor', 0):.2f}")
            
        elif message_type == "session_ended":
            print(f"🏁 Session ended: {response.get('message', '')}")
            
        elif message_type == "pong":
            print("🏓 Pong received")
            
        else:
            print(f"📨 Received: {response}")
    
    async def test_with_synthetic_audio(self, duration=10):
        """Test with synthetic audio (sine wave)."""
        print(f"\n🎵 Testing with synthetic audio for {duration} seconds...")
        
        # Generate sine wave audio
        frequency = 440  # A4 note
        total_samples = int(self.sample_rate * duration)
        
        # Generate audio in chunks
        samples_sent = 0
        start_time = time.time()
        
        while samples_sent < total_samples:
            # Calculate current chunk size
            remaining_samples = total_samples - samples_sent
            current_chunk_size = min(self.chunk_size, remaining_samples)
            
            # Generate sine wave chunk
            t = np.arange(samples_sent, samples_sent + current_chunk_size) / self.sample_rate
            audio_chunk = 0.1 * np.sin(2 * np.pi * frequency * t)  # Low amplitude
            
            # Send audio chunk
            await self.send_audio_chunk(audio_chunk)
            samples_sent += current_chunk_size
            
            # Show progress
            progress = (samples_sent / total_samples) * 100
            print(f"Progress: {progress:.1f}% ({samples_sent}/{total_samples} samples)")
            
            # Wait for real-time playback
            elapsed = time.time() - start_time
            expected_time = samples_sent / self.sample_rate
            if elapsed < expected_time:
                await asyncio.sleep(expected_time - elapsed)
        
        print(f"✅ Finished sending {duration} seconds of synthetic audio")
    
    async def test_with_audio_file(self, file_path):
        """Test with an audio file."""
        try:
            import librosa
            
            print(f"\n🎵 Testing with audio file: {file_path}")
            
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            print(f"Loaded audio: {len(audio)} samples, {len(audio)/sr:.1f} seconds")
            
            # Send audio in chunks
            samples_sent = 0
            start_time = time.time()
            
            while samples_sent < len(audio):
                # Get current chunk
                end_sample = min(samples_sent + self.chunk_size, len(audio))
                audio_chunk = audio[samples_sent:end_sample]
                
                # Send audio chunk
                await self.send_audio_chunk(audio_chunk)
                samples_sent = end_sample
                
                # Show progress
                progress = (samples_sent / len(audio)) * 100
                print(f"Progress: {progress:.1f}% ({samples_sent}/{len(audio)} samples)")
                
                # Wait for real-time playback
                elapsed = time.time() - start_time
                expected_time = samples_sent / self.sample_rate
                if elapsed < expected_time:
                    await asyncio.sleep(expected_time - elapsed)
            
            print(f"✅ Finished sending audio file")
            
        except ImportError:
            print("❌ librosa not available. Please install with: pip install librosa")
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
    
    async def test_control_messages(self):
        """Test control messages."""
        print("\n🎛️  Testing control messages...")
        
        # Test ping
        await self.send_control_message({"type": "ping"})
        await asyncio.sleep(1)
        
        # Request session info
        await self.send_control_message({"type": "session_info"})
        await asyncio.sleep(1)
    
    async def run_interactive_test(self):
        """Run interactive test mode."""
        print("\n🎮 Interactive mode")
        print("Commands:")
        print("  'ping' - Send ping message")
        print("  'info' - Get session info")
        print("  'audio <duration>' - Send synthetic audio for <duration> seconds")
        print("  'file <path>' - Send audio file")
        print("  'quit' - Exit")
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command == "quit":
                    break
                elif command == "ping":
                    await self.send_control_message({"type": "ping"})
                elif command == "info":
                    await self.send_control_message({"type": "session_info"})
                elif command.startswith("audio "):
                    try:
                        duration = float(command.split()[1])
                        await self.test_with_synthetic_audio(duration)
                    except (IndexError, ValueError):
                        print("Usage: audio <duration_seconds>")
                elif command.startswith("file "):
                    try:
                        file_path = command.split()[1]
                        await self.test_with_audio_file(file_path)
                    except IndexError:
                        print("Usage: file <path>")
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="WebSocket Transcription Test Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--session-id", help="Custom session ID")
    parser.add_argument("--audio-file", help="Audio file to test with")
    parser.add_argument("--duration", type=float, default=5, help="Duration for synthetic audio test")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    client = WebSocketTranscriptionClient(args.host, args.port)
    
    try:
        # Connect to server
        connected = await client.connect(args.session_id)
        if not connected:
            return
        
        # Start listening for responses in background
        listen_task = asyncio.create_task(client.listen_for_responses())
        
        # Wait for connection confirmation
        await asyncio.sleep(1)
        
        if args.interactive:
            # Interactive mode
            await client.run_interactive_test()
        else:
            # Automated tests
            await client.test_control_messages()
            
            if args.audio_file:
                await client.test_with_audio_file(args.audio_file)
            else:
                await client.test_with_synthetic_audio(args.duration)
        
        # Wait a bit for final responses
        await asyncio.sleep(2)
        
        # Cancel listening task
        listen_task.cancel()
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("🎤 WebSocket Transcription Test Client")
    print("=" * 50)
    asyncio.run(main())
