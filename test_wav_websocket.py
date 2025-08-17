#!/usr/bin/env python3
"""
WebSocket WAV File Transcription Test Script

This script takes any WAV file and streams it through the WebSocket transcription
pipeline, simulating real-time audio input. Perfect for testing without using
a microphone repeatedly.

Usage:
    python test_wav_websocket.py <wav_file_path> [options]

Examples:
    python test_wav_websocket.py harvard.wav
    python test_wav_websocket.py my_audio.wav --chunk-size 2.0 --realtime
    python test_wav_websocket.py test.wav --session-id my-test-session
"""

import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
import librosa
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any


class WAVWebSocketStreamer:
    """Streams WAV files through WebSocket transcription pipeline."""
    
    def __init__(self, 
                 wav_file_path: str,
                 websocket_url: str = "ws://localhost:8000/ws/transcribe",
                 chunk_duration: float = 2.0,
                 sample_rate: int = 16000,
                 realtime_simulation: bool = False,
                 session_id: Optional[str] = None):
        """
        Initialize the WAV WebSocket streamer.
        
        Args:
            wav_file_path: Path to the WAV file to stream
            websocket_url: WebSocket endpoint URL
            chunk_duration: Duration of each audio chunk in seconds
            sample_rate: Target sample rate for audio
            realtime_simulation: If True, simulate real-time streaming with delays
            session_id: Optional custom session ID
        """
        self.wav_file_path = Path(wav_file_path)
        self.websocket_url = websocket_url
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.realtime_simulation = realtime_simulation
        self.session_id = session_id
        
        # Results storage
        self.transcripts: List[Dict[str, Any]] = []
        self.session_info: Optional[Dict[str, Any]] = None
        self.connected_session_id: Optional[str] = None
        
    def load_audio(self) -> np.ndarray:
        """Load and preprocess the WAV file."""
        if not self.wav_file_path.exists():
            raise FileNotFoundError(f"WAV file not found: {self.wav_file_path}")
        
        print(f"📁 Loading audio file: {self.wav_file_path}")
        
        # Load audio file
        audio_data, original_sample_rate = sf.read(str(self.wav_file_path))
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            print(f"🔄 Converted stereo to mono")
        
        # Resample if needed
        if original_sample_rate != self.sample_rate:
            print(f"🔄 Resampling from {original_sample_rate}Hz to {self.sample_rate}Hz")
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=original_sample_rate, 
                target_sr=self.sample_rate
            )
        
        duration = len(audio_data) / self.sample_rate
        print(f"✅ Audio loaded: {duration:.2f} seconds, {self.sample_rate}Hz")
        
        return audio_data
    
    def chunk_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Split audio into chunks for streaming."""
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        chunks = []
        
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            if len(chunk) > 0:  # Only add non-empty chunks
                chunks.append(chunk)
        
        print(f"🔪 Split audio into {len(chunks)} chunks of {self.chunk_duration}s each")
        return chunks
    
    async def connect_websocket(self) -> websockets.WebSocketServerProtocol:
        """Connect to the WebSocket endpoint."""
        if self.session_id:
            # Use custom session ID endpoint
            url = f"ws://localhost:8000/ws/transcribe/{self.session_id}"
            print(f"🔌 Connecting to WebSocket with custom session ID: {self.session_id}")
        else:
            # Use auto-generated session ID endpoint
            url = self.websocket_url
            print(f"🔌 Connecting to WebSocket: {url}")
        
        websocket = await websockets.connect(url)
        
        # Get initial connection message
        initial_message = await websocket.recv()
        connection_data = json.loads(initial_message)
        
        if connection_data.get("type") == "connected":
            self.connected_session_id = connection_data.get("session_id")
            self.session_info = connection_data
            print(f"✅ Connected! Session ID: {self.connected_session_id}")
            print(f"📊 Session Info: {json.dumps(connection_data, indent=2)}")
        else:
            print(f"⚠️ Unexpected connection message: {connection_data}")
        
        return websocket
    
    async def stream_audio_chunks(self, websocket: websockets.WebSocketServerProtocol, chunks: List[np.ndarray]):
        """Stream audio chunks to the WebSocket."""
        print(f"\n🎵 Starting audio streaming...")
        print(f"📡 Chunks to send: {len(chunks)}")
        print(f"⏱️ Real-time simulation: {'ON' if self.realtime_simulation else 'OFF'}")
        print("=" * 60)
        
        for i, chunk in enumerate(chunks):
            # Convert to int16 format for WebSocket transmission
            audio_int16 = (chunk * 32767).astype(np.int16)
            
            # Send audio chunk
            await websocket.send(audio_int16.tobytes())
            
            chunk_duration_actual = len(chunk) / self.sample_rate
            print(f"📤 Sent chunk {i+1}/{len(chunks)} ({chunk_duration_actual:.2f}s, {len(audio_int16.tobytes())} bytes)")
            
            # Simulate real-time streaming if requested
            if self.realtime_simulation:
                await asyncio.sleep(chunk_duration_actual)
            else:
                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.1)
        
        print(f"✅ All {len(chunks)} chunks sent!")
    
    async def listen_for_transcripts(self, websocket: websockets.WebSocketServerProtocol):
        """Listen for transcript messages from the WebSocket."""
        print(f"\n👂 Listening for transcripts...")
        
        try:
            while True:
                try:
                    # Wait for messages with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "transcript":
                        transcript_data = {
                            "text": data.get("text", ""),
                            "confidence": data.get("confidence", 0.0),
                            "timestamp": data.get("timestamp"),
                            "chunk_id": data.get("chunk_id"),
                            "is_final": data.get("is_final", False)
                        }
                        
                        self.transcripts.append(transcript_data)
                        
                        # Display transcript
                        text = transcript_data["text"]
                        confidence = transcript_data["confidence"]
                        is_final = "FINAL" if transcript_data["is_final"] else "PARTIAL"
                        
                        print(f"📝 [{is_final}] \"{text}\" (confidence: {confidence:.3f})")
                    
                    elif data.get("type") == "error":
                        print(f"❌ Error: {data.get('message', 'Unknown error')}")
                    
                    elif data.get("type") == "session_info":
                        print(f"📊 Session update: {data}")
                    
                    else:
                        print(f"📨 Other message: {data}")
                
                except asyncio.TimeoutError:
                    # No message received in timeout period
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error listening for transcripts: {e}")
    
    async def run_transcription(self, wait_time: float = 30.0) -> List[Dict[str, Any]]:
        """Run the complete transcription process."""
        try:
            # Load and prepare audio
            audio_data = self.load_audio()
            chunks = self.chunk_audio(audio_data)
            
            # Connect to WebSocket
            websocket = await self.connect_websocket()
            
            # Start listening for transcripts in background
            listen_task = asyncio.create_task(self.listen_for_transcripts(websocket))
            
            # Stream audio chunks
            await self.stream_audio_chunks(websocket, chunks)
            
            # Wait for transcription results
            print(f"\n⏳ Waiting {wait_time}s for transcription results...")
            await asyncio.sleep(wait_time)
            
            # Cancel listening task and close connection
            listen_task.cancel()
            await websocket.close()
            
            return self.transcripts
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return []
    
    def print_results(self):
        """Print final transcription results."""
        print("\n" + "=" * 60)
        print("🎯 TRANSCRIPTION RESULTS")
        print("=" * 60)
        
        if not self.transcripts:
            print("❌ No transcripts received")
            return
        
        print(f"📊 Total transcripts received: {len(self.transcripts)}")
        print(f"🆔 Session ID: {self.connected_session_id}")
        
        # Show all transcripts
        print(f"\n📝 All Transcripts:")
        for i, transcript in enumerate(self.transcripts):
            text = transcript["text"]
            confidence = transcript["confidence"]
            is_final = "FINAL" if transcript.get("is_final") else "PARTIAL"
            timestamp = transcript.get("timestamp", "N/A")
            
            print(f"  {i+1:2d}. [{is_final:7s}] \"{text}\" (conf: {confidence:.3f}, time: {timestamp})")
        
        # Show final combined transcript
        final_transcripts = [t for t in self.transcripts if t.get("is_final", False)]
        if final_transcripts:
            combined_text = " ".join([t["text"] for t in final_transcripts])
            avg_confidence = sum([t["confidence"] for t in final_transcripts]) / len(final_transcripts)
            
            print(f"\n🎯 FINAL COMBINED TRANSCRIPT:")
            print(f"📄 Text: \"{combined_text}\"")
            print(f"📊 Average Confidence: {avg_confidence:.3f}")
            print(f"🔢 Final Segments: {len(final_transcripts)}")


async def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Stream WAV file through WebSocket transcription pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_wav_websocket.py harvard.wav
  python test_wav_websocket.py audio.wav --chunk-size 3.0 --realtime
  python test_wav_websocket.py test.wav --session-id my-session --wait-time 60
        """
    )
    
    parser.add_argument("wav_file", help="Path to WAV file to transcribe")
    parser.add_argument("--chunk-size", type=float, default=2.0, 
                       help="Duration of each audio chunk in seconds (default: 2.0)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Target sample rate (default: 16000)")
    parser.add_argument("--realtime", action="store_true",
                       help="Simulate real-time streaming with delays")
    parser.add_argument("--session-id", type=str,
                       help="Custom session ID (optional)")
    parser.add_argument("--wait-time", type=float, default=30.0,
                       help="Time to wait for transcription results in seconds (default: 30)")
    parser.add_argument("--websocket-url", default="ws://localhost:8000/ws/transcribe",
                       help="WebSocket endpoint URL (default: ws://localhost:8000/ws/transcribe)")
    
    args = parser.parse_args()
    
    # Validate WAV file exists
    if not Path(args.wav_file).exists():
        print(f"❌ Error: WAV file not found: {args.wav_file}")
        sys.exit(1)
    
    print("🎵 WAV FILE WEBSOCKET TRANSCRIPTION TEST")
    print("=" * 60)
    print(f"📁 File: {args.wav_file}")
    print(f"⏱️ Chunk size: {args.chunk_size}s")
    print(f"🎛️ Sample rate: {args.sample_rate}Hz")
    print(f"⚡ Real-time: {args.realtime}")
    print(f"🆔 Session ID: {args.session_id or 'Auto-generated'}")
    print(f"⏳ Wait time: {args.wait_time}s")
    print("=" * 60)
    
    # Create streamer and run transcription
    streamer = WAVWebSocketStreamer(
        wav_file_path=args.wav_file,
        websocket_url=args.websocket_url,
        chunk_duration=args.chunk_size,
        sample_rate=args.sample_rate,
        realtime_simulation=args.realtime,
        session_id=args.session_id
    )
    
    # Run transcription
    transcripts = await streamer.run_transcription(wait_time=args.wait_time)
    
    # Print results
    streamer.print_results()
    
    # Exit with appropriate code
    if transcripts:
        print(f"\n✅ Transcription completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Transcription failed or no results received")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
