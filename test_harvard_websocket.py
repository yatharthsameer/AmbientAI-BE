#!/usr/bin/env python3
"""
Test WebSocket transcription with harvard.wav file.
This script reads the harvard.wav file and streams it through the WebSocket for real-time transcription.
"""

import asyncio
import websockets
import json
import soundfile as sf
import numpy as np
import time
from pathlib import Path

class HarvardWebSocketTester:
    def __init__(self, wav_file_path="harvard.wav", websocket_url="ws://localhost:8000/ws/transcribe"):
        self.wav_file_path = Path(wav_file_path)
        self.websocket_url = websocket_url
        self.websocket = None
        self.session_id = None
        self.received_transcripts = []
        
    async def connect(self):
        """Connect to the WebSocket endpoint."""
        print(f"🔌 Connecting to {self.websocket_url}...")
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            print("✅ Connected successfully!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def wait_for_initial_message(self):
        """Wait for the initial connection message from server."""
        try:
            initial_message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            data = json.loads(initial_message)
            print(f"📨 Initial message: {data}")
            
            if data.get("type") == "connected":
                self.session_id = data.get("session_id")
                print(f"🎯 Session ID: {self.session_id}")
                return True
            return False
        except asyncio.TimeoutError:
            print("⏰ Timeout waiting for initial message")
            return False
        except Exception as e:
            print(f"❌ Error receiving initial message: {e}")
            return False
    
    async def send_control_message(self, message_type, **kwargs):
        """Send a control message to the server."""
        message = {"type": message_type, **kwargs}
        await self.websocket.send(json.dumps(message))
        print(f"📤 Sent control message: {message}")
    
    async def listen_for_messages(self):
        """Listen for incoming messages from the server."""
        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "transcript":
                    text = data.get("text", "")
                    confidence = data.get("confidence", 0)
                    chunk_index = data.get("chunk_index", 0)
                    is_final = data.get("is_final", False)
                    
                    self.received_transcripts.append({
                        "text": text,
                        "confidence": confidence,
                        "chunk_index": chunk_index,
                        "is_final": is_final,
                        "timestamp": time.time()
                    })
                    
                    status = "🔒 FINAL" if is_final else "⏳ PARTIAL"
                    print(f"📝 {status} Transcript [Chunk {chunk_index}] (Confidence: {confidence:.3f}): {text}")
                    
                elif data.get("type") == "error":
                    print(f"❌ Server error: {data.get('message')}")
                    
                elif data.get("type") == "session_info":
                    print(f"📊 Session info: {data}")
                    
                else:
                    print(f"📨 Other message: {data}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed by server")
        except Exception as e:
            print(f"❌ Error listening for messages: {e}")
    
    async def stream_wav_file(self, chunk_duration=2.0, real_time_factor=1.0):
        """
        Stream the WAV file in chunks.
        
        Args:
            chunk_duration: Duration of each audio chunk in seconds
            real_time_factor: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, 0.5 = half speed)
        """
        if not self.wav_file_path.exists():
            print(f"❌ WAV file not found: {self.wav_file_path}")
            return False
        
        # Load the audio file
        print(f"📁 Loading audio file: {self.wav_file_path}")
        try:
            audio_data, sample_rate = sf.read(self.wav_file_path)
            print(f"🎵 Audio loaded: {len(audio_data)} samples, {sample_rate}Hz, {len(audio_data)/sample_rate:.2f}s duration")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                print("🔄 Converted stereo to mono")
            
            # Resample to 16kHz if needed (WebSocket service expects 16kHz)
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
                sample_rate = target_sample_rate
                print(f"🔄 Resampled to {target_sample_rate}Hz")
            
            # Calculate chunk size in samples
            chunk_size = int(chunk_duration * sample_rate)
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            
            print(f"📦 Streaming {total_chunks} chunks of {chunk_duration}s each")
            print(f"⚡ Real-time factor: {real_time_factor}x")
            print(f"🕒 Expected streaming time: {(len(audio_data)/sample_rate)/real_time_factor:.2f}s")
            
            # Start streaming
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(audio_data))
                chunk = audio_data[start_idx:end_idx]
                
                # Convert to int16 PCM format
                chunk_int16 = (chunk * 32767).astype(np.int16)
                chunk_bytes = chunk_int16.tobytes()
                
                # Send the audio chunk
                await self.websocket.send(chunk_bytes)
                
                progress = (i + 1) / total_chunks * 100
                current_time = (end_idx / sample_rate)
                print(f"📤 Sent chunk {i+1}/{total_chunks} ({progress:.1f}%) - Audio time: {current_time:.2f}s")
                
                # Wait for real-time playback (adjusted by real_time_factor)
                if i < total_chunks - 1:  # Don't wait after the last chunk
                    sleep_time = chunk_duration / real_time_factor
                    await asyncio.sleep(sleep_time)
            
            print("✅ Finished streaming all audio chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error streaming WAV file: {e}")
            return False
    
    async def run_test(self, chunk_duration=2.0, real_time_factor=1.0, wait_time=60.0):
        """
        Run the complete test: connect, stream, and collect results.
        
        Args:
            chunk_duration: Duration of each audio chunk in seconds
            real_time_factor: Speed multiplier for streaming
            wait_time: How long to wait after streaming for final results
        """
        print("🎤 Harvard WebSocket Transcription Test")
        print("=" * 50)
        
        # Connect to WebSocket
        if not await self.connect():
            return False
        
        # Wait for initial connection message
        if not await self.wait_for_initial_message():
            await self.websocket.close()
            return False
        
        # Start listening for messages in the background
        listen_task = asyncio.create_task(self.listen_for_messages())
        
        # Send a ping to test control messages
        await self.send_control_message("ping")
        await asyncio.sleep(1)
        
        # Stream the audio file
        print(f"\n🎵 Starting to stream {self.wav_file_path}...")
        stream_success = await self.stream_wav_file(chunk_duration, real_time_factor)
        
        if stream_success:
            print(f"\n⏳ Waiting {wait_time}s for final transcription results...")
            await asyncio.sleep(wait_time)
        
        # Send session info request to get final results
        await self.send_control_message("get_session_info")
        await asyncio.sleep(2)
        
        # Close connection
        await self.websocket.close()
        listen_task.cancel()
        
        # Print summary
        self.print_results_summary()
        
        return stream_success
    
    def print_results_summary(self):
        """Print a summary of the transcription results."""
        print("\n" + "=" * 50)
        print("📊 TRANSCRIPTION RESULTS SUMMARY")
        print("=" * 50)
        
        if not self.received_transcripts:
            print("❌ No transcripts received")
            return
        
        print(f"📝 Total transcript chunks received: {len(self.received_transcripts)}")
        
        # Group by chunk_index and show progression
        chunks = {}
        for t in self.received_transcripts:
            chunk_idx = t["chunk_index"]
            if chunk_idx not in chunks:
                chunks[chunk_idx] = []
            chunks[chunk_idx].append(t)
        
        print(f"🔢 Unique chunks processed: {len(chunks)}")
        
        # Show final transcripts for each chunk
        print("\n📋 Final transcripts by chunk:")
        full_transcript = []
        
        for chunk_idx in sorted(chunks.keys()):
            chunk_transcripts = chunks[chunk_idx]
            # Get the final transcript for this chunk (last one or one marked as final)
            final_transcript = None
            for t in reversed(chunk_transcripts):
                if t["is_final"] or t == chunk_transcripts[-1]:
                    final_transcript = t
                    break
            
            if final_transcript and final_transcript["text"].strip():
                text = final_transcript["text"].strip()
                confidence = final_transcript["confidence"]
                print(f"  Chunk {chunk_idx}: [{confidence:.3f}] {text}")
                full_transcript.append(text)
        
        # Show complete transcript
        if full_transcript:
            complete_text = " ".join(full_transcript)
            print(f"\n🎯 COMPLETE TRANSCRIPT:")
            print(f"   {complete_text}")
            print(f"\n📏 Total length: {len(complete_text)} characters")
            
            # Calculate average confidence
            confidences = [t["confidence"] for t in self.received_transcripts if t["confidence"] > 0]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                print(f"📊 Average confidence: {avg_confidence:.3f}")
        else:
            print("❌ No meaningful transcription text received")

async def main():
    """Main function to run the test."""
    tester = HarvardWebSocketTester("harvard.wav")
    
    # Test with 2-second chunks at real-time speed
    success = await tester.run_test(
        chunk_duration=2.0,
        real_time_factor=1.0,  # Real-time streaming
        wait_time=60.0  # Wait longer for transcription results
    )
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")

if __name__ == "__main__":
    asyncio.run(main())
