#!/usr/bin/env python3
"""
Simple Audio File WebSocket Transcription Test

This script streams audio files (MP3, WAV, FLAC, etc.) through the WebSocket 
transcription pipeline without requiring database connectivity. Perfect for 
testing the core transcription functionality.

Supported formats: MP3, WAV, FLAC, OGG, M4A, and more (via librosa)

Usage:
    python simple_wav_test.py

Configuration:
    Edit the HARDCODED CONFIGURATION section in main() to set:
    - AUDIO_FILE: Path to your audio file (MP3, WAV, FLAC, etc.)
    - CHUNK_SIZE: Audio chunk duration in seconds
    - WEBSOCKET_URL: WebSocket endpoint URL
    - OUTPUT_FILE: Text file to save transcripts
"""

import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
import librosa
import sys
import time
from pathlib import Path
from typing import List, Dict, Any


async def test_audio_transcription(
    audio_file_path: str,
    websocket_url: str = "ws://localhost:8000/ws/transcribe",
    chunk_duration: float = 30.0,
    sample_rate: int = 16000,
    realtime_simulation: bool = False,
    wait_time: float = 30.0,
    output_file: str = "transcript_output_30.txt"
):
    """
    Stream a WAV file through WebSocket transcription and collect results.
    
    Args:
        audio_file_path: Path to audio file (MP3, WAV, FLAC, etc.)
        websocket_url: WebSocket endpoint
        chunk_duration: Chunk size in seconds
        sample_rate: Target sample rate
        realtime_simulation: Whether to simulate real-time streaming
        wait_time: How long to wait for results
    
    Returns:
        List of transcript dictionaries
    """
    
    # Validate file exists
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"🎵 SIMPLE AUDIO WEBSOCKET TRANSCRIPTION TEST")
    print("=" * 60)
    print(f"📁 File: {audio_path}")
    print(f"⏱️ Chunk size: {chunk_duration}s")
    print(f"🎛️ Sample rate: {sample_rate}Hz")
    print(f"⚡ Real-time: {realtime_simulation}")
    print(f"⏳ Wait time: {wait_time}s")
    print("=" * 60)
    
    # Load and preprocess audio
    print(f"📁 Loading audio file...")
    try:
        # Try librosa first (supports MP3, WAV, FLAC, etc.)
        audio_data, original_sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        
        # If stereo, librosa returns (channels, samples), we need (samples, channels) or just (samples,)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.T  # Transpose to (samples, channels)
    except Exception as e:
        # Fallback to soundfile for WAV files
        try:
            audio_data, original_sample_rate = sf.read(str(audio_path))
            print(f"📁 Loaded using soundfile (WAV format)")
        except Exception as e2:
            raise Exception(f"Could not load audio file. Librosa error: {e}, Soundfile error: {e2}")
    
    print(f"📁 Loaded using librosa (supports MP3, WAV, FLAC, etc.)")
    
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        print(f"🔄 Converted stereo to mono")
    
    # Resample if needed
    if original_sample_rate != sample_rate:
        print(f"🔄 Resampling from {original_sample_rate}Hz to {sample_rate}Hz")
        audio_data = librosa.resample(
            audio_data, 
            orig_sr=original_sample_rate, 
            target_sr=sample_rate
        )
    
    duration = len(audio_data) / sample_rate
    print(f"✅ Audio loaded: {duration:.2f} seconds")
    
    # Split into chunks
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    
    print(f"🔪 Split into {len(chunks)} chunks of {chunk_duration}s each")
    
    # Storage for results
    transcripts = []
    session_id = None
    
    # Initialize output file
    output_path = Path(output_file)
    print(f"💾 Transcript will be saved to: {output_path.absolute()}")
    
    # Clear/create the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"🎵 AUDIO TRANSCRIPTION LOG\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"📁 Audio File: {audio_path}\n")
        f.write(f"🕐 Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🌐 WebSocket: {websocket_url}\n")
        f.write(f"=" * 60 + "\n\n")
    
    try:
        print(f"🔌 Connecting to WebSocket: {websocket_url}")
        async with websockets.connect(websocket_url) as websocket:
            
            # Get initial connection message
            try:
                initial_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                connection_data = json.loads(initial_message)
                
                if connection_data.get("type") == "connected":
                    session_id = connection_data.get("session_id")
                    print(f"✅ Connected! Session ID: {session_id}")
                else:
                    print(f"⚠️ Unexpected connection message: {connection_data}")
            except asyncio.TimeoutError:
                print(f"❌ Timeout waiting for connection message")
                return []
            
            # Start listening for responses
            async def listen_for_responses():
                """Listen for transcript responses and save them to file"""
                transcript_counter = 0
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "transcript":
                            transcript_counter += 1
                            transcript_data = {
                                "text": data.get("text", ""),
                                "confidence": data.get("confidence", 0.0),
                                "timestamp": data.get("timestamp"),
                                "is_final": data.get("is_final", False)
                            }
                            
                            transcripts.append(transcript_data)
                            
                            # Display transcript
                            text = transcript_data["text"]
                            confidence = transcript_data["confidence"]
                            is_final = "FINAL" if transcript_data["is_final"] else "PARTIAL"
                            
                            print(f"📝 [{is_final}] \"{text}\" (confidence: {confidence:.3f})")
                            
                            # Save transcript chunk to file immediately
                            try:
                                with open(output_path, 'a', encoding='utf-8') as f:
                                    timestamp_str = transcript_data.get("timestamp", time.strftime('%H:%M:%S'))
                                    f.write(f"[{transcript_counter:03d}] [{timestamp_str}] [{is_final}] (conf: {confidence:.3f})\n")
                                    f.write(f"    {text}\n\n")
                                    f.flush()  # Ensure immediate write to disk
                            except Exception as e:
                                print(f"⚠️ Failed to save transcript chunk: {e}")
                        
                        elif data.get("type") == "error":
                            error_msg = data.get('message', 'Unknown error')
                            print(f"❌ Error: {error_msg}")
                            
                            # Save error to file
                            try:
                                with open(output_path, 'a', encoding='utf-8') as f:
                                    f.write(f"❌ ERROR: {error_msg}\n\n")
                                    f.flush()
                            except:
                                pass
                        
                        else:
                            print(f"📨 Other: {data.get('type', 'unknown')}")
                    
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"❌ Listen error: {e}")
                        break
            
            # Start listening task
            listen_task = asyncio.create_task(listen_for_responses())
            
            # Stream audio chunks
            print(f"\\n🎵 Streaming audio chunks...")
            for i, chunk in enumerate(chunks):
                # Convert to int16 format
                audio_int16 = (chunk * 32767).astype(np.int16)
                
                # Send chunk
                await websocket.send(audio_int16.tobytes())
                
                chunk_duration_actual = len(chunk) / sample_rate
                print(f"📤 Sent chunk {i+1}/{len(chunks)} ({chunk_duration_actual:.2f}s)")
                
                # Simulate real-time if requested
                if realtime_simulation:
                    await asyncio.sleep(chunk_duration_actual)
                else:
                    await asyncio.sleep(0.1)  # Small delay
            
            print(f"✅ All chunks sent! Waiting {wait_time}s for results...")
            
            # Wait for transcription results
            await asyncio.sleep(wait_time)
            
            # Cancel listening task
            listen_task.cancel()
    
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 WebSocket connection closed")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return []
    
    # Print results summary
    print(f"\\n" + "=" * 60)
    print(f"🎯 TRANSCRIPTION RESULTS")
    print("=" * 60)
    
    if not transcripts:
        print(f"❌ No transcripts received")
        return []
    
    print(f"📊 Total transcripts: {len(transcripts)}")
    print(f"🆔 Session ID: {session_id}")
    
    # Show all transcripts
    print(f"\\n📝 All Transcripts:")
    for i, transcript in enumerate(transcripts):
        text = transcript["text"]
        confidence = transcript["confidence"]
        is_final = "FINAL" if transcript.get("is_final") else "PARTIAL"
        
        print(f"  {i+1:2d}. [{is_final:7s}] \"{text}\" (conf: {confidence:.3f})")
    
    # Show final combined transcript
    final_transcripts = [t for t in transcripts if t.get("is_final", False)]
    if final_transcripts:
        combined_text = " ".join([t["text"] for t in final_transcripts])
        avg_confidence = sum([t["confidence"] for t in final_transcripts]) / len(final_transcripts)
        
        print(f"\\n🎯 FINAL COMBINED TRANSCRIPT:")
        print(f"📄 \"{combined_text}\"")
        print(f"📊 Average Confidence: {avg_confidence:.3f}")
        print(f"🔢 Final Segments: {len(final_transcripts)}")
        
        # Save final summary to file
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"\n" + "=" * 60 + "\n")
                f.write(f"🎯 FINAL TRANSCRIPTION SUMMARY\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"🕐 Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"📊 Total Segments: {len(transcripts)}\n")
                f.write(f"📊 Final Segments: {len(final_transcripts)}\n")
                f.write(f"📊 Average Confidence: {avg_confidence:.3f}\n")
                f.write(f"🆔 Session ID: {session_id}\n\n")
                f.write(f"🎯 COMPLETE TRANSCRIPT:\n")
                f.write(f"{'-' * 40}\n")
                f.write(f"{combined_text}\n")
                f.write(f"{'-' * 40}\n")
                f.flush()
            
            print(f"💾 Complete transcript saved to: {output_path.absolute()}")
        except Exception as e:
            print(f"⚠️ Failed to save final summary: {e}")
    
    return transcripts


async def main():
    """Main function with hardcoded configuration."""
    
    # HARDCODED CONFIGURATION - Modify these values as needed
    AUDIO_FILE = "harvard.wav"        # Change this to your audio file
    CHUNK_SIZE = 5.0                  # Audio chunk duration in seconds
    SAMPLE_RATE = 16000               # Target sample rate
    REALTIME_SIMULATION = False       # Set to True to simulate real-time streaming
    WAIT_TIME = 20.0                  # Time to wait for results in seconds
    WEBSOCKET_URL = "ws://localhost:8001/ws/transcribe"  # WebSocket endpoint
    OUTPUT_FILE = "transcript_debug.txt"  # Output file for saving transcripts
    
    print(f"🎵 HARDCODED CONFIGURATION:")
    print(f"📁 Audio file: {AUDIO_FILE}")
    print(f"⏱️ Chunk size: {CHUNK_SIZE}s")
    print(f"🎛️ Sample rate: {SAMPLE_RATE}Hz")
    print(f"⚡ Real-time: {REALTIME_SIMULATION}")
    print(f"⏳ Wait time: {WAIT_TIME}s")
    print(f"🌐 WebSocket URL: {WEBSOCKET_URL}")
    print(f"💾 Output file: {OUTPUT_FILE}")
    print("=" * 60)
    
    # Run transcription test
    try:
        transcripts = await test_audio_transcription(
            audio_file_path=AUDIO_FILE,
            websocket_url=WEBSOCKET_URL,
            chunk_duration=CHUNK_SIZE,
            sample_rate=SAMPLE_RATE,
            realtime_simulation=REALTIME_SIMULATION,
            wait_time=WAIT_TIME,
            output_file=OUTPUT_FILE
        )
        
        if transcripts:
            print(f"\\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            print(f"\\n❌ No transcripts received")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\\n⚠️ Interrupted by user")
        sys.exit(1)
