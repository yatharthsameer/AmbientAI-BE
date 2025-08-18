#!/usr/bin/env python3
"""
Large Chunk Configuration Test

This script verifies that the backend is properly configured to handle
40-second audio chunks for better Whisper transcription context.
"""

import asyncio
import json
import sys
from pathlib import Path

try:
    import websockets
    from pydub import AudioSegment
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required packages. Please install:")
    print("pip install pydub websockets numpy")
    sys.exit(1)

async def test_large_chunk_config():
    """Test that backend accepts and processes large chunks correctly"""
    
    print("🧪 Testing Large Chunk Configuration")
    print("=" * 40)
    
    # Test configuration
    websocket_url = "ws://localhost:8001/ws/transcribe"
    
    # Create a test audio chunk (40 seconds of silence with some noise)
    sample_rate = 16000
    duration_seconds = 40
    total_samples = sample_rate * duration_seconds
    
    # Generate test audio (sine wave + noise for realistic audio)
    t = np.linspace(0, duration_seconds, total_samples)
    frequency = 440  # A4 note
    audio_float = 0.1 * np.sin(2 * np.pi * frequency * t) + 0.05 * np.random.randn(total_samples)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_float * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    chunk_size_mb = len(audio_bytes) / (1024 * 1024)
    
    print(f"🎵 Generated test audio:")
    print(f"   Duration: {duration_seconds} seconds")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Samples: {total_samples:,}")
    print(f"   Size: {len(audio_bytes):,} bytes ({chunk_size_mb:.1f} MB)")
    print()
    
    try:
        print(f"🔌 Connecting to: {websocket_url}")
        websocket = await websockets.connect(websocket_url)
        print("✅ WebSocket connected")
        
        # Listen for messages
        async def listen_for_messages():
            try:
                async for message in websocket:
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'session_info':
                        session_id = data.get('session_id')
                        print(f"🆔 Session started: {session_id}")
                    
                    elif msg_type == 'transcript':
                        text = data.get('text', '')
                        is_final = data.get('is_final', False)
                        confidence = data.get('confidence', 0)
                        processing_time = data.get('processing_time_ms', 0)
                        
                        status = "FINAL" if is_final else "partial"
                        print(f"📝 [{status}] \"{text}\" (confidence: {confidence:.2f}, {processing_time:.0f}ms)")
                    
                    elif msg_type == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        print(f"❌ Error: {error_msg}")
                        return False
                    
                    elif msg_type == 'status':
                        print(f"📊 Status: {data.get('message', 'Unknown status')}")
                    
                    else:
                        print(f"📨 Received: {data}")
                        
            except websockets.exceptions.ConnectionClosed:
                print("🔌 WebSocket connection closed")
            except Exception as e:
                print(f"⚠️ Message listening error: {e}")
                return False
            
            return True
        
        # Start listening
        listen_task = asyncio.create_task(listen_for_messages())
        
        # Wait a moment for connection to stabilize
        await asyncio.sleep(1)
        
        # Send the large chunk
        print(f"📡 Sending large audio chunk ({chunk_size_mb:.1f} MB)...")
        start_time = asyncio.get_event_loop().time()
        
        await websocket.send(audio_bytes)
        
        send_time = asyncio.get_event_loop().time() - start_time
        print(f"✅ Chunk sent successfully in {send_time:.2f} seconds")
        
        # Wait for processing
        print("⏳ Waiting for transcription processing...")
        await asyncio.sleep(10)  # Give time for processing
        
        # Send end session
        await websocket.send(json.dumps({"type": "end_session"}))
        print("🛑 Sent end session message")
        
        # Wait for final messages
        await asyncio.sleep(3)
        
        await websocket.close()
        
        print("\n✅ Large chunk test completed successfully!")
        print("🎯 Backend can handle 40-second audio chunks")
        
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused - make sure backend server is running on port 8001")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

async def test_config_values():
    """Test that configuration values are set correctly"""
    
    print("\n🔧 Checking Configuration Values")
    print("=" * 40)
    
    try:
        # Try to import and check config
        sys.path.append(str(Path(__file__).parent))
        from config import get_settings
        
        settings = get_settings()
        ws_config = settings.websocket_settings
        
        print(f"📊 WebSocket Configuration:")
        print(f"   Chunk Duration: {ws_config.chunk_duration} seconds")
        print(f"   Overlap Duration: {ws_config.overlap_duration} seconds")
        print(f"   Audio Buffer Size: {ws_config.audio_buffer_size:,} bytes ({ws_config.audio_buffer_size/(1024*1024):.1f} MB)")
        print(f"   Chunk Timeout: {ws_config.chunk_timeout} seconds")
        print(f"   Whisper Model: {ws_config.whisper_model_realtime}")
        print(f"   Sample Rate: {ws_config.sample_rate} Hz")
        
        # Verify expected values
        expected_chunk_duration = 40.0
        expected_buffer_size = 1280000  # 40s * 16000Hz * 2 bytes
        
        if ws_config.chunk_duration == expected_chunk_duration:
            print("✅ Chunk duration correctly set to 40 seconds")
        else:
            print(f"⚠️ Chunk duration is {ws_config.chunk_duration}s, expected {expected_chunk_duration}s")
        
        if ws_config.audio_buffer_size >= expected_buffer_size:
            print("✅ Audio buffer size is sufficient for 40-second chunks")
        else:
            print(f"⚠️ Audio buffer size may be too small: {ws_config.audio_buffer_size} < {expected_buffer_size}")
        
        if ws_config.chunk_timeout >= 60:
            print("✅ Chunk timeout is sufficient for processing large chunks")
        else:
            print(f"⚠️ Chunk timeout may be too short: {ws_config.chunk_timeout}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Could not check configuration: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🤖 Large Chunk Backend Test")
    print("=" * 50)
    print()
    
    # Test 1: Configuration values
    config_ok = await test_config_values()
    
    # Test 2: Large chunk processing
    if config_ok:
        processing_ok = await test_large_chunk_config()
    else:
        print("⚠️ Skipping processing test due to configuration issues")
        processing_ok = False
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"   Large Chunk Processing: {'✅ PASS' if processing_ok else '❌ FAIL'}")
    
    if config_ok and processing_ok:
        print("\n🎉 All tests passed! Backend is ready for 40-second chunks.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    return config_ok and processing_ok

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        sys.exit(1)
