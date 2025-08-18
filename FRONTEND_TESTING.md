# Frontend Transcription Testing

This document explains how to automatically test your frontend transcription integration using your MP3 audio files.

## Overview

Instead of manually speaking into the microphone, these test scripts will:

1. **Stream your MP3 file** to the WebSocket transcription service
2. **Simulate real-time audio input** as if someone were speaking
3. **Test the complete flow**: Frontend UI → WebSocket → Backend → Transcription → File Download
4. **Verify the integration** works end-to-end

## Test Methods

### Method 1: Simple Test (Recommended)

**Best for**: Quick testing and debugging

**How it works**:
- You manually interact with the frontend (click Start/Stop)
- Script streams audio file to WebSocket in background
- Easy to see what's happening at each step

**Steps**:
```bash
# 1. Install dependencies
pip install pydub websockets

# 2. Start your backend
python main.py

# 3. Start your frontend (in another terminal)
cd frontend
npm run dev

# 4. Open browser to http://localhost:5173
# 5. Navigate to Conversation page
# 6. Click "Start Recording"

# 7. Run the test script
python test_frontend_simple.py

# 8. Click "Stop Recording" when done
```

### Method 2: Full Automation

**Best for**: Hands-off testing and CI/CD

**How it works**:
- Automatically starts backend and frontend servers
- Controls browser with Playwright (clicks buttons automatically)
- Streams audio file while simulating user interaction
- Completely automated end-to-end test

**Steps**:
```bash
# 1. Install dependencies
pip install -r test_requirements.txt
playwright install chromium

# 2. Run automated test
python test_frontend_automation.py
```

### Method 3: Quick Setup Script

**Easiest way to get started**:
```bash
# Run the setup and test script
./run_frontend_test.sh
```

## Audio Files

The scripts will automatically use these audio files (in order of preference):
1. `audio_for_test.mp3` ✅ (Found in your repo)
2. `harvard.wav` ✅ (Found in your repo)

The scripts automatically convert your audio to the required format:
- **Sample Rate**: 16kHz
- **Channels**: Mono (1 channel)  
- **Format**: 16-bit PCM
- **Streaming**: Large chunks (40 seconds each, ~2.5MB per chunk)
- **Benefits**: Better Whisper context and accuracy with larger chunks

## What to Expect

### Console Output
```
🎵 Testing with audio file: audio_for_test.mp3
🔌 Connecting to: ws://localhost:8001/ws/transcribe
✅ WebSocket connected
🆔 Session started: abc-123-def-456
🎵 Audio: 245760 bytes, 15.4s duration
📡 Streaming 1 large chunks (40s each)...
📏 Chunk size: 1,280,000 bytes (1.2 MB each)
📡 Sent chunk 1/1 (245,760 bytes, 15.4s)
📝 [partial] "Hello" (confidence: 0.85)
📝 [FINAL] "Hello, how are you today?" (confidence: 0.92)
📝 [partial] "I'm doing" (confidence: 0.78)
📝 [FINAL] "I'm doing well, thank you for asking." (confidence: 0.94)
...
🛑 Sent end session message
✅ Test completed!
```

### Browser Behavior
- WebSocket status shows "Connected"
- Recording indicator becomes active
- Console shows live transcription messages
- Toast notifications appear for start/stop
- Transcript file downloads automatically when stopped

### File Output
A transcript file will be downloaded to your Downloads folder:
```
transcript-2024-12-19T15-30-45.txt
```

## Troubleshooting

### Common Issues

**"WebSocket connection failed"**
- Make sure backend is running on port 8001
- Check if WebSocket endpoint is accessible: `ws://localhost:8001/ws/transcribe`

**"Audio file not found"**
- Ensure `audio_for_test.mp3` or `harvard.wav` exists in the repo root
- Check file permissions

**"Frontend server failed to start"**
- Make sure you have Node.js installed
- Run `npm install` in the frontend directory first
- Check if port 5173 is available

**"No transcription output"**
- Check backend logs for errors
- Verify audio file is valid and not corrupted
- Ensure WebSocket transcription service is properly configured

### Debug Mode

For more detailed output, modify the scripts:

```python
# In test_frontend_simple.py, add more logging:
print(f"📡 Raw audio chunk: {len(chunk)} bytes")
print(f"📨 WebSocket message: {message}")
```

### Manual Verification

To manually verify the WebSocket connection:
```bash
# Test WebSocket endpoint directly
wscat -c ws://localhost:8001/ws/transcribe
```

## Advanced Usage

### Custom Audio Files

To test with different audio files:
```python
# Modify the script
tester = SimpleAudioTester("path/to/your/audio.mp3")
```

### Different Formats

Supported audio formats (automatically converted):
- MP3, WAV, M4A, FLAC, OGG
- Any format supported by `pydub`

### Performance Testing

To test with longer audio files:
```python
# Adjust timing in the script
await asyncio.sleep(duration_in_seconds)
```

## Integration with CI/CD

For automated testing in CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Test Frontend Transcription
  run: |
    pip install -r test_requirements.txt
    playwright install chromium
    python test_frontend_automation.py
```

## Next Steps

After successful testing, you can:

1. **Add more test cases** with different audio files
2. **Test error scenarios** (network failures, invalid audio)
3. **Performance testing** with longer recordings
4. **Integration testing** with real microphone input
5. **Add assertions** to verify transcript accuracy

The test scripts provide a solid foundation for automated testing of your transcription pipeline!
