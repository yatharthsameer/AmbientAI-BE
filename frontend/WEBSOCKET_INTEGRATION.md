# WebSocket Transcription Integration

This document explains how to use the WebSocket transcription integration with your React frontend.

## Overview

The frontend now integrates with the backend's WebSocket transcription service to provide real-time speech-to-text functionality. When you click the recording button, it will:

1. Connect to the WebSocket transcription service
2. Start recording audio from your microphone
3. Stream audio data to the backend for real-time transcription
4. Display transcription results in the browser console
5. Automatically save the complete transcript to a downloadable text file when recording stops

## How to Use

### 1. Start the Backend Server

Make sure your backend server is running on `localhost:8000`:

```bash
# From the project root
python main.py
# or
python minimal_server.py
```

### 2. Start the Frontend Development Server

```bash
cd frontend
npm run dev
# or
bun dev
```

### 3. Use the Recording Feature

1. Navigate to the Conversation page (default route `/`)
2. Click the "Start" button to begin recording
3. Allow microphone access when prompted
4. Speak into your microphone
5. Open browser console (F12) to see live transcription output
6. Click "Stop" to end the recording session
7. The transcript file will automatically download to your Downloads folder

## Output

### Console Messages
You'll see several types of console messages:

- `🔌 WebSocket connected` - Connection established
- `🎯 Live Transcription:` - Raw transcription messages
- `📝 Transcribed: "text"` - Individual transcription chunks
- `💾 Transcript saved as: filename.txt` - File save confirmation
- `❌ WebSocket Error:` - Any connection or transcription errors

### Transcript File
When you stop recording, a text file will be automatically downloaded with:

- **Filename**: `transcript-YYYY-MM-DDTHH-MM-SS.txt` (timestamped)
- **Location**: Your browser's default Downloads folder
- **Content**: 
  - Recording metadata (date, session ID, duration)
  - Complete transcript text
  - Generated timestamp

## Technical Details

### WebSocket Connection

- **URL**: `ws://localhost:8000/ws/transcribe`
- **Protocol**: Binary audio data + JSON control messages
- **Audio Format**: PCM 16-bit, 16kHz, mono

### Audio Processing

- **Sample Rate**: 16kHz (configurable)
- **Channels**: 1 (mono)
- **Buffer Size**: 16384 samples (optimized for large chunks)
- **Format**: 16-bit signed integers
- **Chunk Size**: 40 seconds (~2.5MB per chunk)
- **Benefits**: Better Whisper context and transcription accuracy

### Features

- ✅ Real-time audio streaming
- ✅ Live transcription display in console
- ✅ Automatic transcript file download
- ✅ Timestamped file naming
- ✅ WebSocket connection status monitoring
- ✅ Error handling and user feedback
- ✅ Session management
- ⚠️ Pause/Resume (UI only - WebSocket doesn't support pause)

## Troubleshooting

### Common Issues

1. **"Failed to access microphone"**
   - Ensure you allow microphone permissions
   - Check if another application is using the microphone
   - Try refreshing the page and allowing permissions again

2. **"WebSocket connection error"**
   - Verify the backend server is running on port 8000
   - Check if the WebSocket endpoint `/ws/transcribe` is accessible
   - Look for CORS issues in the browser console

3. **No transcription output**
   - Check browser console for error messages
   - Verify audio is being captured (check browser's microphone indicator)
   - Ensure the backend transcription service is properly configured

### Browser Compatibility

- Chrome/Chromium: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support (may require HTTPS in production)
- Edge: ✅ Full support

### Security Notes

- Microphone access requires user permission
- WebSocket connections may require HTTPS in production
- Audio data is streamed in real-time (not stored locally)

## Next Steps

To enhance the integration, you could:

1. Display transcription in the UI (not just console)
2. Add transcript export functionality
3. Implement speaker identification
4. Add confidence score visualization
5. Store transcripts in local storage or backend database
