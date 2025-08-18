/**
 * WebSocket Transcription Service
 * Handles real-time audio recording and transcription via WebSocket
 */

export interface TranscriptionMessage {
  type: 'transcript' | 'error' | 'status' | 'session_info';
  text?: string;
  message?: string;
  session_id?: string;
  timestamp?: string;
  is_final?: boolean;
  confidence?: number;
}

export interface TranscriptionConfig {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  bufferSize: number;
}

export class WebSocketTranscriptionService {
  private websocket: WebSocket | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private isRecording = false;
  private sessionId: string | null = null;
  private transcriptBuffer: string[] = [];
  private recordingStartTime: Date | null = null;

  private config: TranscriptionConfig = {
    sampleRate: 16000,
    channels: 1,
    bitsPerSample: 16,
    bufferSize: 16384  // Larger buffer for better performance with big chunks
  };

  private onTranscriptionCallback?: (message: TranscriptionMessage) => void;
  private onErrorCallback?: (error: string) => void;
  private onStatusCallback?: (status: string) => void;

  constructor(
    websocketUrl: string,
    onTranscription?: (message: TranscriptionMessage) => void,
    onError?: (error: string) => void,
    onStatus?: (status: string) => void
  ) {
    this.onTranscriptionCallback = onTranscription;
    this.onErrorCallback = onError;
    this.onStatusCallback = onStatus;
    this.initializeWebSocket(websocketUrl);
  }

  private initializeWebSocket(url: string) {
    try {
      this.websocket = new WebSocket(url);
      
      this.websocket.onopen = () => {
        console.log('🔌 WebSocket connected');
        this.onStatusCallback?.('Connected to transcription service');
      };

      this.websocket.onmessage = (event) => {
        try {
          const message: TranscriptionMessage = JSON.parse(event.data);
          console.log('📝 Transcription received:', message);
          
          if (message.type === 'session_info' && message.session_id) {
            this.sessionId = message.session_id;
            console.log('🆔 Session ID:', this.sessionId);
          }
          
          // Buffer transcript text for file saving
          if (message.type === 'transcript' && message.text && message.is_final) {
            this.transcriptBuffer.push(message.text);
          }
          
          this.onTranscriptionCallback?.(message);
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
          this.onErrorCallback?.('Failed to parse transcription message');
        }
      };

      this.websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.onErrorCallback?.('WebSocket connection error');
      };

      this.websocket.onclose = (event) => {
        console.log('🔌 WebSocket closed:', event.code, event.reason);
        this.onStatusCallback?.('Disconnected from transcription service');
        this.cleanup();
      };
    } catch (error) {
      console.error('❌ Failed to initialize WebSocket:', error);
      this.onErrorCallback?.('Failed to connect to transcription service');
    }
  }

  async startRecording(): Promise<boolean> {
    if (this.isRecording) {
      console.warn('⚠️ Already recording');
      return false;
    }

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // Initialize audio context
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: this.config.sampleRate
      });

      this.source = this.audioContext.createMediaStreamSource(stream);
      
      // Create processor for real-time audio processing
      this.processor = this.audioContext.createScriptProcessor(
        this.config.bufferSize,
        this.config.channels,
        this.config.channels
      );

      this.processor.onaudioprocess = (event) => {
        if (!this.isRecording || !this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
          return;
        }

        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // Convert float32 to int16
        const int16Array = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const sample = Math.max(-1, Math.min(1, inputData[i]));
          int16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }

        // Send audio data to WebSocket
        this.websocket.send(int16Array.buffer);
      };

      // Connect audio nodes
      this.source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      this.isRecording = true;
      this.recordingStartTime = new Date();
      this.transcriptBuffer = []; // Reset transcript buffer
      console.log('🎤 Recording started');
      this.onStatusCallback?.('Recording started');
      
      return true;
    } catch (error) {
      console.error('❌ Failed to start recording:', error);
      this.onErrorCallback?.('Failed to access microphone');
      return false;
    }
  }

  stopRecording() {
    if (!this.isRecording) {
      console.warn('⚠️ Not currently recording');
      return;
    }

    this.isRecording = false;
    
    // Send end session message
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({
        type: 'end_session'
      }));
    }

    // Save transcript to file
    this.saveTranscriptToFile();

    this.cleanup();
    console.log('🛑 Recording stopped');
    this.onStatusCallback?.('Recording stopped');
  }

  private saveTranscriptToFile() {
    if (this.transcriptBuffer.length === 0) {
      console.log('📄 No transcript to save');
      return;
    }

    const transcript = this.transcriptBuffer.join(' ');
    const timestamp = this.recordingStartTime 
      ? this.recordingStartTime.toISOString().replace(/[:.]/g, '-').slice(0, 19)
      : new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    
    const filename = `transcript-${timestamp}.txt`;
    
    // Create file content with metadata
    const fileContent = `Transcript Recording
===================
Date: ${this.recordingStartTime?.toLocaleString() || new Date().toLocaleString()}
Session ID: ${this.sessionId || 'Unknown'}
Duration: ${this.recordingStartTime ? Math.round((Date.now() - this.recordingStartTime.getTime()) / 1000) : 'Unknown'} seconds

Transcript:
-----------
${transcript}

---
Generated by WebSocket Transcription Service
`;

    // Create and download file
    const blob = new Blob([fileContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Clean up the URL object
    URL.revokeObjectURL(url);
    
    console.log(`💾 Transcript saved as: ${filename}`);
    console.log(`📄 Transcript content (${transcript.length} characters):`, transcript);
  }

  private cleanup() {
    // Clean up audio context and nodes
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }

    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }

    this.isRecording = false;
  }

  disconnect() {
    this.stopRecording();
    
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    
    this.sessionId = null;
  }

  isConnected(): boolean {
    return this.websocket?.readyState === WebSocket.OPEN;
  }

  getSessionId(): string | null {
    return this.sessionId;
  }

  getRecordingStatus(): boolean {
    return this.isRecording;
  }
}

// Utility function to create transcription service
export const createTranscriptionService = (
  backendUrl: string = 'ws://localhost:8000',
  onTranscription?: (message: TranscriptionMessage) => void,
  onError?: (error: string) => void,
  onStatus?: (status: string) => void
): WebSocketTranscriptionService => {
  const websocketUrl = `${backendUrl}/ws/transcribe`;
  return new WebSocketTranscriptionService(websocketUrl, onTranscription, onError, onStatus);
};
