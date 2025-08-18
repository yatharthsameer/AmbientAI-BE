import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import QuestionCard from "@/components/QuestionCard";
import {
  Mic,
  Pause,
  Play,
  CheckCircle,
  Square,
  Loader2
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { 
  WebSocketTranscriptionService, 
  TranscriptionMessage, 
  createTranscriptionService 
} from "@/services/websocketTranscription";

const mockQuestions = [
  "Did you assess the patient's pain level using a standardized scale?",
  "Have you documented the patient's current medications and any changes?",
  "Did you evaluate the patient's ability to perform activities of daily living?",
  "Have you assessed the patient's home safety and fall risk factors?",
  "Did you check the patient's vital signs and document any concerns?"
];

const ConversationPage = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [questions, setQuestions] = useState<string[]>([]);
  const [isEnding, setIsEnding] = useState(false);
  const [transcriptionStatus, setTranscriptionStatus] = useState<string>('Disconnected');
  const [fullTranscript, setFullTranscript] = useState<string>('');
  const { toast } = useToast();
  const navigate = useNavigate();
  
  // WebSocket transcription service ref
  const transcriptionServiceRef = useRef<WebSocketTranscriptionService | null>(null);

const WAVE_BARS = 24;
const [levels, setLevels] = useState<number[]>(
  Array.from({ length: WAVE_BARS }, () => 8)
);

useEffect(() => {
  let id: number | undefined;
  if (isRecording && !isPaused) {
    id = window.setInterval(() => {
      setLevels(
        Array.from({ length: WAVE_BARS }, () => Math.floor(Math.random() * 56) + 8)
      );
    }, 180) as unknown as number;
  } else {
    setLevels(Array.from({ length: WAVE_BARS }, () => 8));
  }
  return () => {
    if (id) window.clearInterval(id);
  };
}, [isRecording, isPaused]);

useEffect(() => {
  document.title = "Record Conversation | HealthScribe";
}, []);

// Initialize WebSocket transcription service
useEffect(() => {
  const handleTranscription = (message: TranscriptionMessage) => {
    console.log('🎯 Live Transcription:', message);
    
    if (message.type === 'transcript' && message.text) {
      // Log to console as requested
      console.log(`📝 Transcribed: "${message.text}" (Final: ${message.is_final}, Confidence: ${message.confidence})`);
      
      // Update full transcript
      if (message.is_final) {
        setFullTranscript(prev => prev + ' ' + message.text);
      }
    } else if (message.type === 'error') {
      console.error('❌ Transcription Error:', message.message);
    }
  };

  const handleError = (error: string) => {
    console.error('❌ WebSocket Error:', error);
    setTranscriptionStatus('Error: ' + error);
    toast({
      title: "Transcription Error",
      description: error,
      variant: "destructive",
    });
  };

  const handleStatus = (status: string) => {
    console.log('📊 Status:', status);
    setTranscriptionStatus(status);
  };

  // Create transcription service
  transcriptionServiceRef.current = createTranscriptionService(
    'ws://localhost:8001', // Backend WebSocket URL
    handleTranscription,
    handleError,
    handleStatus
  );

  // Cleanup on unmount
  return () => {
    if (transcriptionServiceRef.current) {
      transcriptionServiceRef.current.disconnect();
    }
  };
}, [toast]);

  const handleStartRecording = async () => {
    if (!transcriptionServiceRef.current) {
      toast({
        title: "Error",
        description: "Transcription service not initialized",
        variant: "destructive",
      });
      return;
    }

    console.log('🎤 Starting recording...');
    const success = await transcriptionServiceRef.current.startRecording();
    
    if (success) {
      setIsRecording(true);
      setIsPaused(false);
      setFullTranscript(''); // Reset transcript
      toast({
        title: "Recording Started",
        description: "Ambient transcription is now active",
      });
    } else {
      toast({
        title: "Recording Failed",
        description: "Could not start audio recording",
        variant: "destructive",
      });
    }
  };

  const handleStopRecording = () => {
    if (transcriptionServiceRef.current) {
      console.log('🛑 Stopping recording...');
      transcriptionServiceRef.current.stopRecording();
    }
    
    setIsRecording(false);
    setIsPaused(false);
    toast({
      title: "Recording Stopped",
      description: "Transcript will be downloaded automatically",
    });
    
    // Show additional toast for file download
    setTimeout(() => {
      toast({
        title: "Transcript Saved",
        description: "Check your Downloads folder for the transcript file",
      });
    }, 1000);
  };

  const handlePause = () => {
    // Note: WebSocket transcription doesn't support pause/resume
    // This is just UI state for now
    setIsPaused(true);
    toast({
      title: "Paused",
      description: "Transcription paused (UI only)",
    });
  };

  const handleResume = () => {
    setIsPaused(false);
    toast({
      title: "Resumed",
      description: "Transcription resumed",
    });
  };

  const handleValidate = async () => {
    setIsValidating(true);
    
    // Simulate API call
    setTimeout(() => {
      const randomQuestions = mockQuestions
        .sort(() => 0.5 - Math.random())
        .slice(0, Math.floor(Math.random() * 3) + 1);
      
      setQuestions(randomQuestions);
      setIsValidating(false);
      
      toast({
        title: "Validation Complete",
        description: `Found ${randomQuestions.length} missing assessment questions`,
      });
    }, 2000);
  };

  const handleEndConversation = async () => {
    setIsEnding(true);
    
    // Simulate processing
    setTimeout(() => {
      navigate("/filling");
    }, 3000);
  };

  const dismissQuestion = (index: number) => {
    setQuestions(prev => prev.filter((_, i) => i !== index));
  };

  if (isEnding) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="p-8 text-center">
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">Processing Conversation</h2>
          <p className="text-muted-foreground">
            Transcribing audio and preparing OASIS form...
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Decorative Background */}
      <div aria-hidden className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute left-1/2 -top-40 -translate-x-1/2 w-[560px] h-[560px] rounded-full bg-primary/15 blur-3xl" />
        <div className="absolute -right-24 -bottom-24 w-[420px] h-[420px] rounded-full bg-primary/10 blur-3xl" />
      </div>

      {/* Question Cards */}
      {questions.map((question, index) => (
        <QuestionCard
          key={index}
          question={question}
          onDismiss={() => dismissQuestion(index)}
        />
      ))}

      <div className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto text-center">
          <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            Record Conversation
          </h1>
          <p className="text-muted-foreground mb-12">
            Start recording to enable ambient transcription and real‑time prompts.
          </p>

          {/* Main Recording Control */}
          <div className="mb-12">
            <div className="relative inline-flex">
              {isRecording && !isPaused && (
                <span
                  aria-hidden
                  className="absolute -inset-6 rounded-full bg-gradient-to-br from-primary/30 to-primary/5 blur-2xl animate-pulse"
                />
              )}
              <Button
                aria-label={!isRecording ? "Start recording" : isPaused ? "Resume recording" : "Pause recording"}
                size="lg"
                onClick={!isRecording ? handleStartRecording : isPaused ? handleResume : handlePause}
                className={`
                  hover-scale h-32 w-32 rounded-full text-lg font-semibold
                  ring-4 ring-primary/25 ring-offset-2 ring-offset-background
                  ${!isRecording 
                    ? 'bg-primary hover:bg-primary/90' 
                    : isPaused 
                      ? 'bg-secondary hover:bg-secondary/90' 
                      : 'bg-destructive hover:bg-destructive/90 animate-recording'
                  }
                  transition-all duration-300
                `}
              >
                <div className="flex flex-col items-center space-y-2">
                  {!isRecording ? (
                    <>
                      <Mic className="h-8 w-8" />
                      <span className="text-sm">Start</span>
                    </>
                  ) : isPaused ? (
                    <>
                      <Play className="h-8 w-8" />
                      <span className="text-sm">Resume</span>
                    </>
                  ) : (
                    <>
                      <Pause className="h-8 w-8" />
                      <span className="text-sm">Pause</span>
                    </>
                  )}
                </div>
              </Button>
            </div>

            {isRecording && !isPaused && (
              <>
                <p className="text-primary font-medium mt-4 pulse">
                  Listening...
                </p>
                {/* Simple Waveform */}
                <div className="mt-6 flex items-end justify-center gap-1 h-20 animate-fade-in">
                  {levels.map((h, i) => (
                    <span
                      key={i}
                      className="w-1.5 rounded-t bg-primary/70"
                      style={{ height: `${h}px` }}
                    />
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button
              variant="outline"
              size="lg"
              onClick={handleValidate}
              disabled={!isRecording || isValidating}
              className="flex items-center space-x-2 hover-scale"
            >
              {isValidating ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <CheckCircle className="h-5 w-5" />
              )}
              <span>
                {isValidating ? "Validating..." : "Validate Conversation"}
              </span>
            </Button>

            <Button
              variant="secondary"
              size="lg"
              onClick={handleEndConversation}
              disabled={!isRecording}
              className="flex items-center space-x-2 hover-scale"
            >
              <Square className="h-5 w-5" />
              <span>End Conversation</span>
            </Button>
          </div>

          {/* Status Card */}
          {isRecording && (
            <Card className="mt-8 p-6 border-primary/20 animate-fade-in">
              <div className="flex items-center justify-center space-x-3">
                <div className={`h-3 w-3 rounded-full ${isPaused ? 'bg-muted-foreground' : 'bg-primary'} animate-pulse`}></div>
                <p className="text-foreground font-medium">
                  {isPaused ? 'Transcription paused' : 'Ambient transcription active'}
                </p>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                {isPaused
                  ? 'Transcription is paused. Tap resume to continue.'
                  : 'Speak naturally with your patient. The system will capture conversation and identify missing assessment questions.'}
              </p>
              <div className="mt-3 pt-3 border-t border-border/50">
                <p className="text-xs text-muted-foreground">
                  WebSocket Status: <span className={transcriptionStatus.includes('Error') ? 'text-destructive' : 'text-primary'}>{transcriptionStatus}</span>
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Transcript will be automatically saved to file when recording stops
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConversationPage;