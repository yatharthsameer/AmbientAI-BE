"""
Transcription service using OpenAI Whisper for the Nurse Conversation Processing API.
Handles audio file transcription with timestamp extraction.
"""
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import whisper
import librosa
import soundfile as sf
from pydub import AudioSegment
from loguru import logger

from config import get_settings, ASRSettings


class TranscriptionService:
    """Service for handling audio transcription using OpenAI Whisper."""
    
    def __init__(self, settings: Optional[ASRSettings] = None):
        self.settings = settings or get_settings().asr_settings
        self._model = None
        self._model_name = None
    
    def _load_model(self, model_name: str = None) -> whisper.model.Whisper:
        """Load the Whisper model."""
        model_name = model_name or self.settings.whisper_model
        
        if self._model is None or self._model_name != model_name:
            logger.info(f"Loading Whisper model: {model_name}")
            start_time = time.time()
            
            try:
                self._model = whisper.load_model(
                    model_name,
                    device=self.settings.device
                )
                self._model_name = model_name
                
                load_time = time.time() - start_time
                logger.info(f"Whisper model loaded in {load_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model {model_name}: {e}")
                raise
        
        return self._model
    
    def _preprocess_audio(self, file_path: str) -> str:
        """
        Preprocess audio file for transcription.
        Converts to WAV format and normalizes if needed.
        """
        file_path = Path(file_path)
        
        # If already WAV, check if we need to process it
        if file_path.suffix.lower() == '.wav':
            try:
                # Try to load with librosa to check if it's valid
                audio, sr = librosa.load(str(file_path), sr=None)
                if sr == 16000:  # Whisper prefers 16kHz
                    return str(file_path)
            except Exception:
                pass
        
        # Convert/process the audio file
        logger.info(f"Preprocessing audio file: {file_path}")
        
        try:
            # Load audio with pydub for format flexibility
            audio_segment = AudioSegment.from_file(str(file_path))
            
            # Convert to mono if stereo
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Set sample rate to 16kHz (optimal for Whisper)
            if audio_segment.frame_rate != 16000:
                audio_segment = audio_segment.set_frame_rate(16000)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                audio_segment.export(temp_path, format='wav')
                
            logger.info(f"Audio preprocessed and saved to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    def _get_audio_duration(self, file_path: str) -> float:
        """Get audio file duration in seconds."""
        try:
            audio_info = AudioSegment.from_file(file_path)
            return len(audio_info) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0.0
    
    def _clean_temp_file(self, temp_path: str):
        """Clean up temporary files."""
        try:
            if os.path.exists(temp_path) and temp_path.startswith(tempfile.gettempdir()):
                os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Could not clean temp file {temp_path}: {e}")
    
    def transcribe_audio(
        self,
        file_path: str,
        model_name: str = None,
        language: str = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        best_of: int = 1,
        patience: float = 1.0,
        beam_size: int = 5
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper.
        
        Args:
            file_path: Path to the audio file
            model_name: Whisper model to use (base, small, medium, large)
            language: Language code for transcription (auto-detect if None)
            task: Either "transcribe" or "translate"
            temperature: Temperature for sampling (0.0 = deterministic)
            best_of: Number of candidates to consider
            patience: Patience for beam search
            beam_size: Beam size for beam search (required when patience > 0)
            
        Returns:
            Dictionary containing transcription results with timestamps
        """
        logger.info(f"Starting transcription for: {file_path}")
        start_time = time.time()
        temp_path = None
        
        try:
            # Get audio duration for progress tracking
            duration = self._get_audio_duration(file_path)
            
            # Preprocess audio
            temp_path = self._preprocess_audio(file_path)
            
            # Load model
            model = self._load_model(model_name)
            
            # Set transcription options
            options = {
                "task": task,
                "language": language,
                "temperature": temperature,
                "best_of": best_of,
                "verbose": False,  # Don't print progress to stdout
            }
            
            # Add beam search options only when needed
            if patience > 0:
                options.update({
                    "patience": patience,
                    "beam_size": beam_size
                })
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            logger.info(f"Transcribing with options: {options}")
            
            # Perform transcription
            result = model.transcribe(temp_path, **options)
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            
            # Process segments to add additional metadata
            processed_segments = []
            for i, segment in enumerate(result["segments"]):
                processed_segment = {
                    "id": i,
                    "seek": segment.get("seek", 0),
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "tokens": segment.get("tokens", []),
                    "temperature": segment.get("temperature", temperature),
                    "avg_logprob": segment.get("avg_logprob", 0.0),
                    "compression_ratio": segment.get("compression_ratio", 0.0),
                    "no_speech_prob": segment.get("no_speech_prob", 0.0),
                    "words": segment.get("words", [])  # Word-level timestamps if available
                }
                processed_segments.append(processed_segment)
            
            # Calculate confidence score based on average log probability
            avg_logprobs = [seg.get("avg_logprob", 0.0) for seg in result["segments"]]
            confidence_score = None
            if avg_logprobs:
                # Convert log probabilities to a 0-1 confidence score
                avg_logprob = sum(avg_logprobs) / len(avg_logprobs)
                confidence_score = min(1.0, max(0.0, (avg_logprob + 5.0) / 5.0))  # Normalize roughly
            
            transcription_result = {
                "text": result["text"].strip(),
                "segments": processed_segments,
                "language": result.get("language"),
                "model_used": self._model_name,
                "processing_time_seconds": processing_time,
                "duration_seconds": duration,
                "confidence_score": confidence_score,
                "options": options
            }
            
            # Format confidence score for logging
            confidence_display = f"{confidence_score:.3f}" if confidence_score is not None else "N/A"
            
            logger.info(
                f"Transcription successful - Language: {result.get('language')}, "
                f"Segments: {len(processed_segments)}, "
                f"Confidence: {confidence_display}"
            )
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
        
        finally:
            # Clean up temporary files
            if temp_path:
                self._clean_temp_file(temp_path)
    
    def transcribe_text_only(self, text: str) -> Dict[str, Any]:
        """
        Process text-only input (no transcription needed).
        Returns a formatted response similar to audio transcription.
        """
        logger.info("Processing text-only input")
        
        # Create segments by splitting on sentence boundaries
        sentences = self._split_into_sentences(text)
        
        segments = []
        current_time = 0.0
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Estimate duration based on character count (rough approximation)
                estimated_duration = len(sentence) * 0.05  # ~50ms per character
                
                segment = {
                    "id": i,
                    "seek": 0,
                    "start": current_time,
                    "end": current_time + estimated_duration,
                    "text": sentence.strip(),
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                    "words": []
                }
                segments.append(segment)
                current_time += estimated_duration
        
        return {
            "text": text.strip(),
            "segments": segments,
            "language": "unknown",
            "model_used": "text_only",
            "processing_time_seconds": 0.1,
            "duration_seconds": current_time,
            "confidence_score": 1.0,
            "options": {"task": "text_only"}
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for segment creation."""
        import re
        
        # Simple sentence splitting - you could use more sophisticated NLP here
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return ["mp3", "wav", "m4a", "flac", "ogg", "wma", "aac"]
    
    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if audio file is supported and readable.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, "File does not exist"
            
            if file_path.stat().st_size == 0:
                return False, "File is empty"
            
            # Check file extension
            extension = file_path.suffix.lower().lstrip('.')
            if extension not in self.get_supported_formats():
                return False, f"Unsupported file format: {extension}"
            
            # Try to load a small portion of the audio
            try:
                AudioSegment.from_file(str(file_path))[:1000]  # First second
            except Exception as e:
                return False, f"Cannot read audio file: {e}"
            
            return True, ""
            
        except Exception as e:
            return False, f"File validation error: {e}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self._model_name,
            "device": self.settings.device,
            "is_loaded": self._model is not None,
            "supported_languages": [
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", 
                "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
            ],
            "available_models": ["tiny", "base", "small", "medium", "large"],
            "model_sizes": {
                "tiny": "~40 MB",
                "base": "~150 MB", 
                "small": "~500 MB",
                "medium": "~1.5 GB",
                "large": "~3 GB"
            }
        }