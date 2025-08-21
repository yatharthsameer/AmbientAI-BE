"""
Transcription service using OpenAI Whisper for the Nurse Conversation Processing API.
Handles audio file transcription with timestamp extraction.
"""
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from loguru import logger
# Heavy imports moved to lazy loading inside functions:
# from faster_whisper import WhisperModel as FWModel
# import librosa
# import soundfile as sf
# from pydub import AudioSegment
# import soxr

from config import get_settings, ASRSettings


class TranscriptionService:
    """Service for handling audio transcription using faster-whisper."""

    def __init__(self, settings: Optional[ASRSettings] = None):
        self.settings = settings or get_settings().asr_settings
        self._model_name = None
        self._fw_model: Optional[Any] = None  # FWModel loaded lazily
        self._use_faster: bool = True  # prefer faster-whisper when available
        # Conservative threading to prevent macOS memory kills and CPU oversubscription
        try:
            import os as _os

            _os.environ.setdefault("OMP_NUM_THREADS", "1")
            _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            _os.environ.setdefault("MKL_NUM_THREADS", "1")
            _os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
            _os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        except Exception:
            pass

    def _load_model(self, model_name: str = None) -> Any:
        """Load the faster-whisper model only (no openai-whisper fallback)."""
        # Lazy import here
        from faster_whisper import WhisperModel as FWModel

        model_name = model_name or self.settings.whisper_model

        # Load faster-whisper model (robust compute_type selection with fallbacks)
        if self._fw_model is None or self._model_name != model_name:
            logger.info(f"Loading faster-whisper model: {model_name}")
            start_time = time.time()
            try:
                # Compute type candidates by device - force int8 on CPU to prevent memory bloat
                if self.settings.device == "cpu":
                    compute_candidates = [
                        "int8"
                    ]  # do not fall back silently to float32 on laptops
                else:
                    compute_candidates = ["float16", "float32"]
                # Use persistent download directory if set via env
                _cache_dir = os.getenv(
                    "FASTER_WHISPER_CACHE_DIR", "./models/faster-whisper"
                )
                os.makedirs(_cache_dir, exist_ok=True)
                last_err = None
                # Limit workers/threads to keep memory stable on laptops
                cpu_threads = (
                    max(1, min(os.cpu_count() or 2, 4))
                    if self.settings.device == "cpu"
                    else None
                )
                num_workers = 1 if self.settings.device == "cpu" else 2
                for ct in compute_candidates:
                    try:
                        kwargs = dict(
                            device=self.settings.device,
                            compute_type=ct,
                            download_root=_cache_dir,
                        )
                        # Only pass CPU-specific tuning when on CPU
                        if self.settings.device == "cpu":
                            kwargs.update(
                                {
                                    "cpu_threads": cpu_threads,
                                    "num_workers": num_workers,
                                }
                            )
                        self._fw_model = FWModel(model_name, **kwargs)
                        self._model_name = model_name
                        load_time = time.time() - start_time
                        logger.info(
                            f"faster-whisper model loaded in {load_time:.2f} seconds (compute_type={ct}, cpu_threads={cpu_threads}, num_workers={num_workers})"
                        )
                        break
                    except Exception as e:
                        last_err = e
                        logger.warning(
                            f"Compute type '{ct}' failed: {e}. Trying next candidate…"
                        )
                        self._fw_model = None
                if self._fw_model is None:
                    raise last_err or RuntimeError(
                        "Could not load faster-whisper model with any compute_type"
                    )
            except Exception as e:
                logger.error(f"Failed to load faster-whisper {model_name}: {e}")
                raise

        return self._fw_model

    def _resample_audio_hq(
        self, audio_data: np.ndarray, orig_sr: int, target_sr: int = 16000
    ) -> np.ndarray:
        """
        High-quality audio resampling using soxr.

        Args:
            audio_data: Input audio as numpy array
            orig_sr: Original sample rate
            target_sr: Target sample rate (default 16kHz for Whisper)

        Returns:
            Resampled audio as numpy array
        """
        if orig_sr == target_sr:
            return audio_data

        # Lazy import
        import soxr

        # Use soxr for high-quality resampling (VHQ quality)
        resampled = soxr.resample(
            audio_data, orig_sr, target_sr, quality="VHQ"  # Very High Quality
        )
        return resampled.astype(np.float32)

    def _preprocess_audio_to_array(self, file_path: str) -> np.ndarray:
        """
        Preprocess audio file directly to numpy array for faster-whisper.

        Args:
            file_path: Path to audio file

        Returns:
            Audio data as float32 numpy array at 16kHz mono
        """
        try:
            # Lazy import
            import librosa

            # Load audio with librosa for better quality control
            audio_data, orig_sr = librosa.load(str(file_path), sr=None, mono=True)

            # High-quality resample to 16kHz if needed
            if orig_sr != 16000:
                audio_data = self._resample_audio_hq(audio_data, orig_sr, 16000)

            return audio_data.astype(np.float32)

        except Exception as e:
            logger.error(f"Audio preprocessing to array failed: {e}")
            raise

    def _preprocess_audio(self, file_path: str) -> str:
        """
        Preprocess audio file for transcription.
        Converts to WAV format and normalizes if needed.
        """
        file_path = Path(file_path)

        # If already WAV, check if we need to process it
        if file_path.suffix.lower() == '.wav':
            try:
                # Lazy import
                import librosa

                # Try to load with librosa to check if it's valid
                audio, sr = librosa.load(str(file_path), sr=None)
                if sr == 16000:  # Whisper prefers 16kHz
                    return str(file_path)
            except Exception:
                pass

        # Convert/process the audio file
        logger.info(f"Preprocessing audio file: {file_path}")

        try:
            # Lazy imports
            import librosa
            import soundfile as sf

            # Load audio with librosa for better quality control
            audio_data, orig_sr = librosa.load(str(file_path), sr=None, mono=True)

            # High-quality resample to 16kHz if needed
            if orig_sr != 16000:
                audio_data = self._resample_audio_hq(audio_data, orig_sr, 16000)

            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_data, 16000)

            logger.info(f"Audio preprocessed and saved to: {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise

    def _get_audio_duration(self, file_path: str) -> float:
        """Get audio file duration in seconds."""
        try:
            # Lazy import
            from pydub import AudioSegment

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

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sr: int = 16000,
        model_name: str = None,
        language: str = "en",
        task: str = "transcribe",
        temperature: Any = 0.0,
        best_of: int = 1,
        patience: float = 1.0,
        beam_size: int = 2,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: Optional[bool] = False,
        word_timestamps: bool = True,
        logprob_threshold: Optional[float] = -0.5,
        no_speech_threshold: Optional[float] = 0.6,
        compression_ratio_threshold: Optional[float] = 2.3,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio from numpy array directly (no disk I/O) - optimized for live streaming."""
        start_time = time.time()
        logger.info(
            f"Starting array transcription with model: {model_name or self.settings.whisper_model}"
        )

        try:
            # Resample if needed
            if sr != 16000:
                audio_array = self._resample_audio_hq(audio_array, sr, 16000)
                sr = 16000

            model = self._load_model(model_name)
            if model is None:
                raise RuntimeError("Faster-whisper model not loaded")

            fw_params = {
                "language": language,
                "task": task,
                "beam_size": beam_size,
                "best_of": best_of,
                "vad_filter": vad_filter,
                "vad_parameters": vad_parameters
                or {"min_silence_duration_ms": 600, "speech_pad_ms": 150},
                "temperature": temperature,
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": condition_on_previous_text,
                "word_timestamps": word_timestamps,
            }
            if logprob_threshold is not None:
                fw_params["log_prob_threshold"] = logprob_threshold
            if no_speech_threshold is not None:
                fw_params["no_speech_threshold"] = no_speech_threshold
            if compression_ratio_threshold is not None:
                fw_params["compression_ratio_threshold"] = compression_ratio_threshold

            logger.info(f"Transcribing array with options: {fw_params}")
            segments, info = model.transcribe(audio_array, **fw_params)

            seg_list, full_text_parts = [], []
            for i, seg in enumerate(segments):
                words = []
                if hasattr(seg, "words") and seg.words:
                    words = [
                        {
                            "word": w.word,
                            "start": float(w.start),
                            "end": float(w.end),
                            "probability": float(getattr(w, "probability", 1.0)),
                        }
                        for w in seg.words
                    ]
                seg_list.append(
                    {
                        "id": i,
                        "seek": 0,
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text.strip(),
                        "tokens": [],
                        "temperature": temperature,
                        "avg_logprob": float(getattr(seg, "avg_logprob", 0.0)),
                        "compression_ratio": float(
                            getattr(seg, "compression_ratio", 0.0)
                        ),
                        "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0)),
                        "words": words,
                    }
                )
                full_text_parts.append(seg.text.strip())

            avg_logprobs = [s["avg_logprob"] for s in seg_list if s is not None]
            confidence_score = None
            if avg_logprobs:
                avg_lp = sum(avg_logprobs) / len(avg_logprobs)
                confidence_score = min(1.0, max(0.0, (avg_lp + 5.0) / 5.0))

            result = {
                "text": " ".join(full_text_parts).strip(),
                "segments": seg_list,
                "language": info.language or language,
                "model_used": self._model_name,
                "confidence_score": confidence_score,
                "options": {
                    "language": language,
                    "beam_size": beam_size,
                    "best_of": best_of,
                    "temperature": temperature,
                    "word_timestamps": word_timestamps,
                    "vad_filter": vad_filter,
                },
            }
            # Optional safe post-process
            result["text"] = self._safe_post_process(result["text"])
            for seg in result["segments"]:
                seg["text"] = self._safe_post_process(seg["text"])

            transcription_time = time.time() - start_time
            logger.info(
                f"Array transcription completed in {transcription_time:.2f} seconds"
            )
            confidence_str = (
                f"{confidence_score:.3f}" if confidence_score is not None else "N/A"
            )
            logger.info(
                f"Transcription successful - Language: {result['language']}, Segments: {len(seg_list)}, Confidence: {confidence_str}"
            )

            return result

        except Exception as e:
            logger.error(f"Array transcription failed: {e}")
            raise

    def transcribe_audio(
        self,
        file_path: str,
        model_name: str = None,
        language: str = None,
        task: str = "transcribe",
        temperature: Any = 0.0,
        best_of: int = 1,
        patience: float = 1.0,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: Optional[bool] = True,
        word_timestamps: Optional[bool] = False,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        compression_ratio_threshold: Optional[float] = None,
        vad_filter: Optional[bool] = None,
        vad_parameters: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Use for batch/offline files; **live path must call `transcribe_array`** to avoid disk I/O.

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

            # Load model
            model = self._load_model(model_name)

            # Preprocess audio to numpy array for faster-whisper
            if model is None:
                raise RuntimeError("Faster-whisper model not loaded")
            audio_array = self._preprocess_audio_to_array(file_path)
            # Validate audio array
            if len(audio_array) == 0:
                raise ValueError("Audio file contains no valid audio data")

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

            # Optional enhancements
            optional = {
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": condition_on_previous_text,
                "word_timestamps": word_timestamps,
                "logprob_threshold": logprob_threshold,
                "no_speech_threshold": no_speech_threshold,
                "compression_ratio_threshold": compression_ratio_threshold,
                "vad_filter": vad_filter,
                "vad_parameters": vad_parameters,
            }
            options.update({k: v for k, v in optional.items() if v is not None})
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            logger.info(f"Transcribing with options: {options}")

            # Perform transcription using faster-whisper only
            logger.info(
                f"Using faster-whisper for transcription with model: {self._model_name}"
            )
            # faster-whisper specific parameters
            fw_params = {
                "language": options.get("language"),
                "task": options.get("task", "transcribe"),
                "beam_size": options.get("beam_size", 5),
                "best_of": options.get("best_of", 1),
                "vad_filter": options.get("vad_filter", True),
                "vad_parameters": options.get(
                    "vad_parameters",
                    {"min_silence_duration_ms": 600, "speech_pad_ms": 150},
                ),
                "temperature": options.get("temperature", 0.0),
                "initial_prompt": options.get("initial_prompt"),
                "condition_on_previous_text": options.get(
                    "condition_on_previous_text", True
                ),
                "word_timestamps": options.get("word_timestamps", False),
            }

            # Add optional parameters only if they're not None
            if options.get("logprob_threshold") is not None:
                # faster-whisper uses log_prob_threshold
                fw_params["log_prob_threshold"] = options.get("logprob_threshold")
            if options.get("no_speech_threshold") is not None:
                fw_params["no_speech_threshold"] = options.get("no_speech_threshold")
            if options.get("compression_ratio_threshold") is not None:
                fw_params["compression_ratio_threshold"] = options.get(
                    "compression_ratio_threshold"
                )

            segments, info = model.transcribe(audio_array, **fw_params)
            seg_list = []
            full_text_parts = []
            for i, seg in enumerate(segments):
                # Extract word-level timestamps if available
                words = []
                if hasattr(seg, "words") and seg.words:
                    words = [
                        {
                            "word": word.word,
                            "start": float(word.start),
                            "end": float(word.end),
                            "probability": (
                                float(word.probability)
                                if hasattr(word, "probability")
                                else 1.0
                            ),
                        }
                        for word in seg.words
                    ]

                seg_list.append(
                    {
                        "id": i,
                        "seek": 0,
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text.strip(),
                        "tokens": [],
                        "temperature": options.get("temperature", 0.0),
                        "avg_logprob": (
                            float(seg.avg_logprob)
                            if hasattr(seg, "avg_logprob")
                            else 0.0
                        ),
                        "compression_ratio": (
                            float(seg.compression_ratio)
                            if hasattr(seg, "compression_ratio")
                            else 0.0
                        ),
                        "no_speech_prob": (
                            float(seg.no_speech_prob)
                            if hasattr(seg, "no_speech_prob")
                            else 0.0
                        ),
                        "words": words,
                    }
                )
                full_text_parts.append(seg.text.strip())
            result = {
                "text": " ".join(full_text_parts).strip(),
                "segments": seg_list,
                "language": info.language or options.get("language"),
                "model_used": self._model_name,
            }

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

            # Apply safer post-processing
            transcription_result["text"] = self._safe_post_process(
                transcription_result["text"]
            )
            for segment in transcription_result["segments"]:
                segment["text"] = self._safe_post_process(segment["text"])

            return transcription_result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

        finally:
            # No temp files are created for faster-whisper path
            pass

    def _safe_post_process(self, text: str) -> str:
        """
        Apply safer post-processing with scoped rules to avoid corrupting content.

        Args:
            text: Raw transcription text

        Returns:
            Cleaned text with scoped domain-specific corrections
        """
        if not text:
            return text

        import re

        # Only replace " over " when flanked by digits (blood pressure, fractions)
        text = re.sub(r"(\d+)\s+over\s+(\d+)", r"\1/\2", text)

        # Fix common medical term transcription errors (case-insensitive)
        medical_corrections = {
            r"\blisinopril\b": "lisinopril",  # Common misspellings
            r"\bmetformin\b": "metformin",
            r"\batorvastatin\b": "atorvastatin",
            r"\bfurosemide\b": "furosemide",
            r"\blevothyroxine\b": "levothyroxine",
            r"\binsulin glargine\b": "insulin glargine",
            r"\bwarfarin\b": "warfarin",
            r"\bapixaban\b": "apixaban",
        }

        for pattern, replacement in medical_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

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
                # Lazy import
                from pydub import AudioSegment

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
            "is_loaded": self._fw_model is not None,
            "supported_languages": [
                "en",
                "zh",
                "de",
                "es",
                "ru",
                "ko",
                "fr",
                "ja",
                "pt",
                "tr",
                "pl",
                "ca",
                "nl",
                "ar",
                "sv",
                "it",
                "id",
                "hi",
                "fi",
                "vi",
                "he",
                "uk",
                "el",
                "ms",
                "cs",
                "ro",
                "da",
                "hu",
                "ta",
                "no",
                "th",
                "ur",
                "hr",
                "bg",
                "lt",
                "la",
                "mi",
                "ml",
                "cy",
                "sk",
                "te",
                "fa",
                "lv",
                "bn",
                "sr",
                "az",
                "sl",
                "kn",
                "et",
                "mk",
                "br",
                "eu",
                "is",
                "hy",
                "ne",
                "mn",
                "bs",
                "kk",
                "sq",
                "sw",
                "gl",
                "mr",
                "pa",
                "si",
                "km",
                "sn",
                "yo",
                "so",
                "af",
                "oc",
                "ka",
                "be",
                "tg",
                "sd",
                "gu",
                "am",
                "yi",
                "lo",
                "uz",
                "fo",
                "ht",
                "ps",
                "tk",
                "nn",
                "mt",
                "sa",
                "lb",
                "my",
                "bo",
                "tl",
                "mg",
                "as",
                "tt",
                "haw",
                "ln",
                "ha",
                "ba",
                "jw",
                "su",
            ],
            "available_models": ["tiny", "base", "small", "medium", "large"],
            "model_sizes": {
                "tiny": "~40 MB",
                "base": "~150 MB",
                "small": "~500 MB",
                "medium": "~1.5 GB",
                "large": "~3 GB",
            },
        }
