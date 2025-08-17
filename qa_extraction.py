"""
Q&A extraction service using multiple AI providers for the Nurse Conversation Processing API.
Supports both traditional Hugging Face models and modern AI services (Gemini, OpenAI).
"""
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    pipeline,
    QuestionAnsweringPipeline
)
import torch
from loguru import logger

from config import get_settings, QASettings, PREDEFINED_QUESTIONS
from services.qa_service import QAService, AIProvider
from services.gemini_service import GeminiService
from services.openai_service import OpenAIService
from services.hybrid_extraction_service import HybridExtractionService
from services.distilbert_service import DistilBERTService
from services.rag_service import SimpleRAGService
from services.final_verification_service import FinalVerificationService


class QAExtractionService:
    """Service for extracting answers to questions from conversation transcripts."""
    
    def __init__(self, use_ai_service: bool = True):
        """Initialize the Q&A extraction service."""
        self.use_ai_service = use_ai_service
        self.logger = logger
        
        # Initialize AI services
        self.gemini_service = None
        self.openai_service = None
        self.distilbert_service = None
        self.hybrid_service = None
        
        try:
            # Initialize Gemini service
            gemini_settings = get_settings().gemini_settings
            if gemini_settings.api_key:
                self.gemini_service = GeminiService(gemini_settings)
                self.logger.info("Gemini service initialized")
            
            # Initialize OpenAI service
            openai_settings = get_settings().openai_settings
            if openai_settings.api_key:
                self.openai_service = OpenAIService(openai_settings)
                self.logger.info("OpenAI service initialized")
            
            # Initialize DistilBERT service
            try:
                self.distilbert_service = DistilBERTService()
                self.logger.info("DistilBERT service initialized")
            except Exception as e:
                self.logger.warning(f"DistilBERT service failed to initialize: {e}")
            
            # Initialize hybrid service if both are available
            if self.gemini_service and self.distilbert_service:
                self.hybrid_service = HybridExtractionService(
                    self.gemini_service, self.distilbert_service
                )
                self.logger.info("Hybrid extraction service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI services: {e}")
        
        # Log available services
        available_services = []
        if self.gemini_service:
            available_services.append(AIProvider.GEMINI)
        if self.openai_service:
            available_services.append(AIProvider.OPENAI)
        if self.distilbert_service:
            available_services.append(AIProvider.DISTILBERT)
        if self.hybrid_service:
            available_services.append("hybrid")
        
        self.logger.info(f"AI services available: {available_services}")
        
        # Traditional Q&A model
        self.qa_model = None # This line was removed from the new_code, so it's removed here.
        
        # Initialize lightweight RAG service (no external deps)
        try:
            self.simple_rag = SimpleRAGService()
            self.logger.info("Simple RAG service initialized")
        except Exception as e:
            self.logger.warning(f"Simple RAG initialization failed: {e}")
            self.simple_rag = None

        # Final verification (second pass)
        try:
            self.final_verifier = FinalVerificationService()
            self.logger.info("Final verification service initialized")
        except Exception as e:
            self.logger.warning(f"Final verification init failed: {e}")
            self.final_verifier = None
    
    def _load_model(self, model_name: str = None) -> Tuple[Any, Any, QuestionAnsweringPipeline]:
        """Load the Q&A model, tokenizer, and create pipeline."""
        model_name = model_name or self.settings.model_name
        
        if (self._model is None or self._tokenizer is None or 
            self._pipeline is None or self._model_name != model_name):
            
            logger.info(f"Loading Q&A model: {model_name}")
            start_time = time.time()
            
            try:
                # Load tokenizer and model
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                
                # Create pipeline
                self._pipeline = pipeline(
                    "question-answering",
                    model=self._model,
                    tokenizer=self._tokenizer,
                    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
                )
                
                self._model_name = model_name
                
                load_time = time.time() - start_time
                logger.info(f"Q&A model loaded in {load_time:.2f} seconds")
                
                return self._model, self._tokenizer, self._pipeline
                
            except Exception as e:
                logger.error(f"Failed to load Q&A model {model_name}: {e}")
                raise
        
        return self._model, self._tokenizer, self._pipeline
    
    def _preprocess_context(self, context: str) -> str:
        """Preprocess the context text for better Q&A performance."""
        # Remove excessive whitespace
        context = re.sub(r'\s+', ' ', context).strip()
        
        # Remove speaker labels like "Speaker 1:", "Nurse:", etc.
        context = re.sub(r'\b(Speaker\s*\d+|Nurse|Patient|Doctor)\s*:\s*', '', context, flags=re.IGNORECASE)
        
        # Remove timestamp markers like [00:01:23]
        context = re.sub(r'\[\d{2}:\d{2}:\d{2}\]', '', context)
        
        # Clean up multiple punctuation
        context = re.sub(r'[.]{2,}', '.', context)
        context = re.sub(r'[!]{2,}', '!', context)
        context = re.sub(r'[?]{2,}', '?', context)
        
        return context.strip()
    
    def _chunk_context(self, context: str, max_length: int = None) -> List[str]:
        """
        Split long context into chunks for processing.
        Tries to split on sentence boundaries.
        """
        max_length = max_length or self.settings.max_context_length
        
        if len(context) <= max_length:
            return [context]
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', context)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed the limit
            if len(current_chunk) + len(sentence) + 2 <= max_length:  # +2 for punctuation and space
                current_chunk += sentence + ". "
            else:
                # Save current chunk and start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If chunks are still too long, split them word-wise
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                words = chunk.split()
                current_word_chunk = ""
                
                for word in words:
                    if len(current_word_chunk) + len(word) + 1 <= max_length:
                        current_word_chunk += word + " "
                    else:
                        if current_word_chunk:
                            final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word + " "
                
                if current_word_chunk:
                    final_chunks.append(current_word_chunk.strip())
        
        return final_chunks
    
    def _find_answer_in_segments(
        self, 
        answer_text: str, 
        segments: List[Dict], 
        context: str
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Find the timestamp and context snippet for an answer within the segments.
        
        Returns:
            Tuple of (start_time, end_time, context_snippet)
        """
        if not answer_text or not segments:
            return None, None, None
        
        # Find the answer text in the full context
        answer_start = context.lower().find(answer_text.lower())
        if answer_start == -1:
            return None, None, None
        
        answer_end = answer_start + len(answer_text)
        
        # Find which segment(s) contain the answer
        current_pos = 0
        start_time = None
        end_time = None
        context_snippet = ""
        
        for segment in segments:
            segment_text = segment.get("text", "").strip()
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)
            
            segment_text_start = current_pos
            segment_text_end = current_pos + len(segment_text)
            
            # Check if answer overlaps with this segment
            if (answer_start < segment_text_end and answer_end > segment_text_start):
                if start_time is None:
                    start_time = segment_start
                end_time = segment_end
                
                # Add segment text to context snippet
                if context_snippet:
                    context_snippet += " "
                context_snippet += segment_text
            
            current_pos = segment_text_end + 1  # +1 for space between segments
        
        # Trim context snippet to reasonable length
        if len(context_snippet) > 200:
            # Try to center around the answer
            answer_pos_in_snippet = context_snippet.lower().find(answer_text.lower())
            if answer_pos_in_snippet != -1:
                start = max(0, answer_pos_in_snippet - 100)
                end = min(len(context_snippet), answer_pos_in_snippet + len(answer_text) + 100)
                context_snippet = context_snippet[start:end]
                if start > 0:
                    context_snippet = "..." + context_snippet
                if end < len(context_snippet):
                    context_snippet = context_snippet + "..."
        
        return start_time, end_time, context_snippet
    
    def _extract_single_answer(
        self, 
        context: str, 
        question: str, 
        segments: List[Dict] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Extract a single answer to a question from the context using the traditional model.
        """
        try:
            # Load model
            model, tokenizer, pipeline_obj = self._load_model()
            
            # Preprocess context
            context = self._preprocess_context(context)
            
            if not context.strip():
                return {
                    "answer": None,
                    "confidence_score": 0.0,
                    "is_confident": False,
                    "context_snippet": None,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "error": "Empty context after preprocessing"
                }
            
            # Split context into chunks if too long
            context_chunks = self._chunk_context(context)
            
            best_answer = None
            best_score = 0.0
            best_chunk = None
            
            # Process each chunk
            for chunk in context_chunks:
                try:
                    # Get answers from pipeline
                    results = pipeline_obj(
                        question=question,
                        context=chunk,
                        top_k=top_k,
                        max_answer_len=200,
                        max_seq_len=512,
                        max_question_len=self.settings.max_question_length
                    )
                    
                    # Handle both single result and list of results
                    if not isinstance(results, list):
                        results = [results]
                    
                    # Find the best answer from this chunk
                    for result in results:
                        score = result.get("score", 0.0)
                        if score > best_score:
                            best_answer = result
                            best_score = score
                            best_chunk = chunk
                            
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue
            
            if best_answer is None:
                return {
                    "answer": None,
                    "confidence_score": 0.0,
                    "is_confident": False,
                    "context_snippet": None,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "error": "No answer found in any context chunk"
                }
            
            answer_text = best_answer.get("answer", "").strip()
            confidence_score = float(best_answer.get("score", 0.0))
            
            # Determine if the answer is confident enough
            is_confident = confidence_score >= self.settings.confidence_threshold
            
            # Find timestamps if segments are provided
            timestamp_start, timestamp_end, context_snippet = None, None, None
            if segments and answer_text:
                timestamp_start, timestamp_end, context_snippet = self._find_answer_in_segments(
                    answer_text, segments, best_chunk
                )
            
            # If no context snippet found, create one from the best chunk
            if not context_snippet and answer_text and best_chunk:
                answer_pos = best_chunk.lower().find(answer_text.lower())
                if answer_pos != -1:
                    start = max(0, answer_pos - 50)
                    end = min(len(best_chunk), answer_pos + len(answer_text) + 50)
                    context_snippet = best_chunk[start:end]
                    if start > 0:
                        context_snippet = "..." + context_snippet
                    if end < len(best_chunk):
                        context_snippet = context_snippet + "..."
            
            # Return "not sure" message if confidence is too low
            if not is_confident and confidence_score < 0.1:
                answer_text = "pls check by yourself, I'm not sure"
            elif not is_confident:
                answer_text = answer_text  # Keep the answer but mark as not confident
            
            return {
                "answer": answer_text,
                "confidence_score": confidence_score,
                "is_confident": is_confident,
                "context_snippet": context_snippet,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "model_used": self._model_name,
                "processing_info": {
                    "chunks_processed": len(context_chunks),
                    "best_chunk_index": context_chunks.index(best_chunk) if best_chunk in context_chunks else -1,
                    "question_length": len(question),
                    "context_length": len(context)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting answer for question '{question}': {e}")
            return {
                "answer": "pls check by yourself, I'm not sure",
                "confidence_score": 0.0,
                "is_confident": False,
                "context_snippet": None,
                "timestamp_start": None,
                "timestamp_end": None,
                "error": str(e)
            }
    
    async def extract_answers_ai(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]] = None,
        provider: AIProvider = None,
        fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Extract answers using AI services (Hybrid Gemini+DistilBERT with verification preferred).
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            provider: Specific AI provider to use
            fallback: Whether to fallback to other provider if primary fails
            
        Returns:
            Dictionary with extracted information
        """
        questions = questions or PREDEFINED_QUESTIONS
        
        try:
            # Try hybrid service first (best accuracy with verification)
            if self.hybrid_service and not provider:
                logger.info("Using hybrid Gemini + DistilBERT extraction with final verification for best accuracy")
                start_time = time.time()
                
                result = await self.hybrid_service.extract_medical_info(
                    conversation_text=conversation_text,
                    questions=questions
                )
                
                extraction_time = time.time() - start_time
                quality_score = result.get('quality_score', 0.0)
                extraction_method = result.get('extraction_method', 'unknown')
                
                logger.info(f"Hybrid extraction with verification completed in {extraction_time:.2f} seconds")
                logger.info(f"Quality score: {quality_score:.2f}, Method: {extraction_method}")
                
                # Remove metadata fields for backward compatibility
                clean_result = {k: v for k, v in result.items() 
                              if k not in ['quality_score', 'extraction_method', 'extraction_timestamp', 'conversation_length']}
                
                return {
                    "extracted_info": clean_result,
                    "method": "hybrid_gemini_distilbert_verified",
                    "provider": "hybrid",
                    "processing_time": extraction_time,
                    "questions_processed": len(questions),
                    "quality_score": quality_score,
                    "extraction_method": extraction_method
                }
            
            # Fallback to individual services
            elif self.gemini_service and (provider == AIProvider.GEMINI or not provider):
                logger.info(f"Using Gemini service for extraction")
                start_time = time.time()
                
                result = await self.gemini_service.extract_medical_info(
                    conversation_text=conversation_text,
                    questions=questions
                )
                
                extraction_time = time.time() - start_time
                logger.info(f"Gemini extraction completed in {extraction_time:.2f} seconds")
                
                return {
                    "extracted_info": result,
                    "method": "ai_service",
                    "provider": "gemini",
                    "processing_time": extraction_time,
                    "questions_processed": len(questions)
                }
            
            # Fallback to DistilBERT if available
            elif self.distilbert_service and (provider == AIProvider.DISTILBERT or not provider):
                logger.info(f"Using DistilBERT service for extraction")
                start_time = time.time()
                
                result = self.distilbert_service.extract_medical_info(
                    conversation_text=conversation_text,
                    questions=questions
                )
                
                extraction_time = time.time() - start_time
                logger.info(f"DistilBERT extraction completed in {extraction_time:.2f} seconds")
                
                return {
                    "extracted_info": result,
                    "method": "ai_service",
                    "provider": "distilbert",
                    "processing_time": extraction_time,
                    "questions_processed": len(questions)
                }
            
            # Fallback to OpenAI if available
            elif self.openai_service and (provider == AIProvider.OPENAI or not provider):
                logger.info(f"Using OpenAI service for extraction")
                start_time = time.time()
                
                result = await self.openai_service.extract_medical_info(
                    conversation_text=conversation_text,
                    questions=questions
                )
                
                extraction_time = time.time() - start_time
                logger.info(f"OpenAI extraction completed in {extraction_time:.2f} seconds")
                
                return {
                    "extracted_info": result,
                    "method": "ai_service",
                    "provider": "openai",
                    "processing_time": extraction_time,
                    "questions_processed": len(questions)
                }
            
            else:
                raise Exception("No AI services available")
                
        except Exception as e:
            logger.error(f"AI extraction failed: {e}")
            if fallback and self.use_ai_service:
                logger.info("Falling back to traditional Q&A model")
                return await self.extract_answers_traditional(conversation_text, questions)
            else:
                raise
    
    async def extract_answers_traditional(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Extract answers using traditional Hugging Face models.
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            
        Returns:
            Dictionary with extracted information
        """
        questions = questions or PREDEFINED_QUESTIONS
        
        try:
            logger.info("Extracting answers using traditional Q&A model")
            start_time = time.time()
            
            # Load model if not already loaded
            self._load_model()
            
            # Preprocess context
            processed_context = self._preprocess_context(conversation_text)
            
            # Extract answers for each question
            extracted_info = {}
            for question_data in questions:
                question_id = question_data["id"]
                question_text = question_data["question"]
                
                try:
                    answer = self._extract_single_answer(processed_context, question_text)
                    if answer:
                        extracted_info[question_id] = answer
                except Exception as e:
                    logger.warning(f"Failed to extract answer for {question_id}: {e}")
                    extracted_info[question_id] = "Extraction failed"
            
            extraction_time = time.time() - start_time
            logger.info(f"Traditional extraction completed in {extraction_time:.2f} seconds")
            
            return {
                "extracted_info": extracted_info,
                "method": "traditional_model",
                "provider": "huggingface",
                "processing_time": extraction_time,
                "questions_processed": len(questions)
            }
            
        except Exception as e:
            logger.error(f"Traditional extraction failed: {e}")
            raise e
    
    async def extract_answers(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]] = None,
        prefer_ai: bool = True,
        provider: AIProvider = None
    ) -> Dict[str, Any]:
        """
        Extract answers using the best available method.
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            prefer_ai: Whether to prefer AI services over traditional models
            provider: Specific AI provider to use
            
        Returns:
            Dictionary with extracted information
        """
        if prefer_ai and self.use_ai_service:
            try:
                return await self.extract_answers_ai(conversation_text, questions, provider)
            except Exception as e:
                logger.warning(f"AI extraction failed, falling back to traditional: {e}")
                return await self.extract_answers_traditional(conversation_text, questions)
        else:
            return await self.extract_answers_traditional(conversation_text, questions)
    
    async def summarize_conversation_ai(
        self,
        conversation_text: str,
        summary_type: str = "medical",
        provider: AIProvider = None
    ) -> str:
        """
        Generate conversation summary using AI services.
        
        Args:
            conversation_text: The transcribed conversation
            summary_type: Type of summary to generate
            provider: Specific AI provider to use
            
        Returns:
            Summary text
        """
        try:
            logger.info(f"Generating {summary_type} summary using AI service")
            return await self.ai_qa_service.summarize_conversation(
                conversation_text, summary_type, provider
            )
        except Exception as e:
            logger.error(f"AI summary generation failed: {e}")
            raise e
    
    async def extract_multiple_answers(
        self,
        questions: List[Dict[str, str]],
        context: str,
        segments: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract answers to multiple questions from the same context using AI services.
        
        Args:
            questions: List of question dictionaries with 'id', 'question', and 'category'
            context: The text context to search for answers
            segments: Optional list of segments with timestamps
            
        Returns:
            List of answer dictionaries
        """
        logger.info(f"Extracting answers for {len(questions)} questions using AI services")
        start_time = time.time()
        
        try:
            # Use AI service to extract answers
            result = await self.extract_answers_ai(
                conversation_text=context,
                questions=questions
            )
            
            # Debug logging to see what we're getting
            logger.info(f"AI service result: {result}")
            logger.info(f"Extracted info: {result.get('extracted_info', {})}")
            logger.info(f"Extracted info type: {type(result.get('extracted_info', {}))}")
            
            # Process the AI results to match the expected format
            extracted_info = result.get("extracted_info", {})
            processed_answers = []
            verification_start = time.time()
            
            for question_data in questions:
                question_id = question_data.get("id")
                question_text = question_data.get("question")
                category = question_data.get("category", "unknown")
                
                if not question_text:
                    logger.warning(f"Skipping question {question_id} - no question text")
                    continue
                
                # Debug logging for each question
                logger.info(f"Processing question {question_id}: {question_text}")
                logger.info(f"Looking for answer in: {extracted_info}")
                
                # Get answer from AI results - handle different response formats
                answer_data = extracted_info.get(question_id)
                logger.info(f"Answer data for {question_id}: {answer_data} (type: {type(answer_data)})")
                
                if answer_data:
                    if isinstance(answer_data, dict):
                        # Structured response with metadata
                        answer_text = answer_data.get("answer", str(answer_data))
                        confidence_score = answer_data.get("confidence_score", 0.0)
                        is_confident = answer_data.get("is_confident", False)
                        context_snippet = answer_data.get("context_snippet", "")
                        timestamp_start = answer_data.get("timestamp_start")
                        timestamp_end = answer_data.get("timestamp_end")
                        model_used = answer_data.get("model_used", "ai_service")
                        
                        # Recalculate confidence if not provided or if it seems wrong
                        if confidence_score == 0.0 or confidence_score == 0.8:
                            confidence_score = self._calculate_answer_confidence(answer_text, question_text, context)
                        
                        # Determine if manual review is needed
                        needs_review = self._needs_manual_review(answer_text, question_text, confidence_score)
                        is_confident = is_confident and not needs_review
                    else:
                        # Simple string response
                        answer_text = str(answer_data)
                        # Calculate confidence based on answer quality
                        confidence_score = self._calculate_answer_confidence(answer_text, question_text, context)
                        is_confident = confidence_score > 0.6
                        context_snippet = ""
                        timestamp_start = None
                        timestamp_end = None
                        model_used = "ai_service"
                    
                    # Determine if manual review is needed
                    needs_review = self._needs_manual_review(answer_text, question_text, confidence_score)
                    is_confident = is_confident and not needs_review
                    
                    logger.info(f"Found answer for {question_id}: {answer_text} (confidence: {confidence_score:.2f}, review: {needs_review})")
                else:
                    # No answer found
                    answer_text = "Information not available in conversation"
                    confidence_score = 0.0
                    is_confident = False
                    context_snippet = ""
                    timestamp_start = None
                    timestamp_end = None
                    model_used = "ai_service"
                    
                    logger.info(f"No answer found for {question_id}")
                
                # Find timestamps if segments are provided and we don't have them
                if segments and not timestamp_start and answer_text and answer_text != "Information not available in conversation":
                    ts_start, ts_end, snippet = self._find_answer_in_segments(
                        answer_text, segments, context
                    )
                    if ts_start is not None:
                        timestamp_start = ts_start
                        timestamp_end = ts_end
                    if snippet and not context_snippet:
                        context_snippet = snippet
                
                # RAG metadata enrichment (non-destructive)
                rag_guidelines = []
                rag_context = []
                if getattr(self, "simple_rag", None):
                    try:
                        rag_meta = self.simple_rag.enhance_answer(
                            base_answer=answer_text,
                            question=question_text,
                            conversation_text=context,
                            top_k=2,
                        )
                        rag_guidelines = rag_meta.get("guidelines_applied", [])
                        rag_context = rag_meta.get("rag_context", [])
                    except Exception as e:
                        logger.warning(f"RAG enhancement failed for {question_id}: {e}")
                
                # Create answer result
                answer_result = {
                    "question_id": question_id,
                    "question_text": question_text,
                    "category": category,
                    "answer": answer_text,
                    "confidence_score": confidence_score,
                    "is_confident": is_confident,
                    "context_snippet": context_snippet,
                    "timestamp_start": timestamp_start,
                    "timestamp_end": timestamp_end,
                    "model_used": model_used,
                    "rag_guidelines": rag_guidelines,
                    "rag_context": rag_context
                }
                
                processed_answers.append(answer_result)
            
            # Optional second-pass verification to improve accuracy
            verification_time = 0.0
            if self.final_verifier and isinstance(extracted_info, dict):
                try:
                    # Build a compact dict {question_id: answer}
                    compact = {a["question_id"]: a["answer"] for a in processed_answers}
                    verified = self.final_verifier.verify_and_clean_results(
                        compact, context, questions
                    )
                    # Apply verified answers back
                    for a in processed_answers:
                        if a["question_id"] in verified:
                            a["answer"] = verified[a["question_id"]]
                finally:
                    verification_time = time.time() - verification_start
            
            processing_time = time.time() - start_time
            logger.info(f"Completed Q&A extraction in {processing_time:.2f} seconds")
            logger.info(f"Extracted {len([a for a in processed_answers if a['answer'] != 'Information not available in conversation'])} answers")
            
            for ans in processed_answers:
                ans["processing_time_seconds"] = round(processing_time, 3)
                if verification_time:
                    ans["verification_time_seconds"] = round(verification_time, 3)
            return processed_answers
            
        except Exception as e:
            logger.error(f"AI-based Q&A extraction failed: {e}")
            # Return fallback answers if AI service fails
            fallback_answers = []
            for question_data in questions:
                fallback_answers.append({
                    "question_id": question_data.get("id"),
                    "question_text": question_data.get("question"),
                    "category": question_data.get("category", "unknown"),
                    "answer": "AI service unavailable - manual review required",
                    "confidence_score": 0.0,
                    "is_confident": False,
                    "context_snippet": "",
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "model_used": "fallback"
                })
            return fallback_answers
    
    def extract_predefined_answers(
        self,
        context: str,
        segments: List[Dict] = None,
        custom_questions: List[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract answers to predefined nurse conversation questions.
        
        Args:
            context: The conversation transcript
            segments: Optional segments with timestamps
            custom_questions: Optional additional questions to ask
            
        Returns:
            List of answer dictionaries
        """
        questions = PREDEFINED_QUESTIONS.copy()
        
        # Add custom questions if provided
        if custom_questions:
            for custom_q in custom_questions:
                if "question" in custom_q:
                    questions.append({
                        "id": custom_q.get("id", f"custom_{len(questions)}"),
                        "question": custom_q["question"],
                        "category": custom_q.get("category", "custom")
                    })
        
        return self.extract_multiple_answers(questions, context, segments)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded Q&A model."""
        return {
            "model_name": self._model_name,
            "is_loaded": self._model is not None,
            "max_context_length": self.settings.max_context_length,
            "max_question_length": self.settings.max_question_length,
            "confidence_threshold": self.settings.confidence_threshold,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "supported_tasks": ["question-answering", "extractive-qa"]
        }
    
    def validate_question(self, question: str) -> Tuple[bool, str]:
        """
        Validate if a question is appropriate for the Q&A model.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if len(question) > self.settings.max_question_length:
            return False, f"Question too long (max {self.settings.max_question_length} characters)"
        
        # Basic question format validation
        question = question.strip()
        if not any(question.lower().startswith(word) for word in 
                  ["what", "who", "when", "where", "why", "how", "which", "is", "does", "can", "did", "will"]):
            return False, "Question should start with a question word (what, who, when, etc.)"
        
        return True, ""

    def _calculate_answer_confidence(self, answer_text: str, question: str, context: str | None = None) -> float:
        """
        Calculate confidence score based on answer quality and relevance.
        
        Args:
            answer_text: The extracted answer
            question: The question being answered
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not answer_text or answer_text == "Information not available in conversation":
            return 0.0
        
        # Base confidence
        confidence = 0.5

        answer_lower = answer_text.lower()
        question_lower = question.lower()

        # Boost if the exact answer text appears in context
        if context:
            try:
                ctx_lower = context.lower()
                if answer_lower and answer_lower in ctx_lower:
                    confidence += 0.25
                else:
                    # Token overlap boost
                    ans_tokens = [t for t in answer_lower.replace(',', ' ').split() if len(t) > 1]
                    if ans_tokens:
                        overlap = sum(1 for t in ans_tokens if t in ctx_lower)
                        ratio = overlap / max(1, len(ans_tokens))
                        if ratio >= 0.6:
                            confidence += 0.15
            except Exception:
                pass
        
        # Boost for specific, factual answers
        if any(word in answer_lower for word in ['mg', 'ml', '°f', '°c', 'mmhg', 'mg/dl']):
            confidence += 0.2  # Medical measurements
        
        if any(word in answer_lower for word in ['yes', 'no', 'true', 'false']):
            confidence += 0.1  # Clear yes/no answers
        
        if any(word in answer_lower for word in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']):
            confidence += 0.15  # Numeric answers

        # Category/intent specific boosts inferred from the question
        # Patient name detection: two capitalized words
        if 'name' in question_lower:
            parts = [p for p in answer_text.split() if p]
            if len(parts) >= 2 and all(p[:1].isupper() for p in parts[:2]):
                confidence += 0.25

        # Follow-up day detection
        if 'follow' in question_lower or 'when' in question_lower:
            days = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
            if any(d in answer_lower for d in days):
                confidence += 0.2

        # Allergies
        if 'allerg' in question_lower and any(k in answer_lower for k in ['penicillin','allergy','rash','swelling']):
            confidence += 0.2

        # Vital signs
        if 'vital' in question_lower or 'blood pressure' in question_lower or 'bp' in question_lower:
            if any(k in answer_lower for k in ['bp','blood pressure','pulse','resp','oxygen','temp','mmhg','°f','°c']) and any(ch.isdigit() for ch in answer_lower):
                confidence += 0.25

        # Medications list / dosing
        if 'medication' in question_lower:
            if any(u in answer_lower for u in ['mg','ml']) or ',' in answer_text or any(ch.isdigit() for ch in answer_lower):
                confidence += 0.25
        
        # Penalty for generic or unclear answers
        if len(answer_text) < 3:
            confidence -= 0.3  # Too short
        
        if answer_text.lower() in ['unknown', 'none', 'n/a', 'not mentioned']:
            confidence -= 0.4  # Generic negative responses
        
        if 'information not available' in answer_text.lower():
            confidence -= 0.5  # No information found
        
        # Penalty for answers that seem to answer different questions
        if 'age' in question_lower and any(word in answer_lower for word in ['cancer', 'diagnosis', 'treatment']):
            confidence -= 0.6  # Wrong type of answer
        
        if 'pain level' in question_lower and any(word in answer_lower for word in ['cancer', 'diagnosis', 'treatment']):
            confidence -= 0.6  # Wrong type of answer
        
        if 'medications' in question_lower and any(word in answer_lower for word in ['age', 'diagnosis', 'pain']):
            confidence -= 0.6  # Wrong type of answer
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))

    def _needs_manual_review(self, answer_text: str, question: str, confidence_score: float) -> bool:
        """
        Determine if an answer needs manual review based on its confidence and content.
        
        Args:
            answer_text: The extracted answer
            question: The question being answered
            confidence_score: The calculated confidence score
            
        Returns:
            True if manual review is needed, False otherwise
        """
        # If confidence is very low, always require manual review
        if confidence_score < 0.1:
            return True
        
        # If answer is very short or generic, require manual review
        if len(answer_text) < 10 or answer_text.lower() in ['unknown', 'none', 'n/a', 'not mentioned']:
            return True
        
        # If answer contains specific medical terms or measurements, but confidence is low
        if any(word in answer_text.lower() for word in ['mg', 'ml', '°f', '°c', 'mmhg', 'mg/dl']) and confidence_score < 0.7:
            return True
        
        # If answer is a yes/no, but confidence is low
        if any(word in answer_text.lower() for word in ['yes', 'no', 'true', 'false']) and confidence_score < 0.7:
            return True
        
        # If answer is a numeric value, but confidence is low
        if any(word in answer_text.lower() for word in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) and confidence_score < 0.7:
            return True
        
        # If answer is a vague or generic response, but confidence is high
        if confidence_score > 0.9 and answer_text.lower() in ['information not available', 'pls check by yourself, i\'m not sure']:
            return True
        
        # If answer is a very confident, but potentially wrong type (e.g., age for pain)
        if confidence_score > 0.8 and any(word in question.lower() for word in ['age', 'pain']):
            if any(word in answer_text.lower() for word in ['cancer', 'diagnosis', 'treatment']):
                return True
        
        # If answer is a very confident, but potentially wrong type (e.g., pain for medications)
        if confidence_score > 0.8 and any(word in question.lower() for word in ['medications']):
            if any(word in answer_text.lower() for word in ['age', 'diagnosis', 'pain']):
                return True
        
        return False