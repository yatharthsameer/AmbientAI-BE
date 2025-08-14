"""
Hybrid medical information extraction service combining Gemini and DistilBERT.
"""
import logging
from typing import Dict, List, Any, Tuple
from .gemini_service import GeminiService
from .distilbert_service import DistilBERTService
from .final_verification_service import FinalVerificationService
import asyncio


class HybridExtractionService:
    """Service that combines Gemini and DistilBERT for medical information extraction."""
    
    def __init__(self, gemini_service: GeminiService, distilbert_service: DistilBERTService):
        self.gemini_service = gemini_service
        self.distilbert_service = distilbert_service
        self.final_verifier = FinalVerificationService()
        self.logger = logging.getLogger(__name__)
    
    async def extract_medical_info(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Extract medical information using both Gemini and DistilBERT with final verification.
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            
        Returns:
            Dictionary with extracted information and confidence scores
        """
        try:
            self.logger.info("Starting hybrid medical information extraction with final verification")
            
            # Extract using both services
            gemini_results = await self.gemini_service.extract_medical_info(
                conversation_text, questions
            )
            
            distilbert_results = self.distilbert_service.extract_medical_info(
                conversation_text, questions
            )
            
            # Combine and compare results
            combined_results = self._combine_results(
                gemini_results, distilbert_results, questions, conversation_text
            )
            
            # Extract just the answers for final verification
            raw_answers = {}
            for question_id, data in combined_results.items():
                if isinstance(data, dict):
                    raw_answers[question_id] = data["answer"]
                else:
                    raw_answers[question_id] = data
            
            # Final verification and cleanup
            verified_results = self.final_verifier.verify_and_clean_results(
                raw_answers, conversation_text, questions
            )
            
            # Calculate quality score
            quality_score = self.final_verifier.get_quality_score(verified_results)
            
            # Add metadata
            verified_results['quality_score'] = quality_score
            verified_results['extraction_method'] = 'hybrid_gemini_distilbert_verified'
            
            self.logger.info(f"Hybrid extraction with verification completed. Quality score: {quality_score:.2f}")
            return verified_results
            
        except Exception as e:
            self.logger.error(f"Hybrid extraction failed: {str(e)}")
            # Fallback to DistilBERT only
            fallback_results = self.distilbert_service.extract_medical_info(conversation_text, questions)
            
            # Still apply final verification
            verified_fallback = self.final_verifier.verify_and_clean_results(
                fallback_results, conversation_text, questions
            )
            
            quality_score = self.final_verifier.get_quality_score(verified_fallback)
            verified_fallback['quality_score'] = quality_score
            verified_fallback['extraction_method'] = 'fallback_distilbert_verified'
            
            return verified_fallback
    
    def _combine_results(
        self,
        gemini_results: Dict[str, Any],
        distilbert_results: Dict[str, Any],
        questions: List[Dict[str, str]],
        conversation_text: str
    ) -> Dict[str, Any]:
        """
        Combine and compare results from both services.
        
        Args:
            gemini_results: Results from Gemini
            distilbert_results: Results from DistilBERT
            questions: List of questions
            conversation_text: The conversation text
            
        Returns:
            Combined results with confidence scores
        """
        combined_results = {}
        
        for question_data in questions:
            question_id = question_data['id']
            question_text = question_data['question']
            
            gemini_answer = gemini_results.get(question_id, "Information not available in conversation")
            distilbert_answer = distilbert_results.get(question_id, "Information not available in conversation")
            
            # Get confidence scores
            gemini_confidence = self._get_gemini_confidence(gemini_answer)
            distilbert_confidence = self.distilbert_service.get_confidence_score(
                distilbert_answer, question_text, conversation_text
            )
            
            # Choose the better answer based on confidence and quality
            best_answer, best_confidence, source = self._select_best_answer(
                gemini_answer, distilbert_answer,
                gemini_confidence, distilbert_confidence,
                question_text, conversation_text
            )
            
            combined_results[question_id] = {
                "answer": best_answer,
                "confidence_score": best_confidence,
                "source": source,
                "gemini_answer": gemini_answer,
                "gemini_confidence": gemini_confidence,
                "distilbert_answer": distilbert_answer,
                "distilbert_confidence": distilbert_confidence
            }
            
            self.logger.info(f"Combined result for {question_id}: {best_answer} (source: {source}, confidence: {best_confidence:.2f})")
        
        return combined_results
    
    def _get_gemini_confidence(self, answer: str) -> float:
        """Calculate confidence score for Gemini answers."""
        if not answer or answer == "Information not available in conversation":
            return 0.0
        
        # Base confidence for Gemini
        confidence = 0.7
        
        # Boost confidence for structured, detailed answers
        if len(answer) > 30:
            confidence += 0.2
        
        # Boost confidence for medical terminology
        medical_terms = ['mg', 'ml', 'blood pressure', 'temperature', 'pain', 'medication', 'dose', 'symptoms']
        if any(term in answer.lower() for term in medical_terms):
            confidence += 0.1
        
        # Reduce confidence for generic answers
        generic_answers = ['yes', 'no', 'maybe', 'sometimes', 'often', 'rarely']
        if answer.lower() in generic_answers:
            confidence -= 0.3
        
        return min(1.0, max(0.0, confidence))
    
    def _select_best_answer(
        self,
        gemini_answer: str,
        distilbert_answer: str,
        gemini_confidence: float,
        distilbert_confidence: float,
        question_text: str,
        conversation_text: str
    ) -> Tuple[str, float, str]:
        """
        Select the best answer between Gemini and DistilBERT.
        
        Returns:
            Tuple of (best_answer, confidence, source)
        """
        # If one service has no answer, use the other
        if gemini_answer == "Information not available in conversation" and distilbert_answer != "Information not available in conversation":
            return distilbert_answer, distilbert_confidence, "distilbert"
        
        if distilbert_answer == "Information not available in conversation" and gemini_answer != "Information not available in conversation":
            return gemini_answer, gemini_confidence, "gemini"
        
        # If both have answers, compare them
        if gemini_answer != "Information not available in conversation" and distilbert_answer != "Information not available in conversation":
            # Check if answers are similar (indicating high confidence)
            similarity_score = self._calculate_answer_similarity(gemini_answer, distilbert_answer)
            
            if similarity_score > 0.7:  # High similarity - both are likely correct
                # Use the more detailed answer
                if len(gemini_answer) > len(distilbert_answer):
                    return gemini_answer, max(gemini_confidence, distilbert_confidence), "gemini"
                else:
                    return distilbert_answer, max(gemini_confidence, distilbert_confidence), "distilbert"
            else:
                # Low similarity - use the higher confidence answer
                if gemini_confidence > distilbert_confidence:
                    return gemini_answer, gemini_confidence, "gemini"
                else:
                    return distilbert_answer, distilbert_confidence, "distilbert"
        
        # Both have no answer
        return "Information not available in conversation", 0.0, "none"
    
    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers."""
        try:
            # Simple similarity based on common words
            words1 = set(answer1.lower().split())
            words2 = set(answer2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception as e:
            self.logger.error(f"Error calculating answer similarity: {str(e)}")
            return 0.0
