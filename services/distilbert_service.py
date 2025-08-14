"""
DistilBERT service for medical Q&A extraction using question-answering model.
"""
import logging
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import re


class DistilBERTService:
    """Service for medical Q&A extraction using DistilBERT."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_name = "distilbert-base-cased-distilled-squad"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the DistilBERT model and tokenizer."""
        try:
            self.logger.info(f"Loading DistilBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.logger.info(f"DistilBERT model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load DistilBERT model: {str(e)}")
            raise
    
    def extract_medical_info(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Extract medical information using DistilBERT question-answering.
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            
        Returns:
            Dictionary with extracted information
        """
        try:
            self.logger.info(f"Extracting medical info using DistilBERT for {len(questions)} questions")
            extracted_info = {}
            
            for question_data in questions:
                question_id = question_data['id']
                question_text = question_data['question']
                
                # Get answer using DistilBERT
                answer = self._get_answer(question_text, conversation_text)
                
                if answer and answer.strip():
                    # Clean up the answer
                    answer = answer.strip()
                    # Remove extra whitespace and normalize
                    answer = re.sub(r'\s+', ' ', answer)
                    
                    # Check if answer is meaningful (not too short or generic)
                    if len(answer) > 3 and answer.lower() not in ['no', 'yes', 'none', 'unknown']:
                        extracted_info[question_id] = answer
                        self.logger.info(f"DistilBERT extracted for {question_id}: {answer}")
                    else:
                        extracted_info[question_id] = "Information not available in conversation"
                        self.logger.info(f"DistilBERT: insufficient answer for {question_id}")
                else:
                    extracted_info[question_id] = "Information not available in conversation"
                    self.logger.info(f"DistilBERT: no answer found for {question_id}")
            
            return extracted_info
            
        except Exception as e:
            self.logger.error(f"DistilBERT extraction failed: {str(e)}")
            return {"error": str(e)}
    
    def _get_answer(self, question: str, context: str) -> str:
        """
        Get answer for a specific question using DistilBERT.
        
        Args:
            question: The question to answer
            context: The conversation context
            
        Returns:
            Extracted answer text
        """
        try:
            # Tokenize the question and context
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get start and end positions
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Find the best start and end positions
            start_idx = torch.argmax(start_scores)
            end_idx = torch.argmax(end_scores)
            
            # Ensure end_idx >= start_idx
            if end_idx < start_idx:
                end_idx = start_idx + 1
            
            # Convert tokens to answer
            answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error getting answer for question '{question}': {str(e)}")
            return ""
    
    def get_confidence_score(self, answer: str, question: str, context: str) -> float:
        """
        Calculate confidence score for the extracted answer.
        
        Args:
            answer: The extracted answer
            question: The original question
            context: The conversation context
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not answer or answer == "Information not available in conversation":
                return 0.0
            
            # Simple confidence scoring based on answer characteristics
            confidence = 0.5  # Base confidence
            
            # Boost confidence for longer, more specific answers
            if len(answer) > 20:
                confidence += 0.2
            
            # Boost confidence if answer contains medical terms
            medical_terms = ['mg', 'ml', 'blood pressure', 'temperature', 'pain', 'medication', 'dose', 'symptoms']
            if any(term in answer.lower() for term in medical_terms):
                confidence += 0.2
            
            # Boost confidence if answer is numeric (vital signs, pain level, etc.)
            if re.search(r'\d+', answer):
                confidence += 0.1
            
            # Reduce confidence for very generic answers
            generic_answers = ['yes', 'no', 'maybe', 'sometimes', 'often', 'rarely']
            if answer.lower() in generic_answers:
                confidence -= 0.2
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

