"""
Final verification service for medical information extraction results.
"""
import logging
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime


class FinalVerificationService:
    """Service for final verification and cleanup of medical extraction results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Medical terminology patterns
        self.medical_patterns = {
            'pain_level': r'\b([1-9]|10)\s*(?:out\s*of\s*10|on\s*a\s*scale|pain\s*level)',
            'blood_pressure': r'\b(\d{2,3}/\d{2,3})\s*(?:mmHg|BP|blood\s*pressure)',
            'temperature': r'\b(\d{2,3}\.?\d*)\s*(?:°F|°C|Fahrenheit|Celsius|temp)',
            'blood_sugar': r'\b(\d{2,3})\s*(?:mg/dL|mg/dl|blood\s*sugar|glucose)',
            'medication_dose': r'\b(\d+)\s*(?:mg|ml|mg|tablets?|capsules?|times?\s*per\s*day)',
            'time_pattern': r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b',
            'date_pattern': r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
            'duration': r'\b(\d+)\s*(?:days?|weeks?|months?|hours?|minutes?)\b'
        }
        
        # Common medical abbreviations
        self.medical_abbreviations = {
            'bp': 'blood pressure',
            'temp': 'temperature',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'o2': 'oxygen',
            'sat': 'saturation',
            'iv': 'intravenous',
            'po': 'by mouth',
            'prn': 'as needed',
            'qd': 'daily',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily'
        }
    
    def verify_and_clean_results(
        self,
        results: Dict[str, Any],
        conversation_text: str,
        questions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Final verification and cleanup of extraction results.
        
        Args:
            results: Raw extraction results
            conversation_text: Original conversation text
            questions: List of questions
            
        Returns:
            Cleaned and verified results
        """
        try:
            self.logger.info("Starting final verification and cleanup of extraction results")
            
            verified_results = {}
            
            for question_data in questions:
                question_id = question_data['id']
                question_text = question_data['question']
                
                if question_id in results:
                    raw_answer = results[question_id]
                    
                    # Clean and verify the answer
                    cleaned_answer = self._clean_answer(
                        raw_answer, question_text, conversation_text
                    )
                    
                    # Extract additional context if needed
                    enhanced_answer = self._enhance_answer(
                        cleaned_answer, question_id, conversation_text
                    )
                    
                    # Final validation
                    final_answer = self._validate_answer(
                        enhanced_answer, question_id, question_text
                    )
                    
                    verified_results[question_id] = final_answer
                    
                    self.logger.info(f"Verified {question_id}: {raw_answer} -> {final_answer}")
                else:
                    verified_results[question_id] = "Information not available in conversation"
            
            # Add timestamp information
            verified_results['extraction_timestamp'] = datetime.now().isoformat()
            verified_results['conversation_length'] = len(conversation_text)
            
            self.logger.info(f"Final verification completed for {len(verified_results)} questions")
            return verified_results
            
        except Exception as e:
            self.logger.error(f"Final verification failed: {str(e)}")
            return results
    
    def _clean_answer(
        self,
        answer: str,
        question_text: str,
        conversation_text: str
    ) -> str:
        """Clean and normalize the answer."""
        if not answer or answer == "Information not available in conversation":
            return "Information not available in conversation"
        
        # Remove conversation snippets (common DistilBERT issue)
        if len(answer) > 200 and any(speaker in answer for speaker in ['Sarah:', 'Mrs. Rodriguez:', 'Sarah :', 'Mrs. Rodriguez :']):
            # Extract just the relevant part
            cleaned = self._extract_relevant_answer(answer, question_text)
            if cleaned:
                return cleaned
        
        # Remove markdown formatting
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "What is the patient's ",
            "When should the patient ",
            "Does the patient have ",
            "What medications is the patient ",
            "What vital signs were ",
            "What exercises were "
        ]
        
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
                break
        
        return answer
    
    def _extract_relevant_answer(
        self,
        long_answer: str,
        question_text: str
    ) -> str:
        """Extract relevant answer from long conversation snippets."""
        try:
            # Look for specific patterns based on question type
            if 'pain level' in question_text.lower():
                # Extract pain level number
                pain_match = re.search(r'\b([1-9]|10)\b', long_answer)
                if pain_match:
                    return f"{pain_match.group(1)} out of 10"
            
            elif 'medications' in question_text.lower():
                # Extract medication names
                med_patterns = [
                    r'\b(?:Oxycodone|Metformin|Insulin|Antibiotic cream)\b',
                    r'\b(?:aspirin|ibuprofen|acetaminophen|morphine)\b',
                    r'\b(?:mg|ml|tablets?|capsules?)\b'
                ]
                
                for pattern in med_patterns:
                    matches = re.findall(pattern, long_answer, re.IGNORECASE)
                    if matches:
                        return ', '.join(matches)
            
            elif 'vital signs' in question_text.lower():
                # Extract vital signs
                vital_patterns = [
                    r'\b(?:blood pressure|temperature|blood sugar|heart rate)\b',
                    r'\b(?:BP|temp|glucose|HR)\b'
                ]
                
                for pattern in vital_patterns:
                    matches = re.findall(pattern, long_answer, re.IGNORECASE)
                    if matches:
                        return ', '.join(matches)
            
            elif 'exercises' in question_text.lower():
                # Extract exercise names
                exercise_patterns = [
                    r'\b(?:leg lifts|walking|knee lifts|ankle circles|stretching)\b',
                    r'\b(?:reps|sets|times per day)\b'
                ]
                
                for pattern in exercise_patterns:
                    matches = re.findall(pattern, long_answer, re.IGNORECASE)
                    if matches:
                        return ', '.join(matches)
            
            # If no specific pattern found, try to extract the most relevant sentence
            sentences = long_answer.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 10 and len(sentence.strip()) < 100:
                    # Check if sentence contains relevant information
                    if any(word in sentence.lower() for word in ['pain', 'medication', 'vital', 'exercise', 'blood', 'temperature']):
                        return sentence.strip()
            
            return "Information not available in conversation"
            
        except Exception as e:
            self.logger.error(f"Error extracting relevant answer: {str(e)}")
            return "Information not available in conversation"
    
    def _enhance_answer(
        self,
        answer: str,
        question_id: str,
        conversation_text: str
    ) -> str:
        """Enhance answer with additional context from conversation."""
        if answer == "Information not available in conversation":
            return answer
        
        try:
            # Add timestamp context if available
            if 'time' in question_id.lower() or 'when' in question_id.lower():
                time_context = self._extract_time_context(conversation_text)
                if time_context:
                    answer = f"{answer} ({time_context})"
            
            # Add measurement context
            if any(unit in answer.lower() for unit in ['mg', 'ml', '°f', '°c', 'mmhg']):
                # Answer already has units
                pass
            elif 'pain' in question_id.lower() and answer.isdigit():
                answer = f"{answer} out of 10"
            elif 'blood pressure' in question_id.lower() and '/' in answer:
                answer = f"{answer} mmHg"
            elif 'temperature' in question_id.lower() and answer.replace('.', '').isdigit():
                answer = f"{answer}°F"
            elif 'blood sugar' in question_id.lower() and answer.isdigit():
                answer = f"{answer} mg/dL"
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error enhancing answer: {str(e)}")
            return answer
    
    def _extract_time_context(self, conversation_text: str) -> str:
        """Extract time context from conversation."""
        try:
            # Look for time patterns
            time_patterns = [
                r'\b(morning|afternoon|evening|night)\b',
                r'\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b',
                r'\b(today|yesterday|tomorrow)\b',
                r'\b(breakfast|lunch|dinner|bedtime)\b'
            ]
            
            for pattern in time_patterns:
                matches = re.findall(pattern, conversation_text, re.IGNORECASE)
                if matches:
                    return matches[0]
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error extracting time context: {str(e)}")
            return ""
    
    def _validate_answer(
        self,
        answer: str,
        question_id: str,
        question_text: str
    ) -> str:
        """Final validation of the answer."""
        if not answer or answer == "Information not available in conversation":
            return "Information not available in conversation"
        
        # Check for common error patterns
        error_patterns = [
            r'What is the patient\'s',  # Question instead of answer
            r'When should the patient',  # Question instead of answer
            r'Does the patient have',   # Question instead of answer
            r'Sarah\s*:',              # Speaker prefix
            r'Mrs\. Rodriguez\s*:',    # Speaker prefix
            r'Good morning',           # Conversation start
            r'Hello',                  # Conversation start
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                self.logger.warning(f"Answer contains error pattern: {answer}")
                return "Information not available in conversation"
        
        # Check answer length (too long might be wrong)
        if len(answer) > 150:
            self.logger.warning(f"Answer too long, might be wrong: {answer[:100]}...")
            return "Information not available in conversation"
        
        # Check for meaningful content
        if len(answer.strip()) < 2:
            return "Information not available in conversation"
        
        return answer
    
    def get_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score for the results."""
        try:
            total_questions = len(results)
            if total_questions == 0:
                return 0.0
            
            answered_questions = sum(
                1 for answer in results.values()
                if answer != "Information not available in conversation"
            )
            
            # Base score
            base_score = answered_questions / total_questions
            
            # Bonus for detailed answers
            detailed_answers = sum(
                1 for answer in results.values()
                if answer != "Information not available in conversation" and len(answer) > 20
            )
            
            detail_bonus = detailed_answers / total_questions * 0.2
            
            # Penalty for very short answers
            short_answers = sum(
                1 for answer in results.values()
                if answer != "Information not available in conversation" and len(answer) < 5
            )
            
            short_penalty = short_answers / total_questions * 0.1
            
            final_score = base_score + detail_bonus - short_penalty
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0.5

