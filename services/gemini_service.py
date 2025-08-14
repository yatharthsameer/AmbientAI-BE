"""
Google Gemini AI service for natural language processing.
"""
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from config import GeminiSettings


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    text: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str


class GeminiService:
    """Service for interacting with Google Gemini AI."""
    
    def __init__(self, settings: GeminiSettings):
        self.settings = settings
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.logger = logging.getLogger(__name__)
        
    async def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GeminiResponse:
        """
        Generate content using Gemini AI.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            GeminiResponse object
        """
        try:
            # Prepare the request payload
            payload = self._build_payload(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature or self.settings.temperature,
                max_tokens=max_tokens or self.settings.max_tokens,
                **kwargs
            )
            
            # Make the API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.settings.model_name}:generateContent",
                    headers={
                        "Content-Type": "application/json",
                        "X-goog-api-key": self.settings.api_key
                    },
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                return self._parse_response(result)
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Gemini API error: {e.response.status_code}")
        except Exception as e:
            self.logger.error(f"Gemini API error: {str(e)}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def extract_medical_info(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Extract medical information from conversation using Gemini.
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            
        Returns:
            Dictionary with extracted information
        """
        # Build a comprehensive prompt for medical extraction
        prompt = self._build_medical_extraction_prompt(conversation_text, questions)
        
        # Generate content with medical context
        response = await self.generate_content(
            prompt=prompt,
            system_prompt=self._get_medical_system_prompt(),
            temperature=0.1,  # Low temperature for consistent medical extraction
            max_tokens=2048
        )
        
        # Parse the response to extract structured data
        return self._parse_medical_response(response.text, questions)
    
    async def summarize_conversation(
        self,
        conversation_text: str,
        summary_type: str = "medical"
    ) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            conversation_text: The transcribed conversation
            summary_type: Type of summary (medical, general, etc.)
            
        Returns:
            Summary text
        """
        prompt = f"""
        Please provide a {summary_type} summary of the following nurse-patient conversation.
        Focus on key medical information, patient status, and care decisions.
        
        Conversation:
        {conversation_text}
        
        Summary:
        """
        
        response = await self.generate_content(
            prompt=prompt,
            system_prompt="You are a medical transcription specialist. Provide clear, concise medical summaries.",
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.text.strip()
    
    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Build the API request payload."""
        parts = []
        
        # Add system prompt if provided
        if system_prompt:
            parts.append({"text": system_prompt})
        
        # Add user prompt
        parts.append({"text": prompt})
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": kwargs.get("top_p", self.settings.top_p),
                "topK": kwargs.get("top_k", self.settings.top_k)
            }
        }
        
        return payload
    
    def _parse_response(self, response_data: Dict[str, Any]) -> GeminiResponse:
        """Parse the API response."""
        try:
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise Exception("No candidates in response")
            
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise Exception("No content parts in response")
            
            text = parts[0].get("text", "")
            
            # Extract usage information if available
            usage = response_data.get("usageMetadata", {})
            
            return GeminiResponse(
                text=text,
                usage=usage,
                model=self.settings.model_name,
                finish_reason=candidate.get("finishReason", "STOP")
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {str(e)}")
            raise Exception(f"Failed to parse Gemini response: {str(e)}")
    
    def _build_medical_extraction_prompt(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]]
    ) -> str:
        """Build prompt for medical information extraction."""
        question_text = "\n".join([
            f"- {q['question']}" for q in questions
        ])
        
        return f"""
        Please analyze the following nurse-patient conversation and extract specific medical information.
        
        Conversation:
        {conversation_text}
        
        Please answer these questions based on the conversation:
        {question_text}
        
        Provide your answers in a structured format. If information is not available, indicate "Not mentioned" or "Not available".
        """
    
    def _get_medical_system_prompt(self) -> str:
        """Get system prompt for medical context."""
        return """
        You are a medical information extraction specialist. Your role is to:
        1. Carefully analyze medical conversations
        2. Extract relevant medical information accurately
        3. Provide clear, concise answers
        4. Maintain medical terminology and accuracy
        5. Indicate when information is not available
        6. Focus on patient safety and care quality
        """
    
    def _parse_medical_response(
        self,
        response_text: str,
        questions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Parse medical extraction response into structured data."""
        try:
            # Debug logging to see the actual response
            self.logger.info(f"Raw Gemini response text: {response_text}")
            self.logger.info(f"Questions to parse: {questions}")
            
            # Try to extract structured information from the response
            extracted_info = {}
            
            # Parse Gemini's formatted response
            lines = response_text.split('\n')
            self.logger.info(f"Parsing {len(lines)} lines from response")
            
            # Create a mapping of question patterns to question IDs
            question_patterns = {}
            for q in questions:
                # Extract key words from the question to match against Gemini's response
                question_lower = q['question'].lower()
                if 'patient' in question_lower and 'name' in question_lower:
                    question_patterns['patient.*name|name.*patient'] = q['id']
                elif 'patient' in question_lower and 'age' in question_lower:
                    question_patterns['patient.*age|age.*patient'] = q['id']
                elif 'chief complaint' in question_lower or 'main problem' in question_lower:
                    question_patterns['chief complaint|main problem'] = q['id']
                elif 'symptoms' in question_lower:
                    question_patterns['symptoms'] = q['id']
                elif 'medications' in question_lower and 'taking' in question_lower:
                    question_patterns['medications.*taking|medications'] = q['id']
                elif 'allergies' in question_lower:
                    question_patterns['allergies'] = q['id']
                elif 'vital signs' in question_lower:
                    question_patterns['vital signs'] = q['id']
                elif 'pain level' in question_lower:
                    question_patterns['pain level'] = q['id']
                elif 'diagnosis' in question_lower:
                    question_patterns['diagnosis'] = q['id']
                elif 'treatment plan' in question_lower:
                    question_patterns['treatment plan'] = q['id']
                elif 'discharge instructions' in question_lower:
                    question_patterns['discharge instructions'] = q['id']
                elif 'follow.up' in question_lower or 'follow up' in question_lower:
                    question_patterns['follow.up|follow up'] = q['id']
                elif 'vital signs.*checked' in question_lower:
                    question_patterns['vital signs.*checked'] = q['id']
                elif 'medications.*reviewed' in question_lower:
                    question_patterns['medications.*reviewed'] = q['id']
                elif 'exercises.*performed' in question_lower or 'exercises.*demonstrated' in question_lower:
                    question_patterns['exercises.*performed|exercises.*demonstrated|exercises'] = q['id']
                elif 'patient.*concerns' in question_lower:
                    question_patterns['patient.*concerns'] = q['id']
            
            self.logger.info(f"Question patterns: {question_patterns}")
            
            # Process each line to find answers
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                self.logger.info(f"Processing line: '{line}'")
                
                # Look for patterns that match our questions
                for pattern, question_id in question_patterns.items():
                    import re
                    if re.search(pattern, line, re.IGNORECASE):
                        self.logger.info(f"Found pattern match: {pattern} -> {question_id}")
                        
                        # Extract the answer after the colon
                        if ':' in line:
                            answer = line.split(':', 1)[1].strip()
                            # Remove markdown formatting
                            answer = answer.replace('*', '').replace('**', '').strip()
                            
                            if answer and answer.lower() not in ['not mentioned', 'not available', 'n/a']:
                                extracted_info[question_id] = answer
                                self.logger.info(f"Added answer for {question_id}: {answer}")
                            else:
                                extracted_info[question_id] = "Information not available in conversation"
                                self.logger.info(f"Added 'not available' for {question_id}")
                        break
            
            self.logger.info(f"Final extracted info: {extracted_info}")
            return extracted_info
            
        except Exception as e:
            self.logger.error(f"Error parsing medical response: {str(e)}")
            # Return raw response if parsing fails
            return {"raw_response": response_text, "parsing_error": str(e)}
