"""
OpenAI API service for natural language processing.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import openai
from config import OpenAISettings


@dataclass
class OpenAIResponse:
    """Response from OpenAI API."""
    text: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str


class OpenAIService:
    """Service for interacting with OpenAI API."""
    
    def __init__(self, settings: OpenAISettings):
        self.settings = settings
        self.client = openai.AsyncOpenAI(api_key=settings.api_key)
        self.logger = logging.getLogger(__name__)
        
    async def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> OpenAIResponse:
        """
        Generate content using OpenAI API.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            temperature: Controls randomness (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            OpenAIResponse object
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.settings.model_name,
                messages=messages,
                temperature=temperature or self.settings.temperature,
                max_tokens=max_tokens or self.settings.max_tokens,
                **kwargs
            )
            
            return self._parse_response(response)
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def extract_medical_info(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Extract medical information from conversation using OpenAI.
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            
        Returns:
            Dictionary with extracted information
        """
        prompt = self._build_medical_extraction_prompt(conversation_text, questions)
        
        response = await self.generate_content(
            prompt=prompt,
            system_prompt=self._get_medical_system_prompt(),
            temperature=0.1,
            max_tokens=2048
        )
        
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
    
    def _parse_response(self, response) -> OpenAIResponse:
        """Parse the OpenAI API response."""
        try:
            choice = response.choices[0]
            
            return OpenAIResponse(
                text=choice.message.content,
                usage=response.usage.model_dump() if response.usage else {},
                model=response.model,
                finish_reason=choice.finish_reason
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing OpenAI response: {str(e)}")
            raise Exception(f"Failed to parse OpenAI response: {str(e)}")
    
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
            extracted_info = {}
            
            lines = response_text.split('\n')
            current_question = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line contains a question
                for q in questions:
                    if q['question'].lower() in line.lower():
                        current_question = q['id']
                        break
                
                # If we have a current question, extract the answer
                if current_question and ':' in line:
                    answer = line.split(':', 1)[1].strip()
                    if answer and answer.lower() not in ['not mentioned', 'not available', 'n/a']:
                        extracted_info[current_question] = answer
                        current_question = None
            
            return extracted_info
            
        except Exception as e:
            self.logger.error(f"Error parsing medical response: {str(e)}")
            return {"raw_response": response_text, "parsing_error": str(e)}
