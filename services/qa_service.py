"""
Unified Question Answering service supporting multiple AI providers.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from .gemini_service import GeminiService
from .openai_service import OpenAIService
from config import GeminiSettings, OpenAISettings


class AIProvider(str, Enum):
    """Available AI providers for Q&A extraction."""
    GEMINI = "gemini"
    OPENAI = "openai"
    DISTILBERT = "distilbert"


class QAService:
    """Unified QA service supporting multiple AI providers."""
    
    def __init__(
        self,
        gemini_settings: Optional[GeminiSettings] = None,
        openai_settings: Optional[OpenAISettings] = None,
        default_provider: AIProvider = AIProvider.GEMINI
    ):
        self.default_provider = default_provider
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.gemini_service = None
        self.openai_service = None
        
        if gemini_settings and gemini_settings.api_key:
            self.gemini_service = GeminiService(gemini_settings)
            self.logger.info("Gemini service initialized")
        
        if openai_settings and openai_settings.api_key:
            self.openai_service = OpenAIService(openai_settings)
            self.logger.info("OpenAI service initialized")
        
        if not self.gemini_service and not self.openai_service:
            self.logger.warning("No AI services initialized - API keys required")
    
    async def extract_medical_info(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]],
        provider: Optional[AIProvider] = None,
        fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Extract medical information using specified or default provider.
        
        Args:
            conversation_text: The transcribed conversation
            questions: List of questions to extract answers for
            provider: Specific AI provider to use
            fallback: Whether to fallback to other provider if primary fails
            
        Returns:
            Dictionary with extracted information
        """
        provider = provider or self.default_provider
        
        try:
            if provider == AIProvider.GEMINI and self.gemini_service:
                return await self.gemini_service.extract_medical_info(
                    conversation_text, questions
                )
            elif provider == AIProvider.OPENAI and self.openai_service:
                return await self.openai_service.extract_medical_info(
                    conversation_text, questions
                )
            else:
                raise Exception(f"Provider {provider} not available")
                
        except Exception as e:
            self.logger.error(f"Primary provider {provider} failed: {str(e)}")
            
            if fallback:
                return await self._fallback_extraction(
                    conversation_text, questions, provider
                )
            else:
                raise e
    
    async def summarize_conversation(
        self,
        conversation_text: str,
        summary_type: str = "medical",
        provider: Optional[AIProvider] = None,
        fallback: bool = True
    ) -> str:
        """
        Generate conversation summary using specified or default provider.
        
        Args:
            conversation_text: The transcribed conversation
            summary_type: Type of summary to generate
            provider: Specific AI provider to use
            fallback: Whether to fallback to other provider if primary fails
            
        Returns:
            Summary text
        """
        provider = provider or self.default_provider
        
        try:
            if provider == AIProvider.GEMINI and self.gemini_service:
                return await self.gemini_service.summarize_conversation(
                    conversation_text, summary_type
                )
            elif provider == AIProvider.OPENAI and self.openai_service:
                return await self.openai_service.summarize_conversation(
                    conversation_text, summary_type
                )
            else:
                raise Exception(f"Provider {provider} not available")
                
        except Exception as e:
            self.logger.error(f"Primary provider {provider} failed: {str(e)}")
            
            if fallback:
                return await self._fallback_summary(
                    conversation_text, summary_type, provider
                )
            else:
                raise e
    
    async def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[AIProvider] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using specified or default provider.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instruction
            provider: Specific AI provider to use
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary with generated content and metadata
        """
        provider = provider or self.default_provider
        
        try:
            if provider == AIProvider.GEMINI and self.gemini_service:
                response = await self.gemini_service.generate_content(
                    prompt, system_prompt, **kwargs
                )
                return {
                    "text": response.text,
                    "provider": "gemini",
                    "model": response.model,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason
                }
            elif provider == AIProvider.OPENAI and self.openai_service:
                response = await self.openai_service.generate_content(
                    prompt, system_prompt, **kwargs
                )
                return {
                    "text": response.text,
                    "provider": "openai",
                    "model": response.model,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason
                }
            else:
                raise Exception(f"Provider {provider} not available")
                
        except Exception as e:
            self.logger.error(f"Content generation failed with {provider}: {str(e)}")
            raise e
    
    async def _fallback_extraction(
        self,
        conversation_text: str,
        questions: List[Dict[str, str]],
        failed_provider: AIProvider
    ) -> Dict[str, Any]:
        """Fallback to alternative provider for medical extraction."""
        fallback_provider = (
            AIProvider.OPENAI if failed_provider == AIProvider.GEMINI 
            else AIProvider.GEMINI
        )
        
        self.logger.info(f"Falling back to {fallback_provider} for medical extraction")
        
        try:
            if fallback_provider == AIProvider.GEMINI and self.gemini_service:
                return await self.gemini_service.extract_medical_info(
                    conversation_text, questions
                )
            elif fallback_provider == AIProvider.OPENAI and self.openai_service:
                return await self.openai_service.extract_medical_info(
                    conversation_text, questions
                )
            else:
                raise Exception(f"Fallback provider {fallback_provider} not available")
                
        except Exception as e:
            self.logger.error(f"Fallback provider {fallback_provider} also failed: {str(e)}")
            raise Exception(f"All AI providers failed: {str(e)}")
    
    async def _fallback_summary(
        self,
        conversation_text: str,
        summary_type: str,
        failed_provider: AIProvider
    ) -> str:
        """Fallback to alternative provider for conversation summary."""
        fallback_provider = (
            AIProvider.OPENAI if failed_provider == AIProvider.GEMINI 
            else AIProvider.GEMINI
        )
        
        self.logger.info(f"Falling back to {fallback_provider} for conversation summary")
        
        try:
            if fallback_provider == AIProvider.GEMINI and self.gemini_service:
                return await self.gemini_service.summarize_conversation(
                    conversation_text, summary_type
                )
            elif fallback_provider == AIProvider.OPENAI and self.openai_service:
                return await self.openai_service.summarize_conversation(
                    conversation_text, summary_type
                )
            else:
                raise Exception(f"Fallback provider {fallback_provider} not available")
                
        except Exception as e:
            self.logger.error(f"Fallback provider {fallback_provider} also failed: {str(e)}")
            raise Exception(f"All AI providers failed: {str(e)}")
    
    def get_available_providers(self) -> List[AIProvider]:
        """Get list of available AI providers."""
        providers = []
        if self.gemini_service:
            providers.append(AIProvider.GEMINI)
        if self.openai_service:
            providers.append(AIProvider.OPENAI)
        return providers
    
    def is_provider_available(self, provider: AIProvider) -> bool:
        """Check if a specific provider is available."""
        if provider == AIProvider.GEMINI:
            return self.gemini_service is not None
        elif provider == AIProvider.OPENAI:
            return self.openai_service is not None
        return False
