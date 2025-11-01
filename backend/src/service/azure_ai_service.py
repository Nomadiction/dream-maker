"""
Azure AI Service for text processing and prompt generation
"""
import time
from typing import Optional, Tuple
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from ..constants.azure_config import (
    AZURE_ENDPOINT,
    MODEL_NAME,
    AZURE_API_KEY,
    ModelParameters
)
from ..constants.prompts import PromptTemplates


class AzureAIService:
    """Service for interacting with Azure AI"""
    
    def __init__(self):
        """Initialize Azure AI client"""
        if not AZURE_API_KEY:
            raise ValueError("AZURE_API_KEY is not set. Please configure environment variables.")
        
        self.client = ChatCompletionsClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_API_KEY)
        )
        self.model_name = MODEL_NAME
        self.prompt_templates = PromptTemplates()
    
    def _call_azure_ai(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """
        Make a call to Azure AI with specified parameters
        
        Args:
            system_prompt: System message for AI
            user_message: User message/question
            max_tokens: Maximum tokens in response
            temperature: Temperature parameter
            top_p: Top P parameter
            
        Returns:
            AI response content as string
            
        Raises:
            AzureError: If Azure AI call fails
        """
        try:
            response = self.client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_message)
                ],
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Extract content from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("Unexpected response format from Azure AI")
                
        except AzureError as e:
            raise AzureError(f"Azure AI API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in Azure AI call: {str(e)}")
    
    def enhance_text(self, user_text: str, develop_mode: bool) -> str:
        """
        Enhance user text with AI
        
        Args:
            user_text: Original user input
            develop_mode: Whether to significantly develop the text
            
        Returns:
            Enhanced text
        """
        system_prompt = self.prompt_templates.TEXT_ENHANCEMENT_SYSTEM_PROMPT
        user_message = self.prompt_templates.get_enhancement_user_message(
            user_text, 
            develop_mode
        )
        
        enhanced_text = self._call_azure_ai(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=ModelParameters.ENHANCEMENT_MAX_TOKENS,
            temperature=ModelParameters.ENHANCEMENT_TEMPERATURE,
            top_p=ModelParameters.ENHANCEMENT_TOP_P
        )
        
        return enhanced_text.strip()
    
    def generate_sora_prompt(self, text: str) -> str:
        """
        Generate Sora 2 cinematic JSON prompt from text
        
        Args:
            text: Input text (can be original or enhanced)
            
        Returns:
            Sora 2 optimized cinematic JSON prompt
        """
        system_prompt = self.prompt_templates.CINEMATIC_JSON_SYSTEM_PROMPT
        user_message = self.prompt_templates.get_cinematic_json_user_message(text)
        
        sora_prompt = self._call_azure_ai(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=ModelParameters.SORA_PROMPT_MAX_TOKENS,
            temperature=ModelParameters.TEMPERATURE,
            top_p=ModelParameters.TOP_P
        )
        
        return sora_prompt.strip()
    
    def process_text_to_sora_prompt(
        self, 
        user_text: str, 
        develop_mode: bool
    ) -> Tuple[Optional[str], str, float]:
        """
        Full pipeline: enhance text (optional) + generate Sora prompt
        
        Args:
            user_text: Original user input
            develop_mode: Whether to enhance text before generating prompt
            
        Returns:
            Tuple of (enhanced_text or None, sora_prompt, processing_time_ms)
        """
        start_time = time.time()
        
        enhanced_text = None
        
        if develop_mode:
            # Stage 1: Enhance text
            enhanced_text = self.enhance_text(user_text, develop_mode)
            text_for_prompt = enhanced_text
        else:
            # Skip enhancement, use original text
            text_for_prompt = user_text
        
        # Stage 2: Generate Sora prompt
        sora_prompt = self.generate_sora_prompt(text_for_prompt)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return enhanced_text, sora_prompt, processing_time_ms
    
    def direct_sora_prompt(self, user_text: str) -> Tuple[str, float]:
        """
        Generate Sora prompt directly without text enhancement
        
        Args:
            user_text: User's detailed text
            
        Returns:
            Tuple of (sora_prompt, processing_time_ms)
        """
        start_time = time.time()
        
        sora_prompt = self.generate_sora_prompt(user_text)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return sora_prompt, processing_time_ms
    
    def custom_completion(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> dict:
        """
        Custom Azure AI completion with user-defined messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens
            temperature: Temperature parameter
            top_p: Top P parameter
            
        Returns:
            Full response from Azure AI
        """
        try:
            # Convert dict messages to Azure AI message objects
            azure_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    azure_messages.append(SystemMessage(content=content))
                else:
                    azure_messages.append(UserMessage(content=content))
            
            response = self.client.complete(
                messages=azure_messages,
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            return response
            
        except AzureError as e:
            raise AzureError(f"Azure AI API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def check_health(self) -> bool:
        """
        Check if Azure AI service is accessible
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Simple health check call
            self._call_azure_ai(
                system_prompt="You are a helpful assistant.",
                user_message="Hello",
                max_tokens=10,
                temperature=0.1,
                top_p=0.1
            )
            return True
        except Exception:
            return False

