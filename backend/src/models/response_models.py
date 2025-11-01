"""
Pydantic models for API responses
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TextToPromptResponse(BaseModel):
    """Response model for text-to-prompt conversion"""
    success: bool = Field(..., description="Whether the operation was successful")
    original_text: str = Field(..., description="Original user input text")
    enhanced_text: Optional[str] = Field(None, description="Enhanced/developed text (if develop_mode was True)")
    sora_prompt: str = Field(..., description="Final Sora 2 optimized prompt")
    develop_mode_used: bool = Field(..., description="Whether text enhancement was applied")
    processing_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "original_text": "A girl walking in the park with her dog",
                "enhanced_text": "A young woman in casual summer clothes walks leisurely through a sun-dappled park...",
                "sora_prompt": "Wide tracking shot of a young woman in casual attire walking with a golden retriever...",
                "develop_mode_used": True,
                "processing_time_ms": 2350.5
            }
        }


class DirectSoraPromptResponse(BaseModel):
    """Response model for direct Sora prompt generation"""
    success: bool = Field(..., description="Whether the operation was successful")
    original_text: str = Field(..., description="Original user input text")
    sora_prompt: str = Field(..., description="Generated Sora 2 prompt")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "original_text": "A cinematic scene of a woman walking through a misty forest",
                "sora_prompt": "Cinematic wide shot tracking a woman walking through dense misty forest at dawn...",
                "processing_time_ms": 1250.3
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Always False for errors")
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "AZURE_API_ERROR",
                "message": "Failed to communicate with Azure AI service",
                "timestamp": "2025-10-26T12:34:56.789Z"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    azure_configured: bool = Field(..., description="Whether Azure AI is properly configured")

