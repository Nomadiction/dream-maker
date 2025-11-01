"""
Pydantic models for API requests
"""
from pydantic import BaseModel, Field
from typing import Optional


class TextToPromptRequest(BaseModel):
    """Request model for text-to-prompt conversion"""
    user_text: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="User's input text to be converted into Sora prompt",
        example="A cat walking on the street at sunset"
    )
    develop_mode: bool = Field(
        default=False,
        description="If True, AI will enhance and develop the text significantly. If False, minimal refinement only.",
        example=True
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_text": "A girl walking in the park with her dog",
                "develop_mode": True
            }
        }


class DirectSoraPromptRequest(BaseModel):
    """Request model for direct Sora prompt generation (skipping enhancement)"""
    user_text: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="Already detailed text to be converted directly into Sora prompt",
        example="A beautiful sunset scene with vibrant orange and pink clouds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_text": "A cinematic scene of a woman walking through a misty forest at dawn, sunlight filtering through tall trees"
            }
        }


class CustomPromptRequest(BaseModel):
    """Request model for custom AI prompts (advanced usage)"""
    messages: list[dict] = Field(
        ...,
        description="Custom messages array for Azure AI",
        example=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    )
    max_tokens: Optional[int] = Field(
        default=3000,
        ge=1,
        le=4000,
        description="Maximum tokens in response"
    )
    temperature: Optional[float] = Field(
        default=0.15,
        ge=0.0,
        le=2.0,
        description="Temperature for response randomness"
    )
    top_p: Optional[float] = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Top P sampling parameter"
    )

