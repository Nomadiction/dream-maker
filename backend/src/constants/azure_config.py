"""
Azure AI Configuration Constants
"""
import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Azure AI Endpoint Configuration
AZURE_ENDPOINT: Final[str] = os.getenv(
    "AZURE_INFERENCE_SDK_ENDPOINT"
)

# Model Configuration
MODEL_NAME: Final[str] = os.getenv(
    "DEPLOYMENT_NAME"
)

# API Key
AZURE_API_KEY: Final[str] = os.getenv(
    "AZURE_INFERENCE_SDK_KEY"
)

# Model Parameters
class ModelParameters:
    """Default parameters for Azure AI model"""
    MAX_TOKENS: Final[int] = 16000
    TEMPERATURE: Final[float] = 0.04
    TOP_P: Final[float] = 0.8
    
    # For text enhancement stage
    ENHANCEMENT_MAX_TOKENS: Final[int] = 4000
    ENHANCEMENT_TEMPERATURE: Final[float] = 0.16
    ENHANCEMENT_TOP_P: Final[float] = 0.32
    
    # For Sora prompt generation stage
    SORA_PROMPT_MAX_TOKENS: Final[int] = 32000
