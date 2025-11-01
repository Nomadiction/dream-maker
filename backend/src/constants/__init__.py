"""
Constants package
"""
from .azure_config import (
    AZURE_ENDPOINT,
    MODEL_NAME,
    AZURE_API_KEY,
    ModelParameters
)
from .prompts import PromptTemplates

__all__ = [
    "AZURE_ENDPOINT",
    "MODEL_NAME",
    "AZURE_API_KEY",
    "ModelParameters",
    "PromptTemplates",
]

