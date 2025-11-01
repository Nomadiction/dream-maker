"""
Dream Maker API - Main Application
FastAPI backend for text-to-Sora-prompt conversion using Azure AI
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from azure.core.exceptions import AzureError

from .models import (
    TextToPromptRequest,
    DirectSoraPromptRequest,
    CustomPromptRequest,
    TextToPromptResponse,
    DirectSoraPromptResponse,
    ErrorResponse,
    HealthCheckResponse
)
from .service import AzureAIService
from .constants import AZURE_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance
azure_ai_service: AzureAIService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global azure_ai_service
    
    # Startup
    logger.info("Starting Dream Maker API...")
    try:
        azure_ai_service = AzureAIService()
        logger.info("Azure AI Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure AI Service: {e}")
        # Continue anyway, errors will be handled in endpoints
    
    yield
    
    # Shutdown
    logger.info("Shutting down Dream Maker API...")


# Initialize FastAPI app
app = FastAPI(
    title="Dream Maker API",
    description="AI-powered text-to-Sora-prompt conversion service using Azure AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(AzureError)
async def azure_error_handler(request, exc: AzureError):
    """Handle Azure AI errors"""
    logger.error(f"Azure AI error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="AZURE_API_ERROR",
            message=f"Azure AI service error: {str(exc)}"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message=f"An unexpected error occurred: {str(exc)}"
        ).model_dump()
    )


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """Check if the service is running and Azure AI is configured"""
    azure_configured = bool(AZURE_API_KEY) and azure_ai_service is not None
    
    return HealthCheckResponse(
        status="healthy" if azure_configured else "degraded",
        service="Dream Maker API",
        azure_configured=azure_configured
    )


# Main endpoint: Text to Sora Prompt with optional enhancement
@app.post(
    "/api/text-to-prompt",
    response_model=TextToPromptResponse,
    tags=["Prompt Generation"],
    summary="Convert user text to Sora 2 prompt",
    description="""
    Main endpoint for converting simple user text into professional Sora 2 prompts.
    
    Process:
    1. If develop_mode=True: AI enhances and develops the text with details
    2. AI generates optimized Sora 2 video generation prompt
    
    The develop_mode option allows users to toggle text enhancement on/off.
    """
)
async def text_to_prompt(request: TextToPromptRequest):
    """
    Convert user text to Sora prompt with optional text enhancement
    
    Args:
        request: TextToPromptRequest with user_text and develop_mode
        
    Returns:
        TextToPromptResponse with enhanced text (if applicable) and Sora prompt
    """
    if azure_ai_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Azure AI service is not initialized"
        )
    
    try:
        logger.info(f"Processing text-to-prompt request (develop_mode={request.develop_mode})")
        
        enhanced_text, sora_prompt, processing_time = await run_in_threadpool(
            azure_ai_service.process_text_to_sora_prompt,
            request.user_text,
            request.develop_mode
        )
        
        logger.info(f"Successfully generated Sora prompt in {processing_time:.2f}ms")
        
        return TextToPromptResponse(
            success=True,
            original_text=request.user_text,
            enhanced_text=enhanced_text,
            sora_prompt=sora_prompt,
            develop_mode_used=request.develop_mode,
            processing_time_ms=processing_time
        )
        
    except AzureError as e:
        logger.error(f"Azure AI error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Azure AI service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing text-to-prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )


# Direct Sora prompt generation (skip enhancement)
@app.post(
    "/api/direct-sora-prompt",
    response_model=DirectSoraPromptResponse,
    tags=["Prompt Generation"],
    summary="Generate Sora prompt directly",
    description="""
    Generate Sora 2 prompt directly from detailed user text, skipping the enhancement stage.
    
    Use this endpoint when:
    - User already has detailed description
    - Want to save processing time
    - Text is already well-structured
    """
)
async def direct_sora_prompt(request: DirectSoraPromptRequest):
    """
    Generate Sora prompt directly without text enhancement
    
    Args:
        request: DirectSoraPromptRequest with user_text
        
    Returns:
        DirectSoraPromptResponse with Sora prompt
    """
    if azure_ai_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Azure AI service is not initialized"
        )
    
    try:
        logger.info("Processing direct-sora-prompt request")
        
        sora_prompt, processing_time = await run_in_threadpool(
            azure_ai_service.direct_sora_prompt,
            request.user_text
        )
        
        logger.info(f"Successfully generated Sora prompt in {processing_time:.2f}ms")
        
        return DirectSoraPromptResponse(
            success=True,
            original_text=request.user_text,
            sora_prompt=sora_prompt,
            processing_time_ms=processing_time
        )
        
    except AzureError as e:
        logger.error(f"Azure AI error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Azure AI service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing direct-sora-prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )


# Custom prompt endpoint (advanced usage)
@app.post(
    "/api/custom-prompt",
    tags=["Advanced"],
    summary="Custom Azure AI completion",
    description="""
    Advanced endpoint for custom Azure AI completions with user-defined messages.
    
    Allows full control over:
    - System and user messages
    - Temperature and sampling parameters
    - Token limits
    """
)
async def custom_prompt(request: CustomPromptRequest):
    """
    Execute custom Azure AI completion
    
    Args:
        request: CustomPromptRequest with messages and parameters
        
    Returns:
        Raw Azure AI response
    """
    if azure_ai_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Azure AI service is not initialized"
        )
    
    try:
        logger.info("Processing custom-prompt request")
        
        response = await run_in_threadpool(
            azure_ai_service.custom_completion,
            request.messages,
            request.max_tokens,
            request.temperature,
            request.top_p
        )
        
        logger.info("Successfully completed custom prompt")
        
        return {"success": True, "response": response}
        
    except AzureError as e:
        logger.error(f"Azure AI error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Azure AI service error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing custom-prompt: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Dream Maker API",
        "version": "1.0.0",
        "description": "AI-powered text-to-Sora-prompt conversion service",
        "endpoints": {
            "health": "/health",
            "text_to_prompt": "/api/text-to-prompt",
            "direct_sora_prompt": "/api/direct-sora-prompt",
            "custom_prompt": "/api/custom-prompt",
            "docs": "/docs"
        }
    }


# Helper function for running blocking operations
async def run_in_threadpool(func, *args, **kwargs):
    """Run blocking function in thread pool"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

