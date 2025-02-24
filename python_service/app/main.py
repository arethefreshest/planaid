"""
FastAPI Application Entry Point

This module serves as the main entry point for the FastAPI application,
handling document uploads and field consistency checks for regulatory documents.

Features:
- CORS middleware configuration
- File upload endpoints
- Consistency checking logic
- Error handling and logging
"""

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from .extract_logic import process_consistency_check
import uvicorn
import httpx
from typing import Dict, Any

# Configure logging levels for different components
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.WARNING)
logging.getLogger("app").setLevel(logging.DEBUG)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Frontend local development
        "http://frontend:3000",      # Frontend in Docker
        "http://localhost:5251",     # Backend local development
        "http://backend:5251",       # Backend in Docker
        "*"                          # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.post("/api/check-field-consistency")
async def check_field_consistency(
    plankart: UploadFile = File(...),
    bestemmelser: UploadFile = File(...),
    sosi: UploadFile = None
):
    """
    Check field consistency between regulatory documents.
    
    Args:
        plankart (UploadFile): Plankart PDF document
        bestemmelser (UploadFile): Bestemmelser PDF document
        sosi (UploadFile, optional): SOSI file for additional validation
        
    Returns:
        dict: Consistency check results
        
    Raises:
        HTTPException: For invalid file types or processing errors
    """
    logger.info(f"Received files: plankart={plankart.filename}, bestemmelser={bestemmelser.filename}")
    try:
        # Reset file positions
        await plankart.seek(0)
        await bestemmelser.seek(0)
        if sosi:
            await sosi.seek(0)
            
        # Call the processing function
        result = await process_consistency_check(plankart, bestemmelser, sosi)
        logger.info("Field consistency check completed successfully")
        return {"status": "success", "result": result}
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions directly
    except Exception as e:
        logger.error(f"Error during consistency check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check health of both services"""
    try:
        async with httpx.AsyncClient() as client:
            # Check NER service health
            ner_response = await client.get("http://ner_service:8001/health")
            ner_response.raise_for_status()
            
            return {
                "status": "healthy",
                "ner_service": "connected"
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "ner_service": "disconnected",
            "error": str(e)
        }

@app.post("/api/log")
async def log_frontend(log_data: Dict[Any, Any] = Body(...)):
    """Forward frontend logs to .NET backend"""
    try:
        async with httpx.AsyncClient() as client:
            # Forward to .NET backend
            response = await client.post('http://backend:5000/api/log', json=log_data)
            response.raise_for_status()
            
        # Also log locally for debugging
        level = log_data.get('level', 'info').upper()
        message = log_data.get('message', '')
        data = log_data.get('data')
        
        log_message = f"Frontend: {message}"
        if data:
            log_message += f" Data: {data}"
            
        logger.info(log_message)  # Log all as info locally
            
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error forwarding log: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)