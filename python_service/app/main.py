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
from .extract_logic import process_consistency_check, extract_fields_from_file
import uvicorn
import httpx
from typing import Dict, Any
from app.metrics import MetricsCollector

# Configure logging levels for different components
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.ERROR)
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
    # Initialize metrics collection
    metrics = MetricsCollector("consistency_check")
    
    try:
        # Add document info
        plankart_content = await plankart.read()
        metrics.add_document_info("plankart", plankart.filename, size=len(plankart_content))
        await plankart.seek(0)  # Reset file position after reading
        
        bestemmelser_content = await bestemmelser.read()
        metrics.add_document_info("bestemmelser", bestemmelser.filename, size=len(bestemmelser_content))
        await bestemmelser.seek(0)  # Reset file position after reading
        
        if sosi:
            sosi_content = await sosi.read()
            metrics.add_document_info("sosi", sosi.filename, size=len(sosi_content))
            await sosi.seek(0)  # Reset file position after reading
        
        # Extract fields from plankart
        plankart_start = metrics.start_timer("plankart_extraction")
        plankart_fields = await extract_fields_from_file(plankart)
        metrics.stop_timer("plankart_extraction", plankart_start)
        metrics.record_field_count("plankart", plankart_fields)
        metrics.record_fields("plankart", plankart_fields)
        
        # Extract fields from bestemmelser
        bestemmelser_start = metrics.start_timer("bestemmelser_extraction")
        bestemmelser_fields = await extract_fields_from_file(bestemmelser)
        metrics.stop_timer("bestemmelser_extraction", bestemmelser_start)
        metrics.record_field_count("bestemmelser", bestemmelser_fields)
        metrics.record_fields("bestemmelser", bestemmelser_fields)
        
        # Extract fields from SOSI if provided
        sosi_fields = set()
        if sosi:
            sosi_start = metrics.start_timer("sosi_extraction")
            sosi_fields = await extract_fields_from_file(sosi)
            metrics.stop_timer("sosi_extraction", sosi_start)
            metrics.record_field_count("sosi", sosi_fields)
            metrics.record_fields("sosi", sosi_fields)
        
        # Perform consistency check
        consistency_start = metrics.start_timer("consistency_check")
        result = await process_consistency_check(plankart, bestemmelser, sosi)
        metrics.stop_timer("consistency_check", consistency_start)
        
        # Record field matching results
        metrics.record_field_count("matching", set(result.matching_fields))
        metrics.record_field_count("only_in_plankart", set(result.only_in_plankart))
        metrics.record_field_count("only_in_bestemmelser", set(result.only_in_bestemmelser))
        metrics.record_field_count("only_in_sosi", set(result.only_in_sosi))
        
        metrics.record_fields("matching", set(result.matching_fields))
        metrics.record_fields("only_in_plankart", set(result.only_in_plankart))
        metrics.record_fields("only_in_bestemmelser", set(result.only_in_bestemmelser))
        metrics.record_fields("only_in_sosi", set(result.only_in_sosi))
        
        # Add consistency result
        metrics.metrics["is_consistent"] = result.is_consistent
        
        # Finalize metrics
        metrics.finalize()
        
        return result
    except Exception as e:
        # Record error in metrics
        metrics.record_error(e)
        
        # Re-raise the exception
        raise

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