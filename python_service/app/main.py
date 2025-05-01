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

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from .extract_logic import process_consistency_check, extract_fields_from_file
import uvicorn
import httpx
from typing import Dict, Any
from app.metrics import MetricsCollector
from app.utils.logger import logger
import os

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
    logger.info(f"Received consistency check request - Files: plankart={plankart.filename}, bestemmelser={bestemmelser.filename}, sosi={sosi.filename if sosi else 'None'}")
    
    try:
        # Start metrics collection
        metrics = MetricsCollector("consistency_check")
        
        # Validate file types
        if not plankart.content_type == 'application/pdf':
            raise HTTPException(status_code=400, detail=f"Invalid plankart file type: {plankart.content_type}")
        if not bestemmelser.content_type == 'application/pdf':
            raise HTTPException(status_code=400, detail=f"Invalid bestemmelser file type: {bestemmelser.content_type}")
        
        # Read and validate file contents
        plankart_content = await plankart.read()
        if not plankart_content:
            raise HTTPException(status_code=400, detail="Plankart file is empty")
        metrics.add_document_info("plankart", plankart.filename, size=len(plankart_content))
        
        bestemmelser_content = await bestemmelser.read()
        if not bestemmelser_content:
            raise HTTPException(status_code=400, detail="Bestemmelser file is empty")
        metrics.add_document_info("bestemmelser", bestemmelser.filename, size=len(bestemmelser_content))
        
        # Reset file positions
        await plankart.seek(0)
        await bestemmelser.seek(0)
        
        if sosi:
            sosi_content = await sosi.read()
            if not sosi_content:
                raise HTTPException(status_code=400, detail="SOSI file is empty")
            metrics.add_document_info("sosi", sosi.filename, size=len(sosi_content))
            await sosi.seek(0)
        
        # Perform consistency check
        consistency_start = metrics.start_timer("consistency_check")
        try:
            result = await process_consistency_check(plankart, bestemmelser, sosi)
            metrics.stop_timer("consistency_check", consistency_start)
            
            # Record results in metrics
            if hasattr(result, 'matching_fields'):
                metrics.record_field_count("matching", set(result.matching_fields))
                metrics.record_field_count("only_in_plankart", set(result.only_in_plankart))
                metrics.record_field_count("only_in_bestemmelser", set(result.only_in_bestemmelser))
                metrics.record_field_count("only_in_sosi", set(result.only_in_sosi))
                
                metrics.record_fields("matching", set(result.matching_fields))
                metrics.record_fields("only_in_plankart", set(result.only_in_plankart))
                metrics.record_fields("only_in_bestemmelser", set(result.only_in_bestemmelser))
                metrics.record_fields("only_in_sosi", set(result.only_in_sosi))
                
                metrics.metrics["is_consistent"] = result.is_consistent
            
            # Record metrics
            metrics.finalize()
            
            logger.info(f"Successfully processed consistency check")
            return result
            
        except Exception as e:
            logger.error(f"Error during consistency check: {str(e)}", exc_info=True)
            metrics.record_error(e)
            raise HTTPException(status_code=500, detail=str(e))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    try:
        async with httpx.AsyncClient() as client:
            # Check NER service health
            ner_url = os.getenv('NER_SERVICE_URL', 'http://157.230.21.199:8001')
            ner_response = await client.get(f"{ner_url}/health")
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
    """Log frontend messages"""
    logger.info(f"Frontend log: {log_data}")
    try:
        async with httpx.AsyncClient() as client:
            # Forward to .NET backend
            response = await client.post('http://backend:5000/api/log', json=log_data)
            response.raise_for_status()
            
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Error forwarding log: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)