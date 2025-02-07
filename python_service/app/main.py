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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .extract_logic import check_field_consistency, process_consistency_check
import uvicorn

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
        "http://backend:5251"        # Backend in Docker
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        # Validate file types
        for file in [plankart, bestemmelser]:
            if file and not file.content_type == 'application/pdf':
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type for {file.filename}. Must be PDF."
                )

        logger.info(f"Processing files: {plankart.filename}, {bestemmelser.filename}")
        
        # Reset file positions
        await plankart.seek(0)
        await bestemmelser.seek(0)
        if sosi:
            await sosi.seek(0)
            
        # Call the extract_logic function directly
        result = await process_consistency_check(plankart, bestemmelser, sosi)
        logger.info("Field consistency check completed successfully")
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"Error during consistency check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)