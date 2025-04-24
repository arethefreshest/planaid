from fastapi import FastAPI, UploadFile, HTTPException, Request
from pathlib import Path
import tempfile
import os
from inference import (
    get_text, 
    get_sent_tokens, 
    get_predictions, 
    process_text_with_spacy, 
    initialize_models,
    model_manager
)
import time
import json
import psutil
from datetime import datetime
from src.utils.logger import setup_logger
from pydantic_settings import BaseSettings
import logging

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Then create specific logger
logger = setup_logger("api", os.getenv("LOG_LEVEL", "INFO"))

class Settings(BaseSettings):
    # Model Configuration
    MODEL_PATH: str = "src/models/nb-bert-base.pth"
    CONFIG_PATH: str = "src/model_params.yaml"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Metrics
    METRICS_DIR: str = "metrics"
    
    # Environment
    IS_DOCKER: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "IS_DOCKER":
                return os.getenv("DOTNET_RUNNING_IN_CONTAINER") == "true"
            return raw_val

settings = Settings()

app = FastAPI()

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    
    logger.info(f"Request: {method} {path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {method} {path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

logger.info("Starting NER service initialization")

# Ensure metrics directory exists
os.makedirs(settings.METRICS_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    try:
        logger.info("Starting NER service initialization")
        
        # Initialize models
        success = initialize_models()
        if not success:
            logger.error("Failed to initialize models during startup")
            # Don't raise an exception, let the service start but it will return errors for requests
        else:
            logger.info("NER service initialization completed successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize NER service: {str(e)}")
        # Don't raise the exception, let the service start but it will return errors for requests

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    try:
        logger.info("Shutting down NER service")
        # Add any cleanup code here if needed
        logger.info("NER service shutdown completed")
    except Exception as e:
        logger.error(f"Error during NER service shutdown: {str(e)}")

@app.post("/api/extract-fields")
async def extract_fields(file: UploadFile):
    """Extract fields from a PDF document"""
    print(f"RECEIVED FILE: {file.filename}", flush=True)
    logger.info(f"=== RECEIVED REQUEST: extract-fields for file {file.filename} ===")
    
    try:
        # Check if models are initialized
        if not model_manager.is_initialized():
            logger.error("Models not initialized")
            raise HTTPException(
                status_code=500, 
                detail="Service not properly initialized. Please check if model files exist and are accessible."
            )
        
        # Start metrics collection
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create a unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize metrics dictionary
        metrics = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "file": file.filename,
            "timings": {},
            "resource_usage": {}
        }
        
        logger.info(f"Processing file: {file.filename}")
        
        if not file.content_type == "application/pdf":
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(400, "Only PDF files are supported")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            if not content:
                logger.error("Empty file received")
                raise HTTPException(400, "File is empty")
            if len(content) < 100:  # Basic check for minimum PDF size
                logger.error(f"File too small to be a valid PDF: {len(content)} bytes")
                raise HTTPException(400, "File appears to be corrupted or invalid")
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            # Get file size
            metrics["file_size_kb"] = len(content) / 1024
            await file.seek(0)  # Reset file position
            
            # Extract text from PDF
            text_extraction_start = time.time()
            pdf_path = str(tmp_path)  # Use the temporary file path
            
            logger.info("Starting text extraction from PDF")
            try:
                text = get_text(pdf_path)
            except ValueError as e:
                logger.error(f"PDF extraction error: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to extract text from PDF: {str(e)}"
                )
            except Exception as e:
                logger.error(f"Unexpected error during PDF extraction: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error during PDF extraction: {str(e)}"
                )
            
            text_extraction_end = time.time()
            metrics["timings"]["text_extraction"] = text_extraction_end - text_extraction_start
            metrics["text_length"] = len(text)
            logger.info(f"Text extraction completed. Length: {len(text)} characters")
            
            # Process text with spaCy
            spacy_start = time.time()
            logger.info("Starting spaCy processing")
            try:
                sent_tokens = process_text_with_spacy(text, model_manager.nlp)
            except Exception as e:
                logger.error(f"Error during spaCy processing: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during text processing: {str(e)}"
                )
            spacy_end = time.time()
            metrics["timings"]["spacy_processing"] = spacy_end - spacy_start
            metrics["sentence_count"] = len(sent_tokens)
            logger.info(f"spaCy processing completed. Found {len(sent_tokens)} sentences")
            
            # Get predictions from model
            prediction_start = time.time()
            logger.info("Starting model predictions")
            try:
                fields = get_predictions(sent_tokens, model_manager.model, model_manager.tokenizer)
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error during field extraction: {str(e)}"
                )
            prediction_end = time.time()
            metrics["timings"]["model_prediction"] = prediction_end - prediction_start
            metrics["field_count"] = len(fields)
            logger.info(f"Model predictions completed. Found {len(fields)} fields")
            
            # Record end metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            metrics["timings"]["total_processing"] = end_time - start_time
            metrics["resource_usage"]["memory_mb"] = end_memory - start_memory
            metrics["resource_usage"]["cpu_percent"] = psutil.Process().cpu_percent()
            
            # Save metrics to file
            metrics_file = f"{settings.METRICS_DIR}/ner_extraction_{run_id}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Also save the extracted fields for later analysis
            fields_file = f"{settings.METRICS_DIR}/ner_fields_{run_id}.json"
            with open(fields_file, "w") as f:
                json.dump({"fields": fields}, f, indent=2)
            
            logger.debug("Starting field extraction process")
            logger.info(f"=== COMPLETED REQUEST: extract-fields for file {file.filename}, found {len(fields)} fields ===")
            print(f"COMPLETED PROCESSING: {file.filename} with {len(fields)} fields", flush=True)
            
            return {"fields": fields}
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
            # Record error in metrics
            metrics["error"] = str(e)
            
            # Save metrics even in case of error
            metrics_file = f"{settings.METRICS_DIR}/ner_extraction_error_{run_id}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Re-raise the exception with more detail
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.info("Field extraction completed successfully")
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
        raise

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"} 