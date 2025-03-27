from fastapi import FastAPI, UploadFile, HTTPException
from pathlib import Path
import tempfile
import os
from inference import get_text, get_sent_tokens, get_predictions, model, tokenizer, nlp, process_text_with_spacy
import logging
import time
import json
import psutil
from datetime import datetime

app = FastAPI()

logger = logging.getLogger(__name__)

# Ensure metrics directory exists
os.makedirs("metrics", exist_ok=True)

@app.post("/api/extract-fields")
async def extract_fields(file: UploadFile):
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
        raise HTTPException(400, "Only PDF files are supported")
        
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    try:
        # Get file size
        metrics["file_size_kb"] = len(content) / 1024
        await file.seek(0)  # Reset file position
        
        # Extract text from PDF
        text_extraction_start = time.time()
        pdf_path = f"/tmp/{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(content)
        
        text = get_text(pdf_path)
        text_extraction_end = time.time()
        metrics["timings"]["text_extraction"] = text_extraction_end - text_extraction_start
        metrics["text_length"] = len(text)
        
        # Process text with spaCy
        spacy_start = time.time()
        sent_tokens = process_text_with_spacy(text, nlp)
        spacy_end = time.time()
        metrics["timings"]["spacy_processing"] = spacy_end - spacy_start
        metrics["sentence_count"] = len(sent_tokens)
        
        # Get predictions from model
        prediction_start = time.time()
        fields = get_predictions(sent_tokens, model, tokenizer)
        prediction_end = time.time()
        metrics["timings"]["model_prediction"] = prediction_end - prediction_start
        metrics["field_count"] = len(fields)
        
        # Record end metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metrics["timings"]["total_processing"] = end_time - start_time
        metrics["resource_usage"]["memory_mb"] = end_memory - start_memory
        metrics["resource_usage"]["cpu_percent"] = psutil.Process().cpu_percent()
        
        # Save metrics to file
        metrics_file = f"metrics/ner_extraction_{run_id}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Also save the extracted fields for later analysis
        fields_file = f"metrics/ner_fields_{run_id}.json"
        with open(fields_file, "w") as f:
            json.dump({"fields": fields}, f, indent=2)
        
        # Clean up temporary file
        os.remove(pdf_path)
        
        return {"fields": fields}
    except Exception as e:
        # Record error in metrics
        metrics["error"] = str(e)
        
        # Save metrics even in case of error
        metrics_file = f"metrics/ner_extraction_error_{run_id}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Clean up temporary file if it exists
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        # Re-raise the exception
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 