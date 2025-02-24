from fastapi import FastAPI, UploadFile, HTTPException
from pathlib import Path
import tempfile
import os
from inference import get_text, get_sent_tokens, get_predictions, model, tokenizer, nlp, process_text_with_spacy
import logging

app = FastAPI()

logger = logging.getLogger(__name__)

@app.post("/api/extract-fields")
async def extract_fields(file: UploadFile):
    logger.info(f"Processing file: {file.filename}")
    
    if not file.content_type == "application/pdf":
        raise HTTPException(400, "Only PDF files are supported")
        
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    try:
        # Process the PDF
        text = get_text(tmp_path)
        logger.debug(f"Extracted text length: {len(text)}")
        
        sent_tokens = process_text_with_spacy(text, nlp)
        logger.debug(f"Number of sentences: {len(sent_tokens)}")
        
        predictions = get_predictions(sent_tokens, model, tokenizer)
        logger.info(f"Extracted fields: {predictions}")
        
        return {"fields": predictions}
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 