"""
Field Extraction and Consistency Logic Module

This module handles the core logic for extracting fields from regulatory documents
and checking consistency between them. It provides functionality for processing
plankart, bestemmelser, and SOSI files.

Features:
- Field extraction from uploaded files
- Consistency checking between documents
- Metadata extraction (plan IDs, dates)
- Text section extraction and analysis
"""

from typing import Set, Tuple, List, Optional
import logging
from .models import ConsistencyResult
from .document.pdf_handler import PdfHandler, extract_text_from_pdf
from .llm.extractor import FieldExtractor
from .config import settings
from fastapi import FastAPI, File, UploadFile, HTTPException
import uuid
import os
import re

logger = logging.getLogger(__name__)

# Initialize FastAPI and field extractor
app = FastAPI()
field_extractor = FieldExtractor(settings.OPENAI_MODEL)

async def extract_fields_from_file(file) -> Set[str]:
    """
    Extract regulatory fields from an uploaded file.
    
    Args:
        file (UploadFile): The uploaded file to process
        
    Returns:
        Set[str]: Set of extracted field names
        
    Note:
        Returns empty set if file is None or extraction fails
    """
    if not file:
        return set()

    try:
        content = await file.read()
        fields = await field_extractor.extract_fields(content)
        logger.info(f"Extracted {len(fields)} fields from {file.filename}")
        return fields
        
    except Exception as e:
        logger.error(f"Extraction error for file {file.filename}: {str(e)}")
        logger.debug("Full error:", exc_info=True)
        return set()

@app.post("/api/check-field-consistency")
async def check_field_consistency(
    plankart: UploadFile = File(...), 
    bestemmelser: UploadFile = File(...),
    sosi: UploadFile = File(None)
):
    """Check consistency between files"""
    try:
        # Log incoming files
        logger.info(f"Received files: plankart={plankart.filename}, bestemmelser={bestemmelser.filename}")
        logger.info(f"File content types: plankart={plankart.content_type}, bestemmelser={bestemmelser.content_type}")
        
        # Validate files
        if not plankart or not bestemmelser:
            raise HTTPException(status_code=400, detail="Missing required files")
            
        if not plankart.content_type in ['application/pdf', 'text/xml']:
            raise HTTPException(status_code=400, detail=f"Invalid plankart file type: {plankart.content_type}")
            
        if not bestemmelser.content_type in ['application/pdf', 'text/xml']:
            raise HTTPException(status_code=400, detail=f"Invalid bestemmelser file type: {bestemmelser.content_type}")

        # Extract fields from each file
        plankart_content = await plankart.read()
        bestemmelser_content = await bestemmelser.read()
        
        plankart_fields = await field_extractor.extract_fields(plankart_content)
        bestemmelser_fields = await field_extractor.extract_fields(bestemmelser_content)
        sosi_fields = set() if not sosi else await field_extractor.extract_fields(await sosi.read())

        # Convert bytes to string for metadata extraction
        bestemmelser_text = bestemmelser_content.decode('utf-8', errors='ignore')

        # Extract metadata
        metadata = {
            "plan_id": extract_plan_id(bestemmelser_text),
            "plankart_dato": extract_date("plankart", bestemmelser_text),
            "bestemmelser_dato": extract_date("bestemmelser", bestemmelser_text),
            "vedtatt_dato": extract_date("vedtatt", bestemmelser_text)
        }

        # Calculate field differences
        matching = plankart_fields & bestemmelser_fields
        only_plankart = plankart_fields - bestemmelser_fields
        only_bestemmelser = bestemmelser_fields - plankart_fields
        only_sosi = sosi_fields - (plankart_fields | bestemmelser_fields) if sosi_fields else set()

        return {
            "result": {
                "matching_fields": sorted(list(matching)),
                "only_in_plankart": sorted(list(only_plankart)),
                "only_in_bestemmelser": sorted(list(only_bestemmelser)),
                "only_in_sosi": sorted(list(only_sosi)),
                "is_consistent": len(only_plankart) == 0 and len(only_bestemmelser) == 0 and len(only_sosi) == 0,
                "document_fields": {
                    "plankart": {
                        "raw_fields": sorted(list(plankart_fields)),
                        "normalized_fields": sorted(list(plankart_fields))
                    },
                    "bestemmelser": {
                        "raw_fields": sorted(list(bestemmelser_fields)),
                        "normalized_fields": sorted(list(bestemmelser_fields))
                    }
                },
                "metadata": metadata
            }
        }
    except Exception as e:
        logger.error(f"Consistency check error: {str(e)}")
        logger.debug("Full error:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def extract_fields_and_text(file: UploadFile) -> Tuple[Set[str], List[str]]:
    """
    Extract both fields and relevant text sections from a file.
    
    Args:
        file (UploadFile): The file to process
        
    Returns:
        Tuple[Set[str], List[str]]: Extracted fields and relevant text sections
        
    Note:
        Creates temporary file for processing and ensures cleanup
    """
    try:
        content = await file.read()
        temp_path = f"/tmp/{uuid.uuid4()}.pdf"
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        extracted_text = extract_text_from_pdf(temp_path)
        fields = field_extractor.extract_fields(extracted_text)
        relevant_text = extract_relevant_sections(extracted_text)
        
        return fields, relevant_text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def extract_relevant_sections(text: str) -> List[str]:
    """
    Extract relevant sections from document text.
    
    Args:
        text (str): Full document text
        
    Returns:
        List[str]: List of relevant text sections
        
    Note:
        Filters sections based on regulatory keywords
    """
    sections = text.split('\n\n')
    relevant_sections = []
    
    keywords = ['plan', 'formål', 'bestemmelse', 'regulering', 'vedtatt']
    for section in sections:
        if any(keyword in section.lower() for keyword in keywords):
            relevant_sections.append(section.strip())
    
    return relevant_sections

def extract_plan_id(text: str) -> Optional[str]:
    """
    Extract plan ID from document text.
    
    Args:
        text (str): Document text to search
        
    Returns:
        Optional[str]: Extracted plan ID or None if not found
    """
    if match := re.search(r'Plan\s*ID:?\s*(\d+)', text, re.IGNORECASE):
        return match.group(1)
    return None

def extract_date(type_: str, text: str) -> Optional[str]:
    """
    Extract date information from document text.
    
    Args:
        type_ (str): Type of date to extract (plankart/bestemmelser/vedtatt)
        text (str): Document text to search
        
    Returns:
        Optional[str]: Extracted date or None if not found
        
    Note:
        Dates should be in DD.MM.YYYY format
    """
    date_patterns = {
        "plankart": r'Plankart\s*dato:?\s*(\d{2}\.\d{2}\.\d{4})',
        "bestemmelser": r'[Bb]estemmelser?\s*dato:?\s*(\d{2}\.\d{2}\.\d{4})',
        "vedtatt": r'[Vv]edtatt:?\s*(\d{2}\.\d{2}\.\d{4})'
    }
    
    if pattern := date_patterns.get(type_):
        if match := re.search(pattern, text):
            return match.group(1)
    return None