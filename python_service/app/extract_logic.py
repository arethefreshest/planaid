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
from .config import settings, extractor
from fastapi import FastAPI, File, UploadFile, HTTPException
import uuid
import os
import re
from pydantic import BaseModel
import asyncio

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

async def extract_fields_from_file(file: UploadFile) -> Set[str]:
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
        # Process file in chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        content = bytearray()
        
        while chunk := await file.read(chunk_size):
            content.extend(chunk)
            
        return await extractor.extract_fields(bytes(content))
    except Exception as e:
        logger.error(f"Field extraction error: {str(e)}")
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

        # Read files concurrently
        content_tasks = [
            plankart.read(),
            bestemmelser.read(),
        ]
        if sosi:
            content_tasks.append(sosi.read())
            
        # Wait for all file reads to complete
        contents = await asyncio.gather(*content_tasks)
        plankart_content = contents[0]
        bestemmelser_content = contents[1]
        sosi_content = contents[2] if len(contents) > 2 else None

        # Convert bytes to string for metadata extraction first
        bestemmelser_text = bestemmelser_content.decode('utf-8', errors='ignore')

        # Extract fields concurrently
        fields_tasks = [
            extractor.extract_fields(plankart_content),
            extractor.extract_fields(bestemmelser_content)
        ]
        if sosi_content:
            fields_tasks.append(extractor.extract_fields(sosi_content))
            
        # Wait for all extractions to complete
        results = await asyncio.gather(*fields_tasks)
        
        plankart_fields = results[0]
        bestemmelser_fields = results[1]
        sosi_fields = results[2] if len(results) > 2 else set()

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
            "status": "success",
            "result": {
                "matching_fields": sorted(list(matching)),
                "only_in_plankart": sorted(list(only_plankart)),
                "only_in_bestemmelser": sorted(list(only_bestemmelser)),
                "only_in_sosi": sorted(list(only_sosi)),
                "is_consistent": len(only_plankart) == 0 and len(only_bestemmelser) == 0 and len(only_sosi) == 0,
                "document_fields": {
                    "plankart": {
                        "raw_fields": sorted(list(plankart_fields)),
                        "normalized_fields": sorted(list(normalize_field(f) for f in plankart_fields))
                    },
                    "bestemmelser": {
                        "raw_fields": sorted(list(bestemmelser_fields)),
                        "normalized_fields": sorted(list(normalize_field(f) for f in bestemmelser_fields))
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
        fields = extractor.extract_fields(extracted_text)
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

def normalize_field(field: str) -> str:
    """Normalize field names by removing prefixes and standardizing format"""
    field = field.upper().strip()
    # Remove common prefixes
    prefixes = ['O_', 'F_', 'H_']
    for prefix in prefixes:
        if field.startswith(prefix):
            field = field[len(prefix):]
    return field

async def process_consistency_check(plankart: UploadFile, bestemmelser: UploadFile, sosi: UploadFile = None):
    """Process consistency check between files"""
    try:
        # Extract fields from each file
        plankart_content = await plankart.read()
        bestemmelser_content = await bestemmelser.read()
        
        plankart_fields = await extractor.extract_fields(plankart_content)
        bestemmelser_fields = await extractor.extract_fields(bestemmelser_content)
        sosi_fields = set() if not sosi else await extractor.extract_fields(await sosi.read())

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
            "status": "success",
            "result": {
                "matching_fields": sorted(list(matching)),
                "only_in_plankart": sorted(list(only_plankart)),
                "only_in_bestemmelser": sorted(list(only_bestemmelser)),
                "only_in_sosi": sorted(list(only_sosi)),
                "is_consistent": len(only_plankart) == 0 and len(only_bestemmelser) == 0 and len(only_sosi) == 0,
                "document_fields": {
                    "plankart": {
                        "raw_fields": sorted(list(plankart_fields)),
                        "normalized_fields": sorted(list(normalize_field(f) for f in plankart_fields))
                    },
                    "bestemmelser": {
                        "raw_fields": sorted(list(bestemmelser_fields)),
                        "normalized_fields": sorted(list(normalize_field(f) for f in bestemmelser_fields))
                    }
                },
                "metadata": metadata
            }
        }
    except Exception as e:
        logger.error(f"Consistency check error: {str(e)}")
        logger.debug("Full error:", exc_info=True)
        raise