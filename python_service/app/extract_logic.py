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

from typing import Set, Tuple, List, Optional, Dict # Dict is not accessed
import logging
from .models import ConsistencyResult # Not accessed
from .document.pdf_handler import PdfHandler, extract_text_from_pdf # PdfHandler is not accessed
from .config import settings, extractor # settings is not accessed
from fastapi import FastAPI, File, UploadFile, HTTPException # FastAPI & File is not accessed
import uuid
import os
import re
from pydantic import BaseModel # Not accessed
import asyncio
import httpx
from app.utils.logger import logger
import tempfile

logger = logging.getLogger(__name__)



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
        
        plankart_text = await extract_text_from_pdf(temp_path)
        fields = extractor.extract_fields(plankart_text)
        relevant_text = extract_relevant_sections(plankart_text)
        
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
    """Normalize field names for comparison"""
    field = field.strip()
    
    # Skip area measurements and coordinates
    if 'a=' in field.lower() or re.match(r'^[NS]\d{7}$', field):
        return None
    
    # Keep original case for special patterns
    if field.startswith('#') and 'SNØ' in field:
        return field
    if re.match(r'^H\d{3}$', field):
        return field
        
    # Handle number ranges in codes
    range_match = re.match(r'([a-zA-Z_]+)(\d+)-(\d+)$', field)
    if range_match:
        prefix, start, end = range_match.groups()
        return f"{prefix}{start}"  # Return just the first in range
        
    # Handle o_ and f_ prefixes
    if field.lower().startswith(('o_', 'f_')):
        prefix = field[:2].lower()
        rest = field[2:].upper()
        return f"{prefix}{rest}"
        
    return field.upper()

async def extract_text_from_pdf(file: UploadFile) -> bytes:
    """Extract raw bytes from PDF file in chunks"""
    try:
        # Read the file content
        content = await file.read()
        if not content:
            logger.error("Empty file received")
            raise HTTPException(status_code=400, detail="File is empty")
            
        # Reset file position for future reads
        await file.seek(0)
        return content
    except Exception as e:
        logger.error(f"Error reading PDF file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading PDF file: {str(e)}")

async def process_consistency_check(plankart: UploadFile, bestemmelser: UploadFile, sosi: UploadFile = None):
    """Process consistency check between files"""
    try:
        if not plankart.content_type == 'application/pdf':
            raise HTTPException(status_code=400, detail=f"Invalid plankart file type: {plankart.content_type}")
            
        # Extract fields from plankart using LLM
        plankart_content = await extract_text_from_pdf(plankart)
        plankart_fields = await extractor.extract_fields(plankart_content)
        logger.debug(f"Plankart fields: {plankart_fields}")
        
        # Clean and normalize fields
        plankart_fields = {
            normalize_field(field) 
            for field in plankart_fields 
            if normalize_field(field) is not None
        }
        
        # Get bestemmelser fields from NER service
        try:
            # Read the file content
            bestemmelser_content = await extract_text_from_pdf(bestemmelser)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(bestemmelser_content)
                temp_file_path = temp_file.name

            # Send to NER service
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(temp_file_path, 'rb') as f:
                    files = {'file': ('bestemmelser.pdf', f, 'application/pdf')}
                    response = await client.post('http://localhost:8001/api/extract-fields', files=files)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPError as e:
                        # Try to get the detailed error message from the NER service
                        error_detail = "Unknown error"
                        try:
                            error_json = response.json()
                            if 'detail' in error_json:
                                error_detail = error_json['detail']
                        except Exception:
                            pass
                        logger.error(f"NER service error: {error_detail}")
                        raise HTTPException(status_code=500, detail=error_detail)
                    
                    bestemmelser_fields = set(response.json()['fields'])

            # Clean up
            os.unlink(temp_file_path)
            
        except HTTPException:
            raise
        except httpx.HTTPError as e:
            logger.error(f"NER service communication error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"Failed to communicate with NER service: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing bestemmelser: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing bestemmelser: {str(e)}")
        
        # Clean and normalize bestemmelser fields
        bestemmelser_fields = {
            normalize_field(field)
            for field in bestemmelser_fields 
            if (
                len(field.strip()) > 1  # Remove single characters
                and not field.strip().startswith('på')  # Remove common noise
                and not field.strip() in ['er', 'langs', 'området', 'skal', 'midlertidig']
                and normalize_field(field) is not None
            )
        }
        logger.debug(f"Cleaned bestemmelser fields: {bestemmelser_fields}")

        # Handle SOSI if present
        sosi_fields = set()
        if sosi:
            sosi_content = await extract_text_from_pdf(sosi)
            sosi_fields = await extractor.extract_fields(sosi_content)
            logger.debug(f"SOSI fields: {sosi_fields}")

        # Calculate differences
        only_plankart = plankart_fields - (bestemmelser_fields | sosi_fields)
        only_bestemmelser = bestemmelser_fields - (plankart_fields | sosi_fields)
        only_sosi = sosi_fields - (plankart_fields | bestemmelser_fields) if sosi else set()
        
        return {
            "matching_fields": sorted(list(plankart_fields & bestemmelser_fields & (sosi_fields if sosi else plankart_fields))),
            "only_in_plankart": sorted(list(only_plankart)),
            "only_in_bestemmelser": sorted(list(only_bestemmelser)),
            "only_in_sosi": sorted(list(only_sosi)),
            "is_consistent": len(only_plankart) == 0 and len(only_bestemmelser) == 0 and len(only_sosi) == 0,
            "document_fields": {
                "plankart": {
                    "raw_fields": sorted(list(plankart_fields)),
                    "normalized_fields": sorted(list({normalize_field(field) for field in plankart_fields})),
                    "text_sections": []
                },
                "bestemmelser": {
                    "raw_fields": sorted(list(bestemmelser_fields)),
                    "normalized_fields": sorted(list({normalize_field(field) for field in bestemmelser_fields})),
                    "text_sections": []
                },
                "sosi": {
                    "raw_fields": sorted(list(sosi_fields)),
                    "normalized_fields": sorted(list({normalize_field(field) for field in sosi_fields})),
                    "text_sections": []
                } if sosi else None
            }
        }
    except Exception as e:
        logger.error(f"Consistency check error: {str(e)}")
        logger.debug("Full error:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def get_bestemmelser_fields(bestemmelser: UploadFile) -> Set[str]:
    """Get fields from bestemmelser using NER service with retry"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Read the file content
            content = await bestemmelser.read()
            if not content:
                logger.error("Empty file received")
                raise HTTPException(status_code=400, detail="File is empty")
                
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Send to NER service
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(temp_file_path, 'rb') as f:
                    files = {'file': ('bestemmelser.pdf', f, 'application/pdf')}
                    response = await client.post('http://localhost:8001/api/extract-fields', files=files)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPError as e:
                        # Try to get the detailed error message from the NER service
                        error_detail = "Unknown error"
                        try:
                            error_json = response.json()
                            if 'detail' in error_json:
                                error_detail = error_json['detail']
                        except Exception:
                            pass
                        logger.error(f"NER service error: {error_detail}")
                        raise HTTPException(status_code=500, detail=error_detail)
                    
                    fields = set(response.json()['fields'])

            # Clean up
            os.unlink(temp_file_path)
            
            # Reset file position
            await bestemmelser.seek(0)
            
            return fields
            
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                logger.error(f"NER service communication error after {max_retries} attempts: {str(e)}")
                raise HTTPException(status_code=502, detail=str(e))
            await asyncio.sleep(retry_delay * (attempt + 1))