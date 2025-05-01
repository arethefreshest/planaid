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
from .document.sosi_handler import extract_fields_from_sosi
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
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Get NER service URL from environment
NER_SERVICE_URL = os.getenv('NER_SERVICE_URL', 'http://157.230.21.199:8001')

# Setup metrics directory
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'metrics')
CONSISTENCY_METRICS_DIR = os.path.join(METRICS_DIR, 'consistency')
os.makedirs(CONSISTENCY_METRICS_DIR, exist_ok=True)

def log_consistency_metrics(result: Dict, plankart_fields: Set[str], bestemmelser_fields: Set[str], sosi_data: Optional[Dict] = None) -> None:
    """Log detailed consistency check metrics"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "document_fields": {
                "plankart": sorted(list(plankart_fields)),
                "bestemmelser": sorted(list(bestemmelser_fields)),
                "sosi": sosi_data["fields"] if sosi_data else None
            },
            "comparison_results": {
                "matching_fields": result["matching_fields"],
                "only_in_plankart": result["only_in_plankart"],
                "only_in_bestemmelser": result["only_in_bestemmelser"],
                "only_in_sosi": result["only_in_sosi"]
            },
            "statistics": {
                "total_fields": {
                    "plankart": len(plankart_fields),
                    "bestemmelser": len(bestemmelser_fields),
                    "sosi": sosi_data["metadata"]["total_fields"] if sosi_data else 0
                },
                "match_percentages": {
                    "plankart_match": round(len(result["matching_fields"]) / len(plankart_fields) * 100, 2) if plankart_fields else 0,
                    "bestemmelser_match": round(len(result["matching_fields"]) / len(bestemmelser_fields) * 100, 2) if bestemmelser_fields else 0,
                    "sosi_match": round(len(result["matching_fields"]) / sosi_data["metadata"]["total_fields"] * 100, 2) if sosi_data else 0
                }
            }
        }
        
        # Log to file
        log_file = os.path.join(CONSISTENCY_METRICS_DIR, f'consistency_check_{timestamp}.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Log summary to console
        logger.info("Consistency Check Results:")
        logger.info(f"Total fields found - Plankart: {len(plankart_fields)}, Bestemmelser: {len(bestemmelser_fields)}, "
                    f"SOSI: {sosi_data['metadata']['total_fields'] if sosi_data else 0}")
        logger.info(f"Matching fields: {len(result['matching_fields'])}")
        logger.info(f"Fields only in Plankart: {len(result['only_in_plankart'])}")
        logger.info(f"Fields only in Bestemmelser: {len(result['only_in_bestemmelser'])}")
        logger.info(f"Fields only in SOSI: {len(result['only_in_sosi'])}")
        
        # Log specific fields
        logger.info("\nDetailed field breakdown:")
        if result["matching_fields"]:
            logger.info("Matching fields: " + ", ".join(result["matching_fields"]))
        if result["only_in_plankart"]:
            logger.info("Only in Plankart: " + ", ".join(result["only_in_plankart"]))
        if result["only_in_bestemmelser"]:
            logger.info("Only in Bestemmelser: " + ", ".join(result["only_in_bestemmelser"]))
        if result["only_in_sosi"]:
            logger.info("Only in SOSI: " + ", ".join(result["only_in_sosi"]))
    except Exception as e:
        logger.error(f"Error logging consistency metrics: {str(e)}")

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

def normalize_field(field: str) -> Optional[str]:
    """Normalize field names for comparison"""
    field = field.strip()
    
    # Skip non-field content
    if any(noise in field for noise in [
        'SAKSBEHANDLING',
        'PLANKONSULENT',
        'PLANEN ER',
        'DATOSIGN',
        'NN2000'
    ]):
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
        
    # Only keep valid field patterns
    if not re.match(r'^([fo]_)?[A-Z]+\d*$', field) and not re.match(r'^#\d+$', field):
        return None
        
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
        bestemmelser_fields = await get_bestemmelser_fields(bestemmelser)
        
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
        sosi_data = None
        if sosi:
            # Read SOSI content
            sosi_content = await sosi.read()
            sosi_data = await extract_fields_from_sosi(sosi_content)
            
            # Get all relevant fields from SOSI
            sosi_fields = set(sosi_data["fields"]["zone_identifiers"])
            sosi_hensynssoner = set(sosi_data["fields"]["hensynssoner"])
            
            # Normalize SOSI fields
            sosi_fields = {
                normalize_field(field)
                for field in sosi_fields
                if normalize_field(field) is not None
            }
            
            # Keep hensynssoner as is since they have special format
            sosi_all_fields = sosi_fields | sosi_hensynssoner

        # Calculate differences
        if sosi_data:
            matching_fields = sorted(list(sosi_all_fields & (plankart_fields | bestemmelser_fields)))
            only_in_plankart = sorted(list(plankart_fields - sosi_all_fields))
            only_in_bestemmelser = sorted(list(bestemmelser_fields - sosi_all_fields))
            only_in_sosi = sorted(list(sosi_all_fields - (plankart_fields | bestemmelser_fields)))
            
            result = {
                "matching_fields": matching_fields,
                "only_in_plankart": only_in_plankart,
                "only_in_bestemmelser": only_in_bestemmelser,
                "only_in_sosi": only_in_sosi,
                "sosi_structure": sosi_data["structure"],
                "sosi_metadata": sosi_data["metadata"],
                "field_types": {
                    "zone_identifiers": sorted(list(sosi_fields)),
                    "hensynssoner": sorted(list(sosi_hensynssoner)),
                    "purposes": sorted(list(sosi_data["fields"]["purposes"]))
                }
            }
        else:
            matching_fields = sorted(list(plankart_fields & bestemmelser_fields))
            only_in_plankart = sorted(list(plankart_fields - bestemmelser_fields))
            only_in_bestemmelser = sorted(list(bestemmelser_fields - plankart_fields))
            
            result = {
                "matching_fields": matching_fields,
                "only_in_plankart": only_in_plankart,
                "only_in_bestemmelser": only_in_bestemmelser,
                "only_in_sosi": [],
                "sosi_structure": None,
                "sosi_metadata": None,
                "field_types": None
            }
        
        # Log detailed metrics
        log_consistency_metrics(result, plankart_fields, bestemmelser_fields, sosi_data)
        
        return result
            
    except Exception as e:
        logger.error(f"Error in consistency check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_bestemmelser_fields(bestemmelser: UploadFile) -> Set[str]:
    """Get fields from bestemmelser using NER service with retry"""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # Read content first
            content = await bestemmelser.read()
            if not content:
                logger.error("Empty file received")
                raise HTTPException(status_code=400, detail="File is empty")

            # ⚠️ Reset filposisjon etter read()
            await bestemmelser.seek(0)

            # Lag midlertidig fil
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Send til NER-service
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(temp_file_path, 'rb') as f:
                    files = {'file': ('bestemmelser.pdf', f, 'application/pdf')}
                    response = await client.post(f'{NER_SERVICE_URL}/api/extract-fields', files=files)
                    response.raise_for_status()
                    fields = set(response.json()['fields'])

            os.unlink(temp_file_path)
            await bestemmelser.seek(0)  # Viktig for videre bruk
            return fields

        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                logger.error(f"NER service communication error after {max_retries} attempts: {str(e)}")
                raise HTTPException(status_code=502, detail=str(e))
            await asyncio.sleep(retry_delay * (attempt + 1))
