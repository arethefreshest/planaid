"""
PDF Document Handler Module

This module provides functionality for extracting text content from PDF documents
using multiple PDF processing libraries for robust extraction.

Features:
- Asynchronous PDF text extraction
- Multiple PDF processing engines (PyPDF, pdfplumber)
- Error handling and logging
"""

from pypdf import PdfReader
import io
from typing import List # List is not accessed
import pdfplumber
from app.utils.logger import logger

class PdfHandler:
    """
    Handler for PDF document processing.
    
    Provides asynchronous text extraction from PDF documents
    using PyPDF library.
    """
    
    async def extract_text(self, content: bytes) -> str:
        """
        Extract text content from PDF bytes.
        
        Args:
            content (bytes): Raw PDF content
            
        Returns:
            str: Extracted text content
            
        Note:
            Returns empty string on extraction failure
        """
        try:
            pdf = PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() for page in pdf.pages)
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}")
            return ""

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using pdfplumber.
    
    Args:
        file_path (str): Path to PDF file
        
    Returns:
        str: Extracted text content
        
    Note:
        Uses pdfplumber for potentially better extraction results
        than PyPDF in some cases
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
    return text 