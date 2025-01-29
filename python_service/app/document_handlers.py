"""
Document Handler Module

This module provides handlers for different document types (PDF, XML, SOSI) used in
the regulatory document processing system. Each handler implements a common interface
for text extraction while handling format-specific requirements.

Key Components:
- DocumentHandler: Abstract base class defining the interface
- PdfHandler: Handles PDF document extraction
- XmlHandler: Handles XML document extraction
- SosiHandler: Handles SOSI file extraction

The handlers are used to extract text content from different file formats while
maintaining a consistent interface for the rest of the application.
"""

from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from pypdf import PdfReader
import io
import logging

# Set up logging
logger = logging.getLogger(__name__)

class DocumentHandler(ABC):
    """
    Abstract base class for document handlers.
    
    Defines the interface that all document handlers must implement for
    consistent text extraction across different file formats.
    """
    
    @abstractmethod
    async def extract_text(self, file_content: bytes) -> str:
        """
        Extract text content from a document.
        
        Args:
            file_content (bytes): Raw binary content of the document
            
        Returns:
            str: Extracted text content
            
        Raises:
            NotImplementedError: Must be implemented by concrete handlers
        """
        pass

class PdfHandler(DocumentHandler):
    """
    Handler for PDF documents.
    
    Extracts text content from PDF files using PyPDF library.
    Handles multi-page documents and concatenates content.
    """
    
    async def extract_text(self, file_content: bytes) -> str:
        """
        Extract text from PDF document.
        
        Args:
            file_content (bytes): Raw PDF file content
            
        Returns:
            str: Concatenated text from all PDF pages
            
        Note:
            Pages are separated by newlines in the output
        """
        try:
            pdf = PdfReader(io.BytesIO(file_content))
            text = "\n".join(page.extract_text() for page in pdf.pages)
            logger.debug(f"Extracted {len(pdf.pages)} pages from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise

class SosiHandler(DocumentHandler):
    """
    Handler for SOSI files.
    
    Processes SOSI (Samordnet Opplegg for Stedfestet Informasjon) files,
    which are commonly used in Norwegian mapping and planning.
    """
    
    async def extract_text(self, file_content: bytes) -> str:
        """
        Extract text from SOSI file.
        
        Args:
            file_content (bytes): Raw SOSI file content
            
        Returns:
            str: Decoded SOSI file content
            
        Note:
            Currently performs basic UTF-8 decoding. May need enhancement
            for specific SOSI parsing requirements.
        """
        try:
            text = file_content.decode('utf-8')
            logger.debug("Successfully decoded SOSI file")
            return text
        except UnicodeDecodeError as e:
            logger.error(f"Error decoding SOSI file: {str(e)}")
            raise

class XmlHandler(DocumentHandler):
    """
    Handler for XML documents.
    
    Processes XML files and converts them to a string representation
    while preserving the structure.
    """
    
    async def extract_text(self, file_content: bytes) -> str:
        """
        Extract text from XML document.
        
        Args:
            file_content (bytes): Raw XML file content
            
        Returns:
            str: String representation of XML content
            
        Note:
            Preserves XML structure in the output string
        """
        try:
            root = ET.fromstring(file_content)
            text = ET.tostring(root, encoding='unicode')
            logger.debug("Successfully parsed XML document")
            return text
        except ET.ParseError as e:
            logger.error(f"Error parsing XML: {str(e)}")
            raise

# Registry of file type handlers
HANDLERS = {
    'application/pdf': PdfHandler(),
    'text/xml': XmlHandler(),
    'application/x-sosi': SosiHandler(),
} 