"""
Field Extraction Module for Regulation Documents

This module handles the extraction of regulatory codes and fields from PDF documents
using LLM-based extraction techniques. It supports processing of plankart, 
bestemmelser, and SOSI files.

Key Components:
- FieldExtractContract: Defines the extraction schema and prompt
- FieldExtractor: Main class for handling field extraction logic
"""

from extract_thinker import Extractor, DocumentLoaderPyPdf, Contract
from typing import Set
import logging
import io
import csv
import os

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FieldExtractContract(Contract):
    """
    Contract for field extraction from regulatory documents.
    
    Defines the structure and prompt for extracting regulation codes from
    Norwegian planning documents (plankart and bestemmelser).
    
    Attributes:
        fields (Set[str]): Set of extracted regulation codes
    """
    fields: Set[str]
    
    class Config:
        prompt = """
        Les gjennom teksten og ekstraher KUN reguleringskoder. Se spesielt etter:

        1. Koder i bestemmelser som:
           - "...på BKS skal..."
           - "...for o_GTD1-2 være..."
           - "...på BR, skal o_SF..."
           - "...og o_SPA være..."

        2. Koder i plankart som:
           - o_GTD1, o_GTD2 (Turdrag)
           - o_SF (Fortau)
           - o_SPA (Parkering)
           - BKS (Konsentrert småhusbebyggelse)
           - BR (Religionsbygg)
           - o_SKV (Kjøreveg)

        Viktig: 
        - Se etter koder i kontekst, spesielt rundt §-tegn og i bestemmelser
        - Inkluder både prefiks (o_, f_) og nummer hvis de finnes
        - Returner BARE selve kodene, ikke beskrivelsene
        - Koder kan være med bindestrek (f.eks. o_GTD1-2)
        
        Eksempel på tekst:
        "§2.3 Før midlertidig brukstillatelse på BKS skal o_GTD1-2 være ferdig opparbeidet."
        
        Skal gi:
        ["BKS", "o_GTD1-2"]
        """

class FieldExtractor:
    """
    Main class for extracting fields from regulatory documents.
    
    Handles the loading of documents, extraction of fields, and validation
    against known SOSI codes and benevnelseskoder.
    
    Attributes:
        extractor (Extractor): LLM-based extraction engine
        document_loader (DocumentLoaderPyPdf): PDF document loader
        sosi_codes (Set[str]): Set of valid SOSI codes
        benevnelseskoder (Set[str]): Set of valid regulation codes
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the field extractor with specified LLM model.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        # Initialize extractor and document loader
        self.extractor = Extractor()
        self.document_loader = DocumentLoaderPyPdf()
        self.extractor.load_document_loader(self.document_loader)
        self.extractor.load_llm(model_name)
        
        # Load reference codes
        self.sosi_codes, self.benevnelseskoder = self._load_codes()
    
    def _load_codes(self) -> tuple[Set[str], Set[str]]:
        """
        Load SOSI codes and benevnelseskoder from CSV file.
        
        Returns:
            tuple[Set[str], Set[str]]: Sets of SOSI codes and benevnelseskoder
        
        Note:
            CSV file should be located at '/app/app/data/Reguleringsplan.csv'
            with columns for 'SOSI-kode' and 'Benevnelseskode'
        """
        sosi_codes = set()
        benevnelseskoder = set()
        csv_path = '/app/app/data/Reguleringsplan.csv'
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    if row.get('SOSI-kode'):
                        # Add both with and without parentheses
                        sosi_codes.add(row['SOSI-kode'].strip('()'))
                        sosi_codes.add(row['SOSI-kode'])
                    if row.get('Benevnelseskode'):
                        benevnelseskoder.add(row['Benevnelseskode'])
                        
            logger.info(f"Loaded {len(benevnelseskoder)} benevnelseskoder and {len(sosi_codes)} SOSI codes")
        except Exception as e:
            logger.warning(f"Could not load codes from {csv_path}: {e}")
        
        return sosi_codes, benevnelseskoder
    
    async def compare_documents(self, plankart_content: bytes, bestemmelser_content: bytes) -> dict:
        """
        Compare fields between plankart and bestemmelser documents.
        
        Args:
            plankart_content (bytes): Content of plankart PDF
            bestemmelser_content (bytes): Content of bestemmelser PDF
            
        Returns:
            dict: Comparison results including matching fields and inconsistencies
        """
        # Extract fields from both documents
        plankart_fields = await self.extract_fields(plankart_content)
        bestemmelser_fields = await self.extract_fields(bestemmelser_content)
        
        # Find inconsistencies
        missing_in_bestemmelser = plankart_fields - bestemmelser_fields
        missing_in_plankart = bestemmelser_fields - plankart_fields
        
        return {
            "status": "success",
            "inconsistencies": {
                "missing_in_bestemmelser": list(missing_in_bestemmelser),
                "missing_in_plankart": list(missing_in_plankart)
            },
            "plankart_fields": list(plankart_fields),
            "bestemmelser_fields": list(bestemmelser_fields),
            "matching_fields": list(plankart_fields & bestemmelser_fields)
        }

    def _is_valid_code(self, field: str, base_codes: Set[str]) -> bool:
        """
        Validate if a field matches known regulation codes.
        
        Args:
            field (str): Field to validate
            base_codes (Set[str]): Set of valid base codes
            
        Returns:
            bool: True if field is valid, False otherwise
        """
        field = field.strip().upper()
        
        # Handle hyphenated codes (e.g., o_GTD1-2)
        base_field = field.split('-')[0]
        
        # Remove prefix if exists
        if base_field.startswith(('O_', 'F_')):
            base = base_field.split('_')[1]
        else:
            base = base_field
        
        # Remove numbers to get base code
        base = ''.join(c for c in base if not c.isdigit())
        
        # Check exact match first
        if base in base_codes:
            return True
        
        # Then check for partial matches (e.g., GTD matches GTD1)
        return any(code.startswith(base) or base.startswith(code) for code in base_codes)

    async def extract_fields(self, content: bytes) -> Set[str]:
        """
        Extract and validate fields from a document.
        
        Args:
            content (bytes): PDF document content
            
        Returns:
            Set[str]: Set of validated regulation codes
            
        Note:
            Uses LLM to extract potential fields and validates them against
            known regulation codes.
        """
        try:
            pdf_file = io.BytesIO(content)
            pdf_file.name = 'document.pdf'
            
            # Add debug logging for the raw text
            raw_text = self.document_loader.load(pdf_file)
            logger.debug(f"Raw text from PDF: {raw_text[:500]}...")  # First 500 chars
            
            result = self.extractor.extract(
                source=pdf_file,
                response_model=FieldExtractContract
            )
            
            if result and hasattr(result, 'fields'):
                base_codes = {code.strip() for code in self.benevnelseskoder}
                logger.debug(f"Available base codes: {base_codes}")
                
                fields = set()
                for field in result.fields:
                    logger.debug(f"Checking field: {field}")
                    if self._is_valid_code(field, base_codes):
                        fields.add(field.strip().upper())
                        logger.debug(f"Added valid code: {field}")
                
                logger.info(f"Extracted fields: {fields}")
                return fields
            
            return set()
            
        except Exception as e:
            logger.error(f"Field extraction error: {str(e)}", exc_info=True)
            return set() 