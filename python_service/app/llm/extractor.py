"""
Field Extraction Module for Regulation Documents

This module handles the extraction of regulatory codes and fields from PDF documents
using LLM-based extraction techniques. It supports processing of plankart, 
bestemmelser, and SOSI files.

Key Components:
- FieldExtractContract: Defines the extraction schema and prompt
- FieldExtractor: Main class for handling field extraction logic
"""

from extract_thinker import Extractor, DocumentLoaderPyPdf, Contract, LLM
from typing import Set, Tuple
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
        Les gjennom teksten og ekstraher KUN reguleringskoder som brukes i arealformål. 
        Ignorer referanser til PBL (Plan- og bygningsloven).

        Se spesielt etter:
        1. Arealformål som:
           - BKS (Konsentrert småhusbebyggelse)
           - BFS (Frittliggende småhusbebyggelse)
           - o_BOP (Offentlig tjenesteyting)
           - o_GF (Friområde)
           - o_SGS (Gang-/sykkelveg)
           - o_SKV (Kjøreveg)

        2. Koder med prefiks:
           - o_ (offentlig)
           - f_ (felles)
           - b_ (bebyggelse)

        Viktig:
        - Returner KUN reguleringskoder, ikke PBL-referanser eller beskrivelser
        - Ta med prefiks og nummer hvis det finnes
        - Ignorer alle referanser til paragrafer (§)
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
        
        # Create and load LLM with Azure configuration
        llm = LLM(model=model_name, token_limit=4000)
        self.extractor.load_llm(llm)
        
        # Load reference codes
        self.sosi_codes, self.benevnelseskoder = self._load_codes()
    
    def _load_codes(self) -> Tuple[Set[str], Set[str]]:
        csv_path = '/app/app/data/Reguleringsplan.csv'
        fallback_path = os.path.join(os.path.dirname(__file__), 'data/Reguleringsplan.csv')
        
        try:
            path_to_use = csv_path if os.path.exists(csv_path) else fallback_path
            logger.debug(f"Attempting to load codes from: {path_to_use}")
            
            if not os.path.exists(path_to_use):
                logger.error(f"Neither {csv_path} nor {fallback_path} exist")
                raise FileNotFoundError(f"Neither {csv_path} nor {fallback_path} exist")
            
            with open(path_to_use, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                sosi_codes = {row['SOSI-kode'].strip() for row in reader if row.get('SOSI-kode')}
                logger.debug(f"Loaded SOSI codes: {sosi_codes}")
            
            with open(path_to_use, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=';')
                benevnelseskoder = {row['Benevnelse'].strip() for row in reader if row.get('Benevnelse')}
                logger.debug(f"Loaded Benevnelseskoder: {benevnelseskoder}")
            
            return sosi_codes, benevnelseskoder
            
        except Exception as e:
            logger.error(f"Failed to load regulation codes: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load required regulation codes: {str(e)}")
    
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
        """Validate a field against known regulation codes."""
        field = field.strip().upper()
        
        # If we have no base codes, accept all fields that look like regulation codes
        if not base_codes:
            logger.warning("No base codes available, using pattern matching")
            # Check if it matches common regulation code patterns
            common_prefixes = ['O_', 'F_', 'B_', 'BFS', 'BKS', 'SKV', 'SGS', 'GF']
            return any(field.startswith(prefix) for prefix in common_prefixes)
        
        # Normal validation against base codes
        if field in base_codes:
            return True
        
        # Remove prefixes for comparison
        stripped_field = field
        for prefix in ['O_', 'F_', 'B_']:
            if field.startswith(prefix):
                stripped_field = field[len(prefix):]
                break
            
        # Check if the stripped field exists in base codes
        if stripped_field in base_codes:
            return True
        
        # Check for partial matches (e.g., GTD matches GTD1)
        return any(code.startswith(stripped_field) or stripped_field.startswith(code) 
                  for code in base_codes)

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