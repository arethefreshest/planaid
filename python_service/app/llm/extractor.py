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
import io # Not accessed
import csv
import os
import re
import tempfile
from app.utils.logger import logger

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
        Les gjennom teksten og ekstraher KUN reguleringskoder. 
        Ignorer referanser til PBL (Plan- og bygningsloven) og lange beskrivelser.

        Se spesielt etter disse kodene (må finnes i teksten):
        1. Offentlige koder:
           - o_SV1
           - o_SV3
           - o_SPA
           - o_GF1, o_GF2, o_GF3
           - o_GTD1, o_GTD2
           - o_SF1, o_SF2
        
        2. Felles koder:
           - f_BE
           - f_VEG
        
        3. Basisformer:
           - BKS
           - BIA
           - BR

        4. Hensynssoner:
           - H210, H220, H320

        5. Spesielle koder:
           - #01 SNØ
           - #02 SNØ

        6. Tallserier skal ekspanderes:
           - o_GF1-3 blir o_GF1, o_GF2, o_GF3
           - o_GTD1-2 blir o_GTD1, o_GTD2

        VIKTIG:
        - Se etter koder med mellomrom eller newline: 
          "o_SV1\\no_SV3" skal bli ["o_SV1", "o_SV3"]
          "f_BE " skal bli "f_BE"
        - Returner KUN korte koder
        - Ta med prefiks og nummer
        - Behold original bokstavstørrelse (o_GF1, ikke O_GF1)
        - Ignorer paragrafer (§)
        - Ekspander tallserier til individuelle koder
        - Godta koder med whitespace rundt (f.eks " o_SV1 ", " f_BE ")
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
        skip_bases (Set[str]): Set of base codes to skip
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
        
        # Initialize skip_bases set
        self.skip_bases = set()
        self.fields = set()
    
    def _load_codes(self) -> Tuple[Set[str], Set[str]]:
        # Try multiple possible paths for the CSV file
        possible_paths = [
            '/app/app/data/Reguleringsplan.csv',  # Docker path
            os.path.join(os.path.dirname(__file__), 'data/Reguleringsplan.csv'),  # Relative to extractor.py
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'app/data/Reguleringsplan.csv')  # Relative to project root
        ]
        
        path_to_use = None
        for path in possible_paths:
            if os.path.exists(path):
                path_to_use = path
                break
                
        if not path_to_use:
            logger.error(f"Could not find Reguleringsplan.csv in any of: {possible_paths}")
            raise FileNotFoundError(f"Could not find Reguleringsplan.csv in any of: {possible_paths}")
            
        logger.debug(f"Loading codes from: {path_to_use}")
        
        try:
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
        field = field.strip()
        
        # Skip property numbers and coordinates
        if re.match(r'^\d+/\d+$', field) or re.match(r'^[NS]\d+$', field):
            logger.debug(f"Skipping number/coordinate: {field}")
            return False
        
        # Skip PBL references
        if 'PBL' in field.upper() or '§' in field:
            logger.debug(f"Skipping PBL reference: {field}")
            return False

        # Clean up field - remove any trailing text after space, but keep SNØ
        if not re.match(r'^#\d+\s+SNØ$', field):
            field = field.split()[0]

        # Special cases that are always valid
        if re.match(r'^#\d+\s+SNØ$', field):
            logger.debug(f"Accepting special code: {field}")
            return True
        if re.match(r'^H\d{3}$', field):
            logger.debug(f"Accepting special code: {field}")
            return True
        
        # Skip standalone SNØ
        if field.upper() == 'SNØ':
            logger.debug(f"Skipping standalone SNØ")
            return False
        
        # Handle number ranges in codes (e.g., o_GTD1-3)
        range_match = re.match(r'([a-zA-Z_]+)(\d+)-(\d+)$', field)
        if range_match:
            prefix, start, end = range_match.groups()
            try:
                # Generate all numbers in the range and add them individually
                expanded_fields = [f"{prefix}{i}" for i in range(int(start), int(end) + 1)]
                logger.debug(f"Expanded range {field} to: {expanded_fields}")
                # Add each expanded field to the set
                for expanded_field in expanded_fields:
                    self.fields.add(expanded_field)
                return False  # Return False to skip adding the range format
            except ValueError:
                return False
            
        # If we have no base codes, use pattern matching
        if not base_codes:
            logger.warning("No base codes available, using pattern matching")
            
            # First check for prefixed codes with numbers (o_SV1, f_BE1)
            prefixed_with_number = re.match(r'^[ofb]_[A-ZÆØÅ]{2,}\d*\b', field, re.IGNORECASE)
            if prefixed_with_number:
                base = re.sub(r'\d+$', '', field).strip()  # Remove numbers
                self.skip_bases.add(base)
                self.skip_bases.add(re.sub(r'^[ofb]_', '', base))
                field = field.split()[0].strip()  # Clean after matching
                logger.debug(f"Accepting prefixed code with number: {field}")
                return True
            
            # Then check for prefixed codes without numbers (f_BE)
            prefixed_no_number = re.match(r'^[ofb]_[A-ZÆØÅ]{2,}$', field, re.IGNORECASE)
            if prefixed_no_number:
                base = re.sub(r'^[ofb]_', '', field)  # Get base code
                self.skip_bases.add(base)  # Skip unprefixed version
                if field not in self.skip_bases:
                    logger.debug(f"Accepting prefixed code without number: {field}")
                    return True
                return False
            
            # Check for base codes with numbers (BE1)
            base_with_number = re.match(r'^[A-ZÆØÅ]{2,}\d+$', field)
            if base_with_number:
                base = re.sub(r'\d+$', '', field)
                if not any(f.endswith(field) for f in self.fields if f.startswith(('o_', 'f_', 'b_'))):
                    self.skip_bases.add(base)
                    logger.debug(f"Accepting base code with number: {field}")
                    return True
                return False
            
            # Finally check for simple base codes (BE)
            base_code = re.match(r'^[A-ZÆØÅ]{2,}$', field)
            if base_code and field not in self.skip_bases:
                # Don't accept if we have a prefixed or numbered version
                if not any(f.endswith(field) or f.endswith(field + '1') 
                          for f in self.fields if f.startswith(('o_', 'f_', 'b_'))):
                    logger.debug(f"Accepting base code: {field}")
                    return True
                return False
            
            return False
        
        return False

    async def extract_fields(self, content: bytes) -> Set[str]:
        """Extract field identifiers from PDF content."""
        try:
            # Create a temporary file for the PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Use the document loader to load the content from the file path
            raw_text = self.document_loader.load(temp_file_path)
            
            # Handle list of dicts, list of strings, or single string
            if isinstance(raw_text, list):
                if raw_text and isinstance(raw_text[0], dict):
                    # Extract text from dict if available
                    cleaned_text = ' '.join(page.get('text', '') for page in raw_text)
                else:
                    # Join list of strings
                    cleaned_text = ' '.join(raw_text)
            else:
                cleaned_text = raw_text
            
            # Clean the text
            cleaned_text = cleaned_text.replace('\\\n', ' ').replace('\\n', '\n')
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)  # Remove non-ASCII characters
            
            logger.debug(f"Cleaned text for LLM: {cleaned_text[:500]}...")  # First 500 chars

            # Pass the file path to the extractor
            result = self.extractor.extract(
                source=temp_file_path,  # Pass the file path
                response_model=FieldExtractContract
            )
            
            if result and hasattr(result, 'fields'):
                base_codes = {code.strip() for code in self.benevnelseskoder}
                logger.debug(f"Available base codes: {base_codes}")
                
                fields = set()
                for field in result.fields:
                    logger.debug(f"Checking field: {field}")
                    if self._is_valid_code(field, base_codes):
                        # Keep original case
                        fields.add(field.strip())
                        logger.debug(f"Added valid code: {field}")
                
                logger.info(f"Extracted fields: {fields}")
                return fields
            
            return set()
        except Exception as e:
            logger.error(f"Field extraction error: {str(e)}", exc_info=True)
            return set()
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path) 