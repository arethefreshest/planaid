"""
Field Reference Module

This module handles the validation and comparison of regulatory field codes
against a reference dataset. It loads and processes field codes from a CSV
file containing official Norwegian regulatory codes.

The module supports:
- Loading and parsing of regulatory codes
- Validation of field names against reference data
- Comparison of fields between different document types
"""

from typing import Dict, List, Set # Set is not accessed
import pandas as pd
from app.utils.logger import logger
import os

class FieldReference:
    """
    Manages reference data for regulatory field codes.
    
    Loads and processes regulatory codes from a CSV file, providing
    validation and comparison functionality for field names.
    
    Attributes:
        df (pd.DataFrame): Raw reference data from CSV
        field_variations (Dict[str, Dict]): Processed reference data with variations
    """
    
    def __init__(self):
        """
        Initialize the field reference system.
        
        Loads reference data from CSV and builds field variations dictionary.
        """
        try:
            # Try multiple possible locations for the CSV file
            possible_paths = [
                'app/data/Reguleringsplan.csv',  # Local development path
                'data/Reguleringsplan.csv',      # Alternative local path
                '../../Reguleringsplan.csv',     # Docker path
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            if not csv_path:
                logger.debug("Could not find Reguleringsplan.csv, using pattern matching")
                self.df = pd.DataFrame()
                self.field_variations = {}
                return
                
            logger.info(f"Loading Reguleringsplan.csv from: {csv_path}")
            self.df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
            self.field_variations = self._build_variations()
            logger.info(f"Loaded {len(self.field_variations)} reference codes")
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            # Don't raise the error, just log it and continue with empty variations
            self.df = pd.DataFrame()
            self.field_variations = {}
    
    def _build_variations(self) -> Dict[str, Dict]:
        """
        Build dictionary of field code variations from reference data.
        
        Returns:
            Dict[str, Dict]: Mapping of base codes to their metadata
        """
        variations = {}
        
        for _, row in self.df.iterrows():
            if pd.isna(row['Formål']) or row['Formål'].startswith('Formål'):
                continue
            
            code = row['Benevnelseskode']
            if pd.isna(code):
                continue
            
            base_code = code.strip()
            variations[base_code] = {
                'purpose': row['Formål'],
                'sosi-code': row['SOSI-kode']
            }
            
        return variations
    
    def is_valid_base_code(self, field_name: str) -> bool:
        """
        Validate if a field name uses a valid base code.
        
        Args:
            field_name (str): Field name to validate (e.g., 'o_KV2')
            
        Returns:
            bool: True if the base code is valid
        """
        base_name = field_name.replace('o_', '').replace('f_', '')
        base_name = ''.join(c for c in base_name if not c.isdigit())
        
        return base_name in self.field_variations
    
    def compare_documents(self, plankart_fields: List[str], 
                         bestemmelser_fields: List[str]) -> Dict:
        """
        Compare field names between plankart and bestemmelser documents.
        
        Args:
            plankart_fields (List[str]): Fields from plankart
            bestemmelser_fields (List[str]): Fields from bestemmelser
            
        Returns:
            Dict: Comparison results including matches and differences
        """
        plankart_set = set(plankart_fields)
        bestemmelser_set = set(bestemmelser_fields)
        
        return {
            'matching_fields': list(plankart_set & bestemmelser_set),
            'only_in_plankart': list(plankart_set - bestemmelser_set),
            'only_in_bestemmelser': list(bestemmelser_set - plankart_set),
            'is_consistent': plankart_set == bestemmelser_set
        }