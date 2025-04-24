"""
SOSI File Handler Module

This module provides functionality for parsing SOSI files and extracting
regulatory fields from them.
"""

from pathlib import Path
from typing import Dict, List, Set
import re
from app.utils.logger import logger

class SosiParser:
    """Parser for SOSI files to extract regulatory fields."""
    
    def __init__(self):
        """Initialize the SOSI parser."""
        self.field_pattern = re.compile(r'(?:^|\s)(o_[A-Za-z]+\d*|f_[A-Za-z]+\d*|[A-Z]+\d*|H\d{3}|#\d+\s+SNØ)(?:\s|$)')
        self.zone_pattern = re.compile(r'\.SONE\s+(\S+)')
        self.objtype_pattern = re.compile(r'\.OBJTYPE\s+(\S+)')
        self.purpose_pattern = re.compile(r'\.FORMÅL\s+(\S+)')
        self.hensynssone_pattern = re.compile(r'\.HENSYNSSONE\s+(H\d{3})')
        
    def parse_content(self, content: str) -> Dict:
        """
        Parse SOSI content and extract fields.
        
        Args:
            content (str): SOSI file content
            
        Returns:
            Dict: Parsed SOSI data including fields
        """
        try:
            # Split content into lines and clean
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Initialize field sets
            fields = set()
            zones = set()
            purposes = set()
            hensynssoner = set()
            
            # Current object being processed
            current_object = {}
            
            for line in lines:
                # Check for new object start
                if line.startswith('.FLATE') or line.startswith('.KURVE'):
                    if current_object:
                        self._process_object(current_object, fields, zones, purposes, hensynssoner)
                    current_object = {'type': line[1:]}
                    continue
                
                # Add attributes to current object
                if line.startswith('.'):
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        attr = parts[0][1:]  # Remove the leading dot
                        value = parts[1].strip()
                        current_object[attr] = value
                        
                        # Direct field extraction from line
                        matches = self.field_pattern.findall(line)
                        fields.update(match.strip() for match in matches)
                        
                        # Check specific patterns
                        if zone_match := self.zone_pattern.search(line):
                            zones.add(zone_match.group(1))
                        if purpose_match := self.purpose_pattern.search(line):
                            purposes.add(purpose_match.group(1))
                        if hensynssone_match := self.hensynssone_pattern.search(line):
                            hensynssoner.add(hensynssone_match.group(1))
            
            # Process last object
            if current_object:
                self._process_object(current_object, fields, zones, purposes, hensynssoner)
            
            # Combine all fields
            all_fields = fields | zones | purposes | hensynssoner
            
            # Expand number ranges (e.g., o_GF1-3)
            expanded_fields = set()
            for field in all_fields:
                if range_match := re.match(r'([a-zA-Z_]+)(\d+)-(\d+)$', field):
                    prefix, start, end = range_match.groups()
                    try:
                        expanded_fields.update(
                            f"{prefix}{i}" for i in range(int(start), int(end) + 1)
                        )
                    except ValueError:
                        expanded_fields.add(field)
                else:
                    expanded_fields.add(field)
            
            logger.info(f"Extracted {len(expanded_fields)} fields from SOSI content")
            logger.debug(f"SOSI fields: {expanded_fields}")
            
            return {
                "fields": list(expanded_fields),
                "raw_content": content,
                "metadata": {
                    "zones": list(zones),
                    "purposes": list(purposes),
                    "hensynssoner": list(hensynssoner)
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing SOSI content: {str(e)}")
            return {"fields": [], "raw_content": content, "metadata": {}}
    
    def _process_object(self, obj: Dict, fields: Set, zones: Set, purposes: Set, hensynssoner: Set):
        """Process a SOSI object and extract relevant fields."""
        if 'SONE' in obj:
            zones.add(obj['SONE'])
        if 'FORMÅL' in obj:
            purposes.add(obj['FORMÅL'])
        if 'HENSYNSSONE' in obj:
            hensynssoner.add(obj['HENSYNSSONE'])
        
        # Look for field patterns in all object values
        for value in obj.values():
            if isinstance(value, str):
                matches = self.field_pattern.findall(value)
                fields.update(match.strip() for match in matches)

async def extract_fields_from_sosi(content: bytes) -> Set[str]:
    """
    Extract fields from SOSI file content.
    
    Args:
        content (bytes): Raw SOSI file content
        
    Returns:
        Set[str]: Set of extracted field names
    """
    try:
        # Decode content
        text_content = content.decode('utf-8')
        
        # Parse SOSI content
        parser = SosiParser()
        sosi_data = parser.parse_content(text_content)
        
        # Get all fields including metadata
        all_fields = set(sosi_data["fields"])
        metadata = sosi_data.get("metadata", {})
        
        # Add metadata fields if they match our patterns
        for field_list in metadata.values():
            all_fields.update(
                field for field in field_list 
                if parser.field_pattern.match(field)
            )
        
        logger.info(f"Extracted {len(all_fields)} total fields from SOSI file")
        return all_fields
        
    except Exception as e:
        logger.error(f"Error processing SOSI file: {str(e)}")
        return set() 