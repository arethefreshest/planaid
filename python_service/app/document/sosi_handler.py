"""
SOSI File Handler Module

This module provides functionality for parsing SOSI files and extracting
regulatory fields from them.
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Any
import re
from app.utils.logger import logger
import json
import os

# Setup logging directory
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'metrics')
SOSI_METRICS_DIR = os.path.join(METRICS_DIR, 'sosi')
os.makedirs(SOSI_METRICS_DIR, exist_ok=True)

class SosiNode:
    """Represents a node in the SOSI document tree"""
    def __init__(self, key: str = "", value: str = "", level: int = 0):
        self.key = key
        self.value = value
        self.level = level
        self.children: List[SosiNode] = []
        self.parent: Optional[SosiNode] = None
        self.attributes: Dict[str, Any] = {}
        
    def add_child(self, node: 'SosiNode') -> None:
        """Add a child node and set its parent"""
        node.parent = self
        self.children.append(node)
        
    def to_dict(self) -> Dict:
        """Convert node to dictionary representation"""
        result = {
            "key": self.key,
            "value": self.value,
            "level": self.level,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children]
        }
        return result

class SosiParser:
    """Parser for SOSI files to extract regulatory fields."""
    
    def __init__(self, purpose_map: Optional[Dict[str, str]] = None):
        """Initialize the SOSI parser."""
        self.purpose_map = purpose_map or {}
        self.root = SosiNode("ROOT", "", 0)
        self.current_node = self.root
        self.field_names: Set[str] = set()
        self.hensynssoner: Set[str] = set()
        self.purposes: Set[str] = set()
        
    def parse_file(self, file_path: Path) -> Dict:
        """Parse a SOSI file from disk."""
        logger.info(f"Parsing SOSI file: {file_path}")
        try:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                raw_content = f.read()
                content = raw_content.encode('ISO-8859-1').decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Failed to decode file: {e}")
            raise e
        return self.parse_content(content)

    def handle_node(self, line: str, level: int) -> None:
        """Process a single line and create/update nodes accordingly"""
        if ' ' in line:
            key, value = line.split(' ', 1)
            key = key.strip()
            value = value.strip()
            
            # Create new node
            new_node = SosiNode(key, value, level)
            
            # Handle special fields
            if key == "FELTNAVN":
                self.field_names.add(value)
            elif key == "HENSYNSONENAVN":
                self.hensynssoner.add(value)
            elif key == "RPAREALFORMÅL":
                self.purposes.add(value)
                new_node.attributes["purpose_name"] = self.purpose_map.get(value, "ukjent")
            
            # Find correct parent based on level
            while self.current_node.level >= level and self.current_node.parent:
                self.current_node = self.current_node.parent
                
            self.current_node.add_child(new_node)
            self.current_node = new_node
        else:
            # Handle lines without values (like KURVE, FLATE etc)
            new_node = SosiNode(line.strip(), "", level)
            while self.current_node.level >= level and self.current_node.parent:
                self.current_node = self.current_node.parent
            self.current_node.add_child(new_node)
            self.current_node = new_node

    def parse_content(self, content: str) -> Dict:
        """
        Parse SOSI content into a hierarchical structure.
        
        Args:
            content (str): SOSI file content
            
        Returns:
            Dict: Parsed SOSI data including hierarchical structure and fields
        """
        try:
            # Reset parser state
            self.root = SosiNode("ROOT", "", 0)
            self.current_node = self.root
            self.field_names.clear()
            self.hensynssoner.clear()
            self.purposes.clear()

            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('!'):
                    continue

                # Handle indentation levels
                level_match = re.match(r'^(\.+)', line)
                level = len(level_match.group(1)) if level_match else 0
                line = line[level_match.end():] if level_match else line
                
                self.handle_node(line, level)

            # Create result structure
            result = {
                "structure": self.root.to_dict(),
                "fields": {
                    "zone_identifiers": sorted(list(self.field_names)),
                    "hensynssoner": sorted(list(self.hensynssoner)),
                    "purposes": sorted(list(self.purposes))
                },
                "metadata": {
                    "total_fields": len(self.field_names),
                    "total_hensynssoner": len(self.hensynssoner),
                    "total_purposes": len(self.purposes)
                }
            }
            
            # Log the structured output
            log_file = os.path.join(SOSI_METRICS_DIR, 'sosi_structure.json')
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"SOSI structure logged to {log_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing SOSI content: {str(e)}")
            raise

async def extract_fields_from_sosi(content: bytes) -> Dict:
    """Extract fields and structure from SOSI content."""
    try:
        # Convert bytes to string
        content_str = content.decode('ISO-8859-1').encode('utf-8', errors='replace').decode('utf-8')
        
        # Create parser and parse content
        parser = SosiParser()
        sosi_data = parser.parse_content(content_str)
        
        # Log results
        logger.info(f"Extracted {sosi_data['metadata']['total_fields']} fields, "
                   f"{sosi_data['metadata']['total_hensynssoner']} hensynssoner, "
                   f"and {sosi_data['metadata']['total_purposes']} purposes from SOSI content")
        
        # Log detailed field information
        logger.info("Zone identifiers found: " + ", ".join(sosi_data["fields"]["zone_identifiers"]))
        if sosi_data["fields"]["hensynssoner"]:
            logger.info("Hensynssoner found: " + ", ".join(sosi_data["fields"]["hensynssoner"]))
        if sosi_data["fields"]["purposes"]:
            logger.info("Purposes found: " + ", ".join(sosi_data["fields"]["purposes"]))
        
        return sosi_data
    except Exception as e:
        logger.error(f"Error extracting fields from SOSI: {str(e)}")
        return {
            "structure": {},
            "fields": {
                "zone_identifiers": [],
                "hensynssoner": [],
                "purposes": []
            },
            "metadata": {
                "total_fields": 0,
                "total_hensynssoner": 0,
                "total_purposes": 0
            }
        } 