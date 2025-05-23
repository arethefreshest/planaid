#!/usr/bin/env python3
import logging
import json
from pathlib import Path
from typing import Set, Dict, Optional, List, Tuple
from parse_sosi import parse_sosi_to_docling, load_sosi_purpose_codes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_field_comparison(field: str) -> Optional[str]:
    """Normalize field names for comparison."""
    if not field:
        return None
    field = field.strip().upper()
    return field

class SOSIExtractor:
    """Class to handle SOSI field extraction"""
    def __init__(self, sosi_codes: Dict[str, str]):
        self.zones = set()  # Only named zones (FELTNAVN)
        self.text_fields = []  # List of (TEKST, STRENG) tuples from TEKST groups
        self.streng_values = set()  # All STRENG values found (including from RpPåskrift)
        self.arealformål_mapping = {}  # Dict mapping FELTNAVN to (code, purpose) tuple
        self.field_names = set()
        self.hensynssoner = set()
        self.sosi_codes = sosi_codes
        
    def extract_all_fields(self, doc) -> Dict:
        """Extract all relevant fields from SOSI document."""
        # Use doc.body if it exists, else use doc directly
        group = getattr(doc, "body", doc)
        self._process_group(group)
        
        # Convert arealformål_mapping to a list of dicts for JSON serialization
        arealformål_list = [
            {
                "feltnavn": feltnavn,
                "code": code,
                "purpose": self.sosi_codes.get(code, "Unknown purpose")
            }
            for feltnavn, code in self.arealformål_mapping.items()
        ]
        
        # Get unique STRENG values
        unique_streng_values = sorted(list(self.streng_values))
        
        # Create visible_annotations by validating against what's actually in STRENG values
        visible_annotations = {
            "zones": [],  # Zone labels that are confirmed visible
            "hensynssoner": [],  # Hensynssone labels that are confirmed visible
            "measurements": [],  # Area measurements and other numeric annotations
            "other": []  # Other visible annotations that don't fit above categories
        }
        
        # Helper function to normalize for comparison
        def normalize(s: str) -> str:
            return s.upper().strip()
        
        # Normalize all STRENG values for comparison
        normalized_streng = {normalize(s) for s in unique_streng_values}
        
        # Process each STRENG value
        for streng in unique_streng_values:
            norm_streng = normalize(streng)
            
            # Check if it's an area measurement
            if streng.startswith('A=') or streng.startswith('#'):
                visible_annotations["measurements"].append(streng)
                continue
                
            # Check if it matches a zone name
            if any(normalize(zone) == norm_streng for zone in self.field_names):
                visible_annotations["zones"].append(streng)
                continue
                
            # Check if it matches a hensynssone
            if any(normalize(sone) == norm_streng for sone in self.hensynssoner):
                visible_annotations["hensynssoner"].append(streng)
                continue
                
            # If it doesn't match any category but is in STRENG values, it's other
            visible_annotations["other"].append(streng)
        
        # Sort all lists for consistency
        for key in visible_annotations:
            visible_annotations[key] = sorted(visible_annotations[key])
        
        return {
            'zones': sorted(list(self.zones)),  # Only named zones
            'text_fields': self.text_fields,  # TEKST+STRENG pairs
            'streng_values': unique_streng_values,  # All unique STRENG values
            'visible_annotations': visible_annotations,  # Validated visible annotations
            'arealformål': arealformål_list,
            'field_names': sorted(list(self.field_names)),
            'hensynssoner': sorted(list(self.hensynssoner))
        }
    
    def _process_group(self, group):
        """Process a group to find target fields."""
        if not hasattr(group, 'name'):
            return  # Skip non-group items (like TextItem)
        logger.debug(f"Processing group: {group.name}")
        
        # Track FELTNAVN and RPAREALFORMÅL within the same group
        current_feltnavn = None
        current_arealformål = None
        
        # First pass to get FELTNAVN and RPAREALFORMÅL
        for child in group.children:
            if not hasattr(child, 'value'):
                continue
                
            value = str(child.value) if child.value is not None else None
            if not value:
                continue
                
            normalized = normalize_field_comparison(value)
            if not normalized:
                continue
                
            # Existing logic for quoted FELTNAVN
            if hasattr(child, 'name') and child.name == "FELTNAVN":
                logger.debug(f"Found FELTNAVN: {normalized}")
                current_feltnavn = normalized
                self.field_names.add(normalized)
                self.zones.add(normalized)
            # --- NEW LOGIC for unquoted FELTNAVN ---
            elif hasattr(child, 'name') and child.name not in (
                "OBJTYPE", "RPAREALFORMÅL", "NASJONALAREALPLANID", "KOMM", "PLANID", "VERTNIV", "EIERFORM", "OPPDATERINGSDATO",
                "IDENT", "LOKALID", "NAVNEROM", "VERSJONID", "REF", "NØ", "BESKRIVELSE", "HENSYNSONENAVN", "FØRSTEDIGITALISERINGSDATO",
                "KVALITET", "STRENG", "RPPÅSKRIFTTYPE", "BESTEMMELSEOMRNAVN", "SLUTT", "SYMBOL"
            ):
                # Heuristic: if the group name looks like a field identifier, add it
                potential_id = normalize_field_comparison(str(child.name))
                if potential_id:
                    # Optionally confirm: does this group have a TextItem child with the same text?
                    has_confirming_text = False
                    if hasattr(child, 'children'):
                        for sub_child in child.children:
                            if hasattr(sub_child, 'text') and sub_child.text == child.name:
                                has_confirming_text = True
                                break
                    if has_confirming_text:
                        logger.debug(f"Found unquoted FELTNAVN: {potential_id}")
                        self.field_names.add(potential_id)
                        self.zones.add(potential_id)
            elif hasattr(child, 'name') and child.name == "RPAREALFORMÅL":
                logger.debug(f"Found RPAREALFORMÅL: {normalized}")
                current_arealformål = normalized
                self.zones.add(normalized)
            elif hasattr(child, 'name') and child.name == "HENSYNSONENAVN":
                logger.debug(f"Found HENSYNSONENAVN: {normalized}")
                self.hensynssoner.add(normalized)
            elif hasattr(child, 'name') and child.name == "STRENG":
                logger.debug(f"Found STRENG: {normalized}")
                if current_feltnavn:
                    self.visible_annotations.setdefault(current_feltnavn, set()).add(normalized)
                if current_arealformål:
                    self.visible_annotations.setdefault(current_arealformål, set()).add(normalized)
                self.streng_values.add(normalized)
        
        # Second pass to process nested groups
        for child in group.children:
            if hasattr(child, 'children'):
                self._process_group(child)

def save_results_to_json(results: Dict, output_path: Path, sosi_filename: str):
    """Save extraction results to a JSON file."""
    # Create results directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on input SOSI file
    output_file = output_path / f"{sosi_filename}_extracted_fields.json"
    
    # Save to JSON file with nice formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_file}")
    return output_file

def main():
    """Main function to test SOSI extraction."""
    # Setup paths
    script_dir = Path(__file__).parent
    sosi_codes_csv = script_dir / "data/Reguleringsplan.csv"
    test_cases = [
        script_dir / "data/mock/sosi/case1/Evjetun_leirsted.sos",
        script_dir / "data/mock/sosi/case2/Kjetså_massetak.sos"
    ]
    
    # Load SOSI codes
    logger.info(f"Loading SOSI codes from {sosi_codes_csv}")
    sosi_codes = load_sosi_purpose_codes(sosi_codes_csv)
    if not sosi_codes:
        logger.error("Failed to load SOSI codes")
        return
    
    # Process each test case
    for sosi_file in test_cases:
        if not sosi_file.exists():
            logger.error(f"SOSI file not found: {sosi_file}")
            continue
            
        logger.info(f"\nProcessing SOSI file: {sosi_file.name}")
        
        # Parse SOSI to DoclingDocument
        doc = parse_sosi_to_docling(sosi_file, sosi_codes, sosi_file.stem)
        if not doc:
            logger.error("Failed to parse SOSI file")
            continue
            
        # Extract all fields
        extractor = SOSIExtractor(sosi_codes)
        results = extractor.extract_all_fields(doc)
        
        # Save results to JSON file
        output_dir = script_dir / "results" / "field_extractions"
        output_file = save_results_to_json(results, output_dir, sosi_file.stem)
        
        # Print results to console
        logger.info(f"\nResults for {sosi_file.name}:")
        logger.info(f"Total zones (FELTNAVN) found: {len(results['zones'])}")
        logger.info(f"Zones: {results['zones']}")
        logger.info(f"\nAll STRENG values found: {len(results['streng_values'])}")
        logger.info(f"STRENG values: {results['streng_values']}")
        if results['text_fields']:
            logger.info(f"\nTEKST/STRENG pairs found: {len(results['text_fields'])}")
            for tekst, streng in results['text_fields']:
                logger.info(f"TEKST: {tekst} | STRENG: {streng}")
        logger.info(f"\nArealformål mappings found: {len(results['arealformål'])}")
        for mapping in results['arealformål']:
            logger.info(f"FELTNAVN: {mapping['feltnavn']} -> Code: {mapping['code']} ({mapping['purpose']})")
        logger.info(f"\nField names found: {len(results['field_names'])}")
        logger.info(f"Field names: {results['field_names']}")
        logger.info(f"\nHensynssoner found: {len(results['hensynssoner'])}")
        logger.info(f"Hensynssoner: {results['hensynssoner']}")

if __name__ == "__main__":
    main() 