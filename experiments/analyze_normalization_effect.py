import sys
from pathlib import Path
# Add sys.path hacks to allow direct import from sibling files
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import logging
import asyncio
from typing import Dict, Set, Tuple, Any # Added Any
import re
from test_sosi_extraction import SOSIExtractor

# --- Configure logging (same as yours) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress DEBUG logs from LiteLLM and litellm
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# --- Path setup (same as yours) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Use resolve() for robustness
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'python_service'))
sys.path.append(str(PROJECT_ROOT / 'experiments')) # Ensure experiments itself is in path for imports if needed


# --- Import necessary components (same as yours, with robust DOCLING_AVAILABLE) ---
try:
    from experiments.parse_planbestemmelser import parse_planbestemmelser
    from experiments.parse_plankart import parse_plankart
    from experiments.parse_sosi import load_sosi_purpose_codes, parse_sosi_to_docling # Assuming this is the correct import
    DOCLING_PARSERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import Docling parsers from experiments: {e}. Docling method analysis will be limited.")
    DOCLING_PARSERS_AVAILABLE = False
    # Define dummy functions if parsers are not available
    def parse_planbestemmelser(path, doc_id): return None
    def parse_plankart(path, doc_id): return None
    def parse_sosi_to_docling(path, purpose_map, doc_id): return None
    def load_sosi_purpose_codes(csv_path): return {}


try:
    from python_service.app.llm.extractor import FieldExtractor
    from python_service.app.document.sosi_handler import SosiParser as LegacySosiParser
    from python_service.app.config import model_name # Ensure this is correctly importable or defined
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Legacy components (FieldExtractor, LegacySosiParser, model_name) not found: {e}. Legacy method analysis will be limited.")
    LEGACY_COMPONENTS_AVAILABLE = False
    FieldExtractor = None
    LegacySosiParser = None
    model_name = "dummy_model_name" # Provide a dummy if not available to avoid further errors

try:
    from docling_core.types.doc import (
        DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel,
        BoundingBox, ProvenanceItem, GroupLabel
    )
    # from docling_core.types.doc.document import ( # Not strictly needed for this script if above are sufficient
    #     NodeItem, SectionHeaderItem, PageItem, Size
    # )
    DOCLING_TYPES_AVAILABLE = True
except ImportError:
    logger.warning("docling-core types not found. Docling method analysis may not function as expected.")
    DOCLING_TYPES_AVAILABLE = False
    # Define dummy classes if docling-core is not available
    class DoclingDocument: pass
    class TextItem: pass
    class GroupItem: pass
    class DocItemLabel: TEXT = "text" # Define needed attributes

# --- Define test case and paths (similar to yours) ---
CASE1_NAME = "Evjetun_Leirsted_Case1"
CASE1_PATHS = {
    "plankart": PROJECT_ROOT / "experiments/data/mock/plankart/case1/Evjetun_leirsted.pdf",
    "bestemmelser": PROJECT_ROOT / "experiments/data/mock/planbestemmelser/case1/Evjetun_leirsted.pdf",
    "sosi": PROJECT_ROOT / "experiments/data/mock/sosi/case1/Evjetun_leirsted.sos"
}
PURPOSE_CODES_PATH = PROJECT_ROOT / "experiments/data/Reguleringsplan.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments/results/normalization_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- Normalization functions (same as yours) ---
def normalize_field_comparison(field: str) -> str | None: # Return type hint
    """Original normalization function"""
    if not field: return None
    field = str(field).strip().upper() # Ensure field is string
    # Don't remove prefixes as they might be meaningful
    field = re.sub(r'[ .\-]', '', field) # Remove spaces, dots, and hyphens
    
    # More permissive regex to match Norwegian planning IDs
    # Allow for letters, numbers, and underscores in various combinations
    if not re.fullmatch(r'(?:[A-ZÆØÅ0-9]+(?:_[A-ZÆØÅ0-9]+)*)', field): return None
    
    # Shorter exclusion list - only exclude very common non-identifiers
    excluded_terms = [
        "PLANID", "PS", "SIDE", "AV", "PBL", "TEK", "NN2000", "BYA",
        "PLANBESTEMMELSER", "FELLESBESTEMMELSER", "AREALFORMÅL"
    ]
    if field in excluded_terms or field.startswith("FIGUR") or field.startswith("TABELL"): return None
    if len(field) < 2 and not field.startswith("#"): return None # Filter out very short strings unless they are #ID
    return field

# --- Legacy Method Field Extraction ---
async def get_raw_and_normalized_legacy(file_path: Path, doc_type: str) -> Tuple[Set[str], Set[str]]:
    raw_fields: Set[str] = set()
    if not LEGACY_COMPONENTS_AVAILABLE:
        logger.warning(f"Legacy components not available for {doc_type}")
        return set(), set()

    try:
        if doc_type == "plankart" or doc_type == "bestemmelser":
            if FieldExtractor and model_name:
                extractor = FieldExtractor(model_name=model_name) # Assuming model_name is configured
                with open(file_path, 'rb') as f:
                    content = f.read()
                extracted_strings = await extractor.extract_fields(content)
                raw_fields = {str(s).strip() for s in extracted_strings if s}
            else:
                logger.warning(f"FieldExtractor or model_name not available for legacy {doc_type}")
        elif doc_type == "sosi":
            if LegacySosiParser:
                parser = LegacySosiParser(purpose_map={}) # Provide empty map if not used, or load it
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                sosi_data = parser.parse_content(content) # Ensure this method exists and works
                if sosi_data and "fields" in sosi_data and "zone_identifiers" in sosi_data["fields"]:
                    raw_fields = {str(s).strip() for s in sosi_data["fields"]["zone_identifiers"] if s}
                else:
                    logger.warning(f"Could not parse zone_identifiers from legacy SOSI for {file_path.name}")
            else:
                logger.warning("LegacySosiParser not available")
    except Exception as e:
        logger.error(f"Error in legacy extraction for {doc_type} ({file_path.name}): {e}", exc_info=True)

    normalized_fields = {norm_val for f in raw_fields if (norm_val := normalize_field_comparison(f))}
    return raw_fields, normalized_fields


# --- Docling Method Field Extraction ---

def expand_range(match):
    """Expand identifier ranges like GN1-GN4 to GN1, GN2, GN3, GN4."""
    prefix = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    return [f"{prefix}{i}" for i in range(start, end+1)]

def extract_identifiers_from_text(text):
    """Extract identifiers and expand ranges from a text string."""
    identifiers = set()
    # Find all patterns like F_TV1-4, GN1-GN4, o_AVG1-4, BAT1-2, etc.
    range_pattern = re.compile(r'([A-ZÆØÅa-zæøå_]+)([0-9]+)-([0-9]+)')
    for match in range_pattern.finditer(text):
        identifiers.update(expand_range(match))
    # Find all patterns like F_TV1, GN3, o_AVG2, BAT2, H140, etc.
    id_pattern = re.compile(r'[A-ZÆØÅa-zæøå_]+[0-9]+')
    identifiers.update(id_pattern.findall(text))
    return identifiers

def _extract_raw_from_docling_node(node, collected_raw_fields):
    """Recursively extract identifiers from Docling nodes."""
    # GroupItem: check .name and .children
    if hasattr(node, 'name') and node.name and node.name not in ("_root_", "body"):
        collected_raw_fields.update(extract_identifiers_from_text(str(node.name)))
    # TextItem: check .text and .orig
    if hasattr(node, 'text') and node.text:
        collected_raw_fields.update(extract_identifiers_from_text(str(node.text)))
    if hasattr(node, 'orig') and node.orig and node.orig != node.text:
        collected_raw_fields.update(extract_identifiers_from_text(str(node.orig)))
    # Recurse into children
    if hasattr(node, 'children'):
        for child in node.children:
            _extract_raw_from_docling_node(child, collected_raw_fields)
    # If the node has .groups or .texts (top-level DoclingDocument)
    if hasattr(node, 'groups'):
        for group in node.groups:
            _extract_raw_from_docling_node(group, collected_raw_fields)
    if hasattr(node, 'texts'):
        for text in node.texts:
            _extract_raw_from_docling_node(text, collected_raw_fields)


def get_raw_and_normalized_docling(file_path: Path, doc_type: str, sosi_codes_map: Dict = None) -> Tuple[Set[str], Set[str]]:
    if not DOCLING_PARSERS_AVAILABLE or not DOCLING_TYPES_AVAILABLE:
        logger.warning(f"Docling components not available for {doc_type}")
        return set(), set()

    doc = None
    try:
        if doc_type == "plankart":
            doc = parse_plankart(file_path, doc_id=file_path.stem)
        elif doc_type == "bestemmelser":
            doc = parse_planbestemmelser(file_path, doc_id=file_path.stem)
        elif doc_type == "sosi":
            if sosi_codes_map is None: sosi_codes_map = load_sosi_purpose_codes(PURPOSE_CODES_PATH)
            doc = parse_sosi_to_docling(file_path, purpose_map=sosi_codes_map, doc_id=file_path.stem)
    except Exception as e:
        logger.error(f"Error parsing {doc_type} with Docling ({file_path.name}): {e}", exc_info=True)
        return set(), set()

    raw_fields = set()
    if doc:
        if doc_type == "sosi":
            extractor = SOSIExtractor(sosi_codes_map)
            # Pass doc.body if available, otherwise pass doc directly
            doc_to_extract = doc.body if hasattr(doc, 'body') and doc.body is not None else doc
            results = extractor.extract_all_fields(doc_to_extract)
            raw_fields = set(results['zones']) | set(results['field_names']) | set(results['hensynssoner'])
            for category in results['visible_annotations'].values():
                raw_fields.update(category)
        else:
            _extract_raw_from_docling_node(doc, raw_fields)

    normalized_fields = {norm_val for f in raw_fields if (norm_val := normalize_field_comparison(f))}
    return raw_fields, normalized_fields


# --- Main Analysis Function ---
async def analyze_normalization_effect_for_case(case_name: str, case_paths: Dict[str, Path]):
    logger.info(f"\n--- Analyzing Normalization for Case: {case_name} ---")
    
    sosi_codes = load_sosi_purpose_codes(PURPOSE_CODES_PATH)
    if not PURPOSE_CODES_PATH.exists():
        logger.warning(f"SOSI Purpose codes file not found at {PURPOSE_CODES_PATH}. SOSI parsing may be affected.")

    data_for_case = {
        "case_name": case_name,
        "legacy_method": {"plankart": {}, "bestemmelser": {}, "sosi": {}, "all_docs_raw": set(), "all_docs_normalized": set()},
        "docling_method": {"plankart": {}, "bestemmelser": {}, "sosi": {}, "all_docs_raw": set(), "all_docs_normalized": set()}
    }

    # Legacy Method
    if LEGACY_COMPONENTS_AVAILABLE:
        logger.info("Processing with Legacy Method...")
        raw_lp_legacy, norm_lp_legacy = await get_raw_and_normalized_legacy(case_paths["plankart"], "plankart")
        raw_lb_legacy, norm_lb_legacy = await get_raw_and_normalized_legacy(case_paths["bestemmelser"], "bestemmelser")
        raw_ls_legacy, norm_ls_legacy = await get_raw_and_normalized_legacy(case_paths["sosi"], "sosi")  # Added await here

        data_for_case["legacy_method"]["plankart"]["raw_count"] = len(raw_lp_legacy)
        data_for_case["legacy_method"]["plankart"]["normalized_count"] = len(norm_lp_legacy)
        data_for_case["legacy_method"]["bestemmelser"]["raw_count"] = len(raw_lb_legacy)
        data_for_case["legacy_method"]["bestemmelser"]["normalized_count"] = len(norm_lb_legacy)
        data_for_case["legacy_method"]["sosi"]["raw_count"] = len(raw_ls_legacy)
        data_for_case["legacy_method"]["sosi"]["normalized_count"] = len(norm_ls_legacy)
        
        data_for_case["legacy_method"]["all_docs_raw"] = raw_lp_legacy | raw_lb_legacy | raw_ls_legacy
        data_for_case["legacy_method"]["all_docs_normalized"] = norm_lp_legacy | norm_lb_legacy | norm_ls_legacy
        
        data_for_case["legacy_method"]["common_raw_count"] = len(raw_lp_legacy & raw_lb_legacy & raw_ls_legacy)
        data_for_case["legacy_method"]["common_normalized_count"] = len(norm_lp_legacy & norm_lb_legacy & norm_ls_legacy)
    else:
        logger.warning("Legacy components not fully available. Skipping legacy method analysis.")


    # Docling Method
    if DOCLING_PARSERS_AVAILABLE and DOCLING_TYPES_AVAILABLE:
        logger.info("Processing with Docling Method...")
        raw_lp_docling, norm_lp_docling = get_raw_and_normalized_docling(case_paths["plankart"], "plankart")
        raw_lb_docling, norm_lb_docling = get_raw_and_normalized_docling(case_paths["bestemmelser"], "bestemmelser")
        raw_ls_docling, norm_ls_docling = get_raw_and_normalized_docling(case_paths["sosi"], "sosi", sosi_codes)

        data_for_case["docling_method"]["plankart"]["raw_count"] = len(raw_lp_docling)
        data_for_case["docling_method"]["plankart"]["normalized_count"] = len(norm_lp_docling)
        data_for_case["docling_method"]["bestemmelser"]["raw_count"] = len(raw_lb_docling)
        data_for_case["docling_method"]["bestemmelser"]["normalized_count"] = len(norm_lb_docling)
        data_for_case["docling_method"]["sosi"]["raw_count"] = len(raw_ls_docling)
        data_for_case["docling_method"]["sosi"]["normalized_count"] = len(norm_ls_docling)

        data_for_case["docling_method"]["all_docs_raw"] = raw_lp_docling | raw_lb_docling | raw_ls_docling
        data_for_case["docling_method"]["all_docs_normalized"] = norm_lp_docling | norm_lb_docling | norm_ls_docling

        data_for_case["docling_method"]["common_raw_count"] = len(raw_lp_docling & raw_lb_docling & raw_ls_docling)
        data_for_case["docling_method"]["common_normalized_count"] = len(norm_lp_docling & norm_lb_docling & norm_ls_docling) # This is the '10' for Case 1
    else:
        logger.warning("Docling components not fully available. Skipping Docling method analysis.")

    # --- Output Results ---
    print(f"\n--- Summary for {case_name} ---")
    if LEGACY_COMPONENTS_AVAILABLE:
        lm = data_for_case["legacy_method"]
        print("\nLegacy Method:")
        print(f"  Total Unique Raw Identifiers (across all docs): {len(lm['all_docs_raw'])}")
        print(f"  Total Unique Normalized Identifiers (across all docs): {len(lm['all_docs_normalized'])}")
        print(f"  Common Raw Identifiers (P & B & S): {lm['common_raw_count']}")
        print(f"  Common Normalized Identifiers (P & B & S): {lm['common_normalized_count']}")
        reduction_total_legacy = len(lm['all_docs_raw']) - len(lm['all_docs_normalized'])
        percent_reduction_legacy = (reduction_total_legacy / len(lm['all_docs_raw'])) * 100 if len(lm['all_docs_raw']) > 0 else 0
        print(f"  Reduction in Total Unique Identifiers by Normalization: {reduction_total_legacy} ({percent_reduction_legacy:.1f}%)")
        
        if lm['common_raw_count'] > 0:
            improvement_common_legacy = ((lm['common_normalized_count'] - lm['common_raw_count']) / lm['common_raw_count']) * 100
            print(f"  Improvement in Common Entity Identification: {improvement_common_legacy:.1f}%")
        else:
            print(f"  Improvement in Common Entity Identification: N/A (0 common raw entities)")


    if DOCLING_PARSERS_AVAILABLE and DOCLING_TYPES_AVAILABLE:
        dm = data_for_case["docling_method"]
        print("\nDocling Method:")
        print(f"  Total Unique Raw Identifiers (across all docs): {len(dm['all_docs_raw'])}")
        print(f"  Total Unique Normalized Identifiers (across all docs): {len(dm['all_docs_normalized'])}")
        print(f"  Common Raw Identifiers (P & B & S): {dm['common_raw_count']}")
        print(f"  Common Normalized Identifiers (P & B & S): {dm['common_normalized_count']}") # This should be 10 for Case 1
        reduction_total_docling = len(dm['all_docs_raw']) - len(dm['all_docs_normalized'])
        percent_reduction_docling = (reduction_total_docling / len(dm['all_docs_raw'])) * 100 if len(dm['all_docs_raw']) > 0 else 0
        print(f"  Reduction in Total Unique Identifiers by Normalization: {reduction_total_docling} ({percent_reduction_docling:.1f}%)")

        if dm['common_raw_count'] > 0:
            improvement_common_docling = ((dm['common_normalized_count'] - dm['common_raw_count']) / dm['common_raw_count']) * 100
            print(f"  Improvement in Common Entity Identification: {improvement_common_docling:.1f}%")
        else:
            print(f"  Improvement in Common Entity Identification: N/A (0 common raw entities)")
            
    # Save data_for_case to JSON
    with open(OUTPUT_DIR / f"{case_name}_normalization_analysis.json", 'w', encoding='utf-8') as f:
        # Convert sets to lists for JSON serialization
        def convert_sets_to_lists(d):
            for k, v in d.items():
                if isinstance(v, set):
                    d[k] = sorted(list(v))
                elif isinstance(v, dict):
                    convert_sets_to_lists(v)
            return d
        json.dump(convert_sets_to_lists(data_for_case), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed normalization analysis for {case_name} to JSON.")
    return data_for_case


async def main():
    results_case1 = await analyze_normalization_effect_for_case(CASE1_NAME, CASE1_PATHS)
    # You can add calls for other cases if needed.
    # For now, we just need the numbers for Case 1 for the thesis text.

if __name__ == "__main__":
    asyncio.run(main())