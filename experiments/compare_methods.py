# experiments/compare_methods.py (Updated)
import json
import time
import logging
import requests
import pandas as pd
import httpx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, List, Any
import re
import asyncio
import os
import sys
import csv

# Configure logging - set to INFO by default
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Disable debug logging for noisy libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('litellm').setLevel(logging.WARNING)
logging.getLogger('litellm.llms').setLevel(logging.WARNING)
logging.getLogger('litellm.utils').setLevel(logging.WARNING)
logging.getLogger('extractthinker').setLevel(logging.WARNING)
logging.getLogger('extractthinker.core').setLevel(logging.WARNING)
logging.getLogger('extractthinker.llm').setLevel(logging.WARNING)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'python_service'))

# Import Docling parsers
from experiments.parse_planbestemmelser import parse_planbestemmelser
from experiments.parse_plankart import parse_plankart
from experiments.parse_sosi import load_sosi_purpose_codes, parse_sosi_to_docling

# Import Docling types
try:
    from docling_core.types.doc import (
        DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel,
        BoundingBox, ProvenanceItem, GroupLabel
    )
    from docling_core.types.doc.document import (
        NodeItem, SectionHeaderItem, PageItem, Size
    )
except ImportError as e:
    print(f"ERROR: Failed to import docling-core types: {e}")
    print("Make sure docling-core is installed and in your PYTHONPATH")
    sys.exit(1)

# Import Legacy components
try:
    from python_service.app.llm.extractor import FieldExtractor
    from python_service.app.config import model_name
    from python_service.app.document.sosi_handler import SosiParser as LegacySosiParser
except ImportError as e:
    print(f"WARN: Could not import legacy components: {e}.")
    FieldExtractor = None
    LegacySosiParser = None
    model_name = None

# Add path setup near the top, after imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PYTHON_SERVICE_DIR = PROJECT_ROOT / "python_service"
DATA_DIR = SCRIPT_DIR / "data"
COMPARISON_OUTPUT_DIR = SCRIPT_DIR / "results/comparison"

# Define test cases
CASES = {
    "case1": {
        "plankart": DATA_DIR / "mock/plankart/case1/Evjetun_leirsted.pdf",
        "bestemmelser": DATA_DIR / "mock/planbestemmelser/case1/Evjetun_leirsted.pdf",
        "sosi": DATA_DIR / "mock/sosi/case1/Evjetun_leirsted.sos"
    },
    "case2": {
        "plankart": DATA_DIR / "mock/plankart/case2/Kjetså_massetak.pdf",
        "bestemmelser": DATA_DIR / "mock/planbestemmelser/case2/Kjetså_massetak.pdf",
        "sosi": DATA_DIR / "mock/sosi/case2/Kjetså_massetak.sos"
    }
}

# Add python_service to path
sys.path.append(str(PYTHON_SERVICE_DIR))

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
BASE_DATA_DIR = SCRIPT_DIR / "data"
MOCK_DATA_DIR = BASE_DATA_DIR / "mock"
RESULTS_DIR = SCRIPT_DIR / "results" # Go up one level from experiments
DOCLING_OUTPUT_DIR = RESULTS_DIR / "docling_parsed"
COMPARISON_OUTPUT_DIR = RESULTS_DIR / "comparison_results"
NER_SERVICE_URL = os.getenv('NER_SERVICE_URL', "http://157.230.21.199:8001")
if not NER_SERVICE_URL.endswith('/api/extract-fields'):
    NER_SERVICE_URL = f"{NER_SERVICE_URL}/api/extract-fields"
SOSI_CODES_CSV = BASE_DATA_DIR / "Reguleringsplan.csv"  # Updated path to match actual location

# Create organized output directories
OUTPUT_DIRS = {
    "comparison": COMPARISON_OUTPUT_DIR,
    "plots": COMPARISON_OUTPUT_DIR / "plots",
    "error_analysis": COMPARISON_OUTPUT_DIR / "error_analysis",
    "statistical": COMPARISON_OUTPUT_DIR / "statistical",
    "semantic": COMPARISON_OUTPUT_DIR / "semantic",
    "field_analysis": COMPARISON_OUTPUT_DIR / "field_analysis",
    "summary": COMPARISON_OUTPUT_DIR / "summary"
}

# Create all output directories
for dir_path in OUTPUT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def normalize_field_comparison(field: str) -> Optional[str]:
    if not field: return None
    field = field.strip().upper()
    field = re.sub(r'^[OF]_', '', field)
    field = re.sub(r'[ .-]', '', field)
    if not re.fullmatch(r'(?:H\d+|#\d+|[A-ZÆØÅ]+[0-9]*)', field): return None
    if field in ["PLANID", "PS", "SIDE", "AV", "PBL", "TEK", "NN2000", "BYA", "PLANBESTEMMELSER", "FELLESBESTEMMELSER", "AREALFORMÅL"]: return None
    return field

def extract_zones_from_docling_plankart(doc: DoclingDocument) -> Set[str]:
    zones = set()
    ZONE_ID_REGEX_COMPARE = re.compile(r"^(?:[fo]_)?([A-ZÆØÅ]+)\d+(?:-\d+)?$|^#\d+$")
    if hasattr(doc, 'texts'):
        for item in doc.texts:
            # Ensure item has necessary attributes before accessing them
            if hasattr(item, 'label') and item.label == DocItemLabel.TEXT and hasattr(item, 'text') and item.text:
                if ZONE_ID_REGEX_COMPARE.match(item.text.strip()):
                    normalized = normalize_field_comparison(item.text)
                    if normalized: zones.add(normalized)
    return zones

def extract_zones_from_docling_bestemmelser(doc: DoclingDocument) -> Set[str]:
     zones = set()
     # Make regex slightly more robust to catch edge cases if needed
     ZONE_MENTION_REGEX = re.compile(r'\b((?:[of]_)?(?:[A-ZÆØÅ]+)\d+(?:-\d+)?(?:/[A-ZÆØÅ\d]+)*|#\d+|H\d{3,})\b')
     if hasattr(doc, 'texts'):
         for item in doc.texts:
              if hasattr(item, 'text') and item.text:
                   matches = ZONE_MENTION_REGEX.findall(item.text)
                   for match in matches:
                        normalized = normalize_field_comparison(match)
                        if normalized: zones.add(normalized)
     return zones

def extract_zones_from_docling_sosi(doc: DoclingDocument) -> Set[str]:
    """
    Extract all relevant fields from SOSI document using the improved extraction method.
    Returns a set of normalized field names.
    """
    class SOSIExtractor:
        """Class to handle SOSI field extraction"""
        def __init__(self):
            self.zones = set()  # Only named zones (FELTNAVN)
            self.text_fields = []  # List of (TEKST, STRENG) tuples from TEKST groups
            self.streng_values = []  # All STRENG values found (including from RpPåskrift)
            self.arealformål_mapping = {}  # Dict mapping FELTNAVN to code
            self.field_names = set()
            self.hensynssoner = set()
            
        def extract_all_fields(self, doc) -> Set[str]:
            """Extract all relevant fields and return normalized set of field names."""
            self._process_group(doc.body)
            
            # Combine all relevant fields into a single set
            all_fields = set()
            
            # Add field names (FELTNAVN)
            all_fields.update(self.field_names)
            
            # Add hensynssoner
            all_fields.update(self.hensynssoner)
            
            # Add STRENG values that represent visible annotations
            for streng in self.streng_values:
                normalized = normalize_field_comparison(streng)
                if normalized:
                    all_fields.add(normalized)
            
            return all_fields
        
        def _process_group(self, group):
            """Process a group to find target fields."""
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
                    
                if child.name == "FELTNAVN":
                    current_feltnavn = normalized
                elif child.name == "RPAREALFORMÅL":
                    current_arealformål = normalized
                elif child.name == "STRENG":  # Direct STRENG values
                    self.streng_values.append(normalized)
            
            # If we found both in the same group, add to mapping
            if current_feltnavn and current_arealformål:
                self.arealformål_mapping[current_feltnavn] = current_arealformål
                self.field_names.add(current_feltnavn)
            elif current_feltnavn:
                self.field_names.add(current_feltnavn)
            
            # Special handling for TEKST groups that contain numbered children
            if group.name == "TEKST":
                for numbered_child in group.children:
                    if hasattr(numbered_child, 'children'):
                        # Look for STRENG within the numbered child's children
                        for sub_child in numbered_child.children:
                            if sub_child.name == "STRENG" and hasattr(sub_child, 'value') and sub_child.value:
                                streng_value = str(sub_child.value)
                                normalized = normalize_field_comparison(streng_value)
                                if normalized:
                                    self.streng_values.append(normalized)
            
            # Handle HENSYNSONENAVN
            for child in group.children:
                if not hasattr(child, 'value'):
                    continue
                    
                value = str(child.value) if child.value is not None else None
                if not value:
                    continue
                    
                normalized = normalize_field_comparison(value)
                if not normalized:
                    continue
                    
                if child.name == "HENSYNSONENAVN":
                    self.hensynssoner.add(normalized)
            
            # Process children recursively
            for child in group.children:
                if hasattr(child, 'children'):
                    self._process_group(child)
    
    # Create extractor and process document
    extractor = SOSIExtractor()
    return extractor.extract_all_fields(doc)

# --- Legacy Method ---
async def run_legacy_method(plankart_path: Path, bestemmelser_path: Path, sosi_path: Optional[Path]) -> Dict:
    logger.info("Running Legacy Method Simulation...")
    results = {"plankart": set(), "bestemmelser": set(), "sosi": set(), "error": None, "time": 0.0}
    start_time = time.time()
    errors = []

    # 1. Plankart (LLM Extractor - Ensure FieldExtractor was imported)
    if FieldExtractor and model_name:
        try:
            extractor_llm = FieldExtractor(model_name)
            with open(plankart_path, 'rb') as f: content = f.read()
            extracted_set = await extractor_llm.extract_fields(content) # Ensure this is awaited
            results["plankart"] = {normalize_field_comparison(f) for f in extracted_set if normalize_field_comparison(f)}
            logger.info(f"Legacy Plankart Extracted (LLM): {len(results['plankart'])} fields")
        except Exception as e: logger.error(f"Legacy Plankart Extraction failed: {e}", exc_info=True); errors.append(f"Plankart Error: {e}")
    else: logger.warning("Legacy FieldExtractor (LLM) not available."); errors.append("Legacy FieldExtractor (LLM) not available.")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(bestemmelser_path, 'rb') as f:
                files = {'file': (bestemmelser_path.name, f, 'application/pdf')}
                logger.info(f"Calling NER service at {NER_SERVICE_URL} for {bestemmelser_path.name}")
                response = await client.post(NER_SERVICE_URL, files=files)
                response.raise_for_status()
                ner_fields = response.json().get('fields', [])
                results["bestemmelser"] = {normalize_field_comparison(f) for f in ner_fields if normalize_field_comparison(f)}
                logger.info(f"Legacy Bestemmelser Extracted (NER): {len(results['bestemmelser'])} fields")
    except Exception as e: logger.error(f"Legacy Bestemmelser Extraction (NER) failed: {e}", exc_info=True); errors.append(f"Bestemmelser Error: {e}")

    if LegacySosiParser and sosi_path and sosi_path.exists():
        try:
            sosi_codes_map = load_sosi_purpose_codes(SOSI_CODES_CSV)
            legacy_sosi_parser = LegacySosiParser(purpose_map=sosi_codes_map)
            parsed_sosi = legacy_sosi_parser.parse_file(sosi_path)
            sosi_zones = parsed_sosi.get("fields", {}).get("zone_identifiers", [])
            sosi_hensyn = parsed_sosi.get("fields", {}).get("hensynssoner", [])
            results["sosi"] = {normalize_field_comparison(f) for f in sosi_zones if normalize_field_comparison(f)} | \
                              {normalize_field_comparison(h) for h in sosi_hensyn if normalize_field_comparison(h)}
            logger.info(f"Legacy SOSI Extracted: {len(results['sosi'])} fields/zones")
        except Exception as e: logger.error(f"Legacy SOSI Parsing failed: {e}", exc_info=True); errors.append(f"SOSI Error: {e}")
    elif not LegacySosiParser: logger.warning("Legacy SOSI Parser not available."); errors.append("Legacy SOSI Parser not available.")
    elif not sosi_path or not sosi_path.exists(): logger.warning("SOSI file not provided/found for legacy method.")

    results["error"] = "; ".join(errors) if errors else None
    results["time"] = time.time() - start_time
    return results


# --- Docling Method Execution ---
def run_docling_method(plankart_path: Path, bestemmelser_path: Path, sosi_path: Optional[Path], sosi_codes_map: Dict) -> Dict:
    logger.info("Running Docling Method...")
    results = {"plankart": set(), "bestemmelser": set(), "sosi": set(), "error": None, "time": 0.0}
    start_time = time.time()
    errors = []

    # 1. Parse Plankart
    try:
        doc_plankart = parse_plankart(plankart_path, plankart_path.stem)
        if doc_plankart:
             results["plankart"] = extract_zones_from_docling_plankart(doc_plankart)
             logger.info(f"Docling Plankart Parsed: {len(results['plankart'])} zones identified")
             doc_plankart.save_as_json(DOCLING_OUTPUT_DIR / f"{doc_plankart.name}_structure.json", image_mode='placeholder')
        else: raise ValueError("Plankart parsing returned None")
    except Exception as e:
        logger.error(f"Docling Plankart Parsing failed: {e}", exc_info=True)
        errors.append(f"Plankart Error: {e}")

    # 2. Parse Bestemmelser
    try:
        doc_bestemmelser = parse_planbestemmelser(bestemmelser_path, bestemmelser_path.stem)
        if doc_bestemmelser:
            results["bestemmelser"] = extract_zones_from_docling_bestemmelser(doc_bestemmelser)
            logger.info(f"Docling Bestemmelser Parsed: {len(results['bestemmelser'])} potential zones mentioned")
            doc_bestemmelser.save_as_json(DOCLING_OUTPUT_DIR / f"{doc_bestemmelser.name}_structure.json")
        else: raise ValueError("Planbestemmelser parsing returned None")
    except Exception as e:
        logger.error(f"Docling Bestemmelser Parsing failed: {e}", exc_info=True)
        errors.append(f"Bestemmelser Error: {e}")

    # 3. Parse SOSI
    if sosi_path and sosi_path.exists():
        try:
            doc_sosi = parse_sosi_to_docling(sosi_path, sosi_codes_map, sosi_path.stem)
            if doc_sosi:
                results["sosi"] = extract_zones_from_docling_sosi(doc_sosi)
                logger.info(f"Docling SOSI Parsed: {len(results['sosi'])} zones identified")
                doc_sosi.save_as_json(DOCLING_OUTPUT_DIR / f"{doc_sosi.name}_structure.json")
            else: raise ValueError("SOSI parsing returned None")
        except Exception as e:
            logger.error(f"Docling SOSI Parsing failed: {e}", exc_info=True)
            errors.append(f"SOSI Error: {e}")
    else:
         logger.warning("SOSI file not provided/found for docling method.")

    results["error"] = "; ".join(errors) if errors else None
    results["time"] = time.time() - start_time
    return results


# --- Comparison Logic ---
def compare_results(legacy_res: Dict, docling_res: Dict, case_id: str) -> Dict:
    logger.info(f"Comparing results for case: {case_id}")
    comparison = {
        "case_id": case_id,
        "times": {"legacy": legacy_res["time"], "docling": docling_res["time"]},
        "errors": {"legacy": legacy_res["error"], "docling": docling_res["error"]},
        "field_counts": {
            "legacy_plankart": len(legacy_res.get("plankart", set())), # Use get with default
            "docling_plankart": len(docling_res.get("plankart", set())),
            "legacy_bestemmelser": len(legacy_res.get("bestemmelser", set())),
            "docling_bestemmelser": len(docling_res.get("bestemmelser", set())),
            "legacy_sosi": len(legacy_res.get("sosi", set())),
            "docling_sosi": len(docling_res.get("sosi", set())),
        },
        "set_comparisons": {}, "consistency": {} }

    for doc_type in ["plankart", "bestemmelser", "sosi"]:
        legacy_set = legacy_res.get(doc_type, set())
        docling_set = docling_res.get(doc_type, set())
        union_len = len(legacy_set | docling_set)
        comparison["set_comparisons"][doc_type] = {
            "only_in_legacy": sorted(list(legacy_set - docling_set)),
            "only_in_docling": sorted(list(docling_set - legacy_set)),
            "common": sorted(list(legacy_set & docling_set)),
            "jaccard_similarity": len(legacy_set & docling_set) / union_len if union_len > 0 else 1.0
        }

    lp, lb, ls = legacy_res.get("plankart", set()), legacy_res.get("bestemmelser", set()), legacy_res.get("sosi", set())
    legacy_all = lp | lb | ls if ls else lp | lb
    legacy_matching = lp & lb & ls if ls else lp & lb
    comparison["consistency"]["legacy"] = {
        "matching": sorted(list(legacy_matching)),
        "only_plankart": sorted(list(lp - lb - (ls if ls else set()))),
        "only_bestemmelser": sorted(list(lb - lp - (ls if ls else set()))),
        "only_sosi": sorted(list(ls - lp - lb if ls else set())),
        "is_consistent": len(legacy_all - legacy_matching) == 0 if legacy_all else True
    }

    dp, db, ds = docling_res.get("plankart", set()), docling_res.get("bestemmelser", set()), docling_res.get("sosi", set())
    docling_all = dp | db | ds if ds else dp | db
    docling_matching = dp & db & ds if ds else dp & db
    comparison["consistency"]["docling"] = {
        "matching": sorted(list(docling_matching)),
        "only_plankart": sorted(list(dp - db - (ds if ds else set()))),
        "only_bestemmelser": sorted(list(db - dp - (ds if ds else set()))),
        "only_sosi": sorted(list(ds - dp - db if ds else set())),
        "is_consistent": len(docling_all - docling_matching) == 0 if docling_all else True
    }
    return comparison

def create_comparison_plots(df_summary: pd.DataFrame, output_dir: Path):
    """Create enhanced visualization plots for the comparison results"""
    if df_summary.empty:
        logger.warning("No data available for plotting")
        return
        
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use default style instead of seaborn
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': [12, 8],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    try:
        # 1. Field Counts Comparison (Stacked Bar Chart)
        fig, ax = plt.subplots()
        x = np.arange(len(df_summary))
        width = 0.35
        
        # Calculate stacked values
        legacy_plankart = df_summary['legacy_plankart_fields']
        legacy_bestemm = df_summary['legacy_bestemm_fields']
        legacy_sosi = df_summary['legacy_sosi_fields']
        docling_plankart = df_summary['docling_plankart_fields']
        docling_bestemm = df_summary['docling_bestemm_fields']
        docling_sosi = df_summary['docling_sosi_fields']
        
        # Plot stacked bars
        ax.bar(x - width/2, legacy_plankart, width, label='Plankart (Legacy)', color='lightcoral')
        ax.bar(x - width/2, legacy_bestemm, width, bottom=legacy_plankart, label='Bestemmelser (Legacy)', color='salmon')
        ax.bar(x - width/2, legacy_sosi, width, bottom=legacy_plankart+legacy_bestemm, label='SOSI (Legacy)', color='indianred')
        
        ax.bar(x + width/2, docling_plankart, width, label='Plankart (Docling)', color='lightblue')
        ax.bar(x + width/2, docling_bestemm, width, bottom=docling_plankart, label='Bestemmelser (Docling)', color='skyblue')
        ax.bar(x + width/2, docling_sosi, width, bottom=docling_plankart+docling_bestemm, label='SOSI (Docling)', color='steelblue')
        
        ax.set_xlabel('Cases')
        ax.set_ylabel('Number of Fields')
        ax.set_title('Field Count Comparison by Document Type')
        ax.set_xticks(x)
        ax.set_xticklabels(df_summary.index, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'field_counts_stacked.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Processing Time Comparison (Log Scale)
        fig, ax = plt.subplots()
        x = np.arange(len(df_summary))
        width = 0.35
        
        ax.bar(x - width/2, df_summary['legacy_time'], width, label='Legacy Method', color='lightcoral')
        ax.bar(x + width/2, df_summary['docling_time'], width, label='Docling Method', color='lightblue')
        
        ax.set_xlabel('Cases')
        ax.set_ylabel('Processing Time (seconds)')
        ax.set_title('Processing Time Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df_summary.index, rotation=45)
        ax.set_yscale('log')  # Use log scale for better visualization
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'processing_times.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Jaccard Similarity Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        similarity_data = df_summary[['plankart_jaccard', 'bestemm_jaccard', 'sosi_jaccard']].apply(
            lambda x: x.str.replace('N/A', '0').astype(float)
        )
        
        sns.heatmap(similarity_data, annot=True, cmap='YlOrRd', vmin=0, vmax=1, 
                   xticklabels=['Plankart', 'Bestemmelser', 'SOSI'],
                   yticklabels=df_summary.index)
        
        plt.title('Jaccard Similarity Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / 'jaccard_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance Improvement Radar Chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate performance metrics
        speed_improvement = df_summary['legacy_time'] / df_summary['docling_time']
        accuracy_improvement = similarity_data.mean(axis=1)
        
        # Prepare radar chart data
        categories = ['Speed', 'Plankart Accuracy', 'Bestemmelser Accuracy', 'SOSI Accuracy']
        values = np.array([
            speed_improvement.mean(),
            similarity_data['plankart_jaccard'].mean(),
            similarity_data['bestemm_jaccard'].mean(),
            similarity_data['sosi_jaccard'].mean()
        ])
        
        # Normalize values for radar chart
        values = values / values.max()
        
        # Plot radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # Complete the polygon
        angles = np.concatenate((angles, [angles[0]]))  # Complete the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Performance Improvement Overview')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
        logger.error("Available columns: " + ", ".join(df_summary.columns))

def generate_summary_report(df_summary: pd.DataFrame, output_dir: Path) -> Path:
    """Generate an enhanced summary report in LaTeX format."""
    report_path = output_dir / "summary_report.tex"
    
    # Convert time columns to numeric, handling any string values
    df_summary['legacy_time'] = pd.to_numeric(df_summary['legacy_time'], errors='coerce')
    df_summary['docling_time'] = pd.to_numeric(df_summary['docling_time'], errors='coerce')
    
    # Calculate averages and improvements
    avg_legacy_time = df_summary['legacy_time'].mean()
    avg_docling_time = df_summary['docling_time'].mean()
    speed_improvement = avg_legacy_time / avg_docling_time
    
    # Calculate field extraction accuracy
    accuracy_metrics = {
        'plankart': df_summary['plankart_jaccard'].str.replace('N/A', '0').astype(float).mean(),
        'bestemmelser': df_summary['bestemm_jaccard'].str.replace('N/A', '0').astype(float).mean(),
        'sosi': df_summary['sosi_jaccard'].str.replace('N/A', '0').astype(float).mean()
    }
    
    with open(report_path, 'w') as f:
        f.write("\\section{Summary Report}\n\n")
        
        # Executive Summary
        f.write("\\subsection{Executive Summary}\n")
        f.write("\\begin{itemize}\n")
        f.write(f"\\item Average Processing Time Improvement: {speed_improvement:.2f}x faster\n")
        f.write(f"\\item Average Field Extraction Accuracy: {sum(accuracy_metrics.values())/3:.2%}\n")
        f.write("\\end{itemize}\n\n")
        
        # Performance Metrics
        f.write("\\subsection{Performance Metrics}\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|cc}\n")
        f.write("\\hline\n")
        f.write("Metric & Legacy & Docling \\\\\n")
        f.write("\\hline\n")
        f.write(f"Average Processing Time (s) & {avg_legacy_time:.2f} & {avg_docling_time:.2f} \\\\\n")
        f.write(f"Speed Improvement & {speed_improvement:.2f}x & - \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Performance Comparison}\n")
        f.write("\\end{table}\n\n")
        
        # Accuracy Metrics
        f.write("\\subsection{Accuracy Metrics}\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|c}\n")
        f.write("\\hline\n")
        f.write("Document Type & Jaccard Similarity \\\\\n")
        f.write("\\hline\n")
        for doc_type, accuracy in accuracy_metrics.items():
            f.write(f"{doc_type.title()} & {accuracy:.2%} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Field Extraction Accuracy}\n")
        f.write("\\end{table}\n\n")
        
        # Detailed Results
        f.write("\\subsection{Detailed Results}\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|cc|cc|ccc}\n")
        f.write("\\hline\n")
        f.write("Case & Legacy Time & Docling Time & Legacy Fields & Docling Fields & Plankart Sim & Bestemm Sim & SOSI Sim \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df_summary.iterrows():
            f.write(f"{row.name} & {row['legacy_time']:.2f} & {row['docling_time']:.2f} & ")
            f.write(f"{row['legacy_plankart_fields'] + row['legacy_bestemm_fields'] + row['legacy_sosi_fields']} & ")
            f.write(f"{row['docling_plankart_fields'] + row['docling_bestemm_fields'] + row['docling_sosi_fields']} & ")
            f.write(f"{row['plankart_jaccard']} & {row['bestemm_jaccard']} & {row['sosi_jaccard']} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Detailed Results by Case}\n")
        f.write("\\end{table}\n\n")
        
        # Conclusions
        f.write("\\subsection{Conclusions}\n")
        f.write("\\begin{itemize}\n")
        f.write(f"\\item The Docling method is {speed_improvement:.2f}x faster than the legacy method\n")
        f.write(f"\\item Field extraction accuracy is highest for SOSI ({accuracy_metrics['sosi']:.2%}) and lowest for Bestemmelser ({accuracy_metrics['bestemmelser']:.2%})\n")
        f.write("\\end{itemize}\n")
    
    return report_path

def analyze_field_extraction_errors(legacy_res: Dict, docling_res: Dict, case_id: str) -> Dict:
    """Analyze field extraction errors and differences between methods."""
    analysis = {
        "case_id": case_id,
        "false_positives": {},
        "false_negatives": {},
        "field_type_analysis": {},
        "error_patterns": []
    }
    
    # Analyze each document type
    for doc_type in ["plankart", "bestemmelser", "sosi"]:
        legacy_set = legacy_res.get(doc_type, set())
        docling_set = docling_res.get(doc_type, set())
        
        # False positives (in legacy but not in docling)
        false_positives = legacy_set - docling_set
        # False negatives (in docling but not in legacy)
        false_negatives = docling_set - legacy_set
        
        analysis["false_positives"][doc_type] = sorted(list(false_positives))
        analysis["false_negatives"][doc_type] = sorted(list(false_negatives))
        
        # Analyze field type patterns
        field_types = {}
        for field in legacy_set | docling_set:
            # Extract field type (e.g., H1, H2, etc.)
            match = re.match(r'([A-ZÆØÅ]+)(\d+)', field)
            if match:
                field_type = match.group(1)
                if field_type not in field_types:
                    field_types[field_type] = {"legacy": 0, "docling": 0}
                if field in legacy_set:
                    field_types[field_type]["legacy"] += 1
                if field in docling_set:
                    field_types[field_type]["docling"] += 1
        
        analysis["field_type_analysis"][doc_type] = field_types
        
        # Identify error patterns
        patterns = []
        for fp in false_positives:
            # Check for similar fields in docling
            similar_fields = [f for f in docling_set if f.startswith(fp[:2])]
            if similar_fields:
                patterns.append({
                    "type": "similar_field",
                    "legacy_field": fp,
                    "similar_docling_fields": similar_fields
                })
        
        analysis["error_patterns"].extend(patterns)
    
    return analysis

def generate_confusion_matrices(legacy_res: Dict, docling_res: Dict, case_id: str) -> Dict:
    """Generate confusion matrices for field extraction comparison."""
    matrices = {
        "case_id": case_id,
        "plankart": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "bestemmelser": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "sosi": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    }
    
    for doc_type in ["plankart", "bestemmelser", "sosi"]:
        legacy_set = legacy_res.get(doc_type, set())
        docling_set = docling_res.get(doc_type, set())
        
        # True Positives (fields found by both methods)
        tp = len(legacy_set & docling_set)
        # False Positives (fields found by legacy but not docling)
        fp = len(legacy_set - docling_set)
        # False Negatives (fields found by docling but not legacy)
        fn = len(docling_set - legacy_set)
        
        matrices[doc_type].update({
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        })
    
    return matrices

def create_field_comparison_table(legacy_res: Dict, docling_res: Dict, case_id: str) -> pd.DataFrame:
    """Create a detailed comparison table of extracted fields."""
    rows = []
    
    for doc_type in ["plankart", "bestemmelser", "sosi"]:
        legacy_set = legacy_res.get(doc_type, set())
        docling_set = docling_res.get(doc_type, set())
        
        # Get all unique fields
        all_fields = sorted(legacy_set | docling_set)
        
        for field in all_fields:
            rows.append({
                "Case": case_id,
                "Document Type": doc_type,
                "Field": field,
                "Legacy Extracted": field in legacy_set,
                "Docling Extracted": field in docling_set,
                "Status": "Match" if (field in legacy_set) == (field in docling_set) else "Mismatch"
            })
    
    return pd.DataFrame(rows)

def analyze_field_consistency(legacy_res: Dict, docling_res: Dict, case_id: str) -> Dict:
    """Analyze consistency of field extraction across document types."""
    analysis = {
        "case_id": case_id,
        "legacy_consistency": {
            "consistent_fields": set(),
            "inconsistent_fields": set(),
            "consistency_score": 0.0
        },
        "docling_consistency": {
            "consistent_fields": set(),
            "inconsistent_fields": set(),
            "consistency_score": 0.0
        }
    }
    
    # Analyze legacy consistency
    legacy_fields = {
        "plankart": legacy_res.get("plankart", set()),
        "bestemmelser": legacy_res.get("bestemmelser", set()),
        "sosi": legacy_res.get("sosi", set())
    }
    
    # Analyze docling consistency
    docling_fields = {
        "plankart": docling_res.get("plankart", set()),
        "bestemmelser": docling_res.get("bestemmelser", set()),
        "sosi": docling_res.get("sosi", set())
    }
    
    # Calculate consistency for each method
    for method, fields in [("legacy", legacy_fields), ("docling", docling_fields)]:
        all_fields = set().union(*fields.values())
        consistent_fields = set()
        inconsistent_fields = set()
        
        for field in all_fields:
            # Check if field appears in all document types
            if all(field in doc_set for doc_set in fields.values() if doc_set):
                consistent_fields.add(field)
            else:
                inconsistent_fields.add(field)
        
        analysis[f"{method}_consistency"].update({
            "consistent_fields": sorted(list(consistent_fields)),
            "inconsistent_fields": sorted(list(inconsistent_fields)),
            "consistency_score": len(consistent_fields) / len(all_fields) if all_fields else 0.0
        })
    
    return analysis

def analyze_results(case_results: List[Dict]) -> Dict:
    """Analyze the comparison results."""
    analysis = {
        'cases': {},
        'summary': {
            'avg_legacy_time': 0,
            'avg_docling_time': 0,
            'avg_plankart_sim': 0,
            'avg_bestemm_sim': 0,
            'avg_sosi_sim': 0
        }
    }
    
    total_cases = len(case_results)
    total_legacy_time = 0
    total_docling_time = 0
    total_plankart_sim = 0
    total_bestemm_sim = 0
    total_sosi_sim = 0
    valid_sims = {'plankart': 0, 'bestemm': 0, 'sosi': 0}
    
    for result in case_results:
        case_name = result['case']
        analysis['cases'][case_name] = {
            'legacy': {
                'plankart_fields': result['legacy_plankart_fields'],
                'bestemm_fields': result['legacy_bestemm_fields'],
                'sosi_fields': result['legacy_sosi_fields'],
                'time': result['legacy_time']
            },
            'docling': {
                'plankart_fields': result['docling_plankart_fields'],
                'bestemm_fields': result['docling_bestemm_fields'],
                'sosi_fields': result['docling_sosi_fields'],
                'time': result['docling_time']
            },
            'similarities': {
                'plankart': result['plankart_jaccard'],
                'bestemm': result['bestemm_jaccard'],
                'sosi': result['sosi_jaccard']
            }
        }
        
        total_legacy_time += result['legacy_time']
        total_docling_time += result['docling_time']
        
        # Handle similarity calculations
        if result['plankart_jaccard'] != 'N/A':
            total_plankart_sim += float(result['plankart_jaccard'])
            valid_sims['plankart'] += 1
            
        if result['bestemm_jaccard'] != 'N/A':
            total_bestemm_sim += float(result['bestemm_jaccard'])
            valid_sims['bestemm'] += 1
            
        if result['sosi_jaccard'] != 'N/A':
            total_sosi_sim += float(result['sosi_jaccard'])
            valid_sims['sosi'] += 1
    
    # Calculate averages
    analysis['summary']['avg_legacy_time'] = total_legacy_time / total_cases
    analysis['summary']['avg_docling_time'] = total_docling_time / total_cases
    analysis['summary']['avg_plankart_sim'] = total_plankart_sim / valid_sims['plankart'] if valid_sims['plankart'] > 0 else 0
    analysis['summary']['avg_bestemm_sim'] = total_bestemm_sim / valid_sims['bestemm'] if valid_sims['bestemm'] > 0 else 0
    analysis['summary']['avg_sosi_sim'] = total_sosi_sim / valid_sims['sosi'] if valid_sims['sosi'] > 0 else 0
    
    return analysis

def generate_error_analysis_plots(error_analysis: Dict[str, Any], output_dir: Path) -> None:
    """Generate plots for error analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a default style instead of seaborn
    plt.style.use('default')
    
    # Handle both old and new dictionary structures
    cases_data = error_analysis.get('cases', error_analysis)
    if not cases_data:
        logger.warning("No cases data found in error analysis")
        return
        
    cases = list(cases_data.keys())
    x = np.arange(len(cases))
    width = 0.35
    
    # Plot field counts comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting - handle both dictionary structures
    legacy_plankart = []
    docling_plankart = []
    for case in cases:
        case_data = cases_data[case]
        if isinstance(case_data, dict) and 'legacy' in case_data:
            # New structure
            legacy_plankart.append(case_data['legacy']['plankart_fields'])
            docling_plankart.append(case_data['docling']['plankart_fields'])
        else:
            # Old structure
            legacy_plankart.append(case_data.get('legacy_plankart_fields', 0))
            docling_plankart.append(case_data.get('docling_plankart_fields', 0))
    
    plt.bar(x - width/2, legacy_plankart, width, label='Legacy')
    plt.bar(x + width/2, docling_plankart, width, label='Docling')
    
    plt.xlabel('Cases')
    plt.ylabel('Plankart Field Count')
    plt.title('Plankart Field Count Comparison')
    plt.xticks(x, cases)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'plankart_field_counts.png')
    plt.close()
    
    # Plot processing times comparison
    plt.figure(figsize=(10, 6))
    
    legacy_times = []
    docling_times = []
    for case in cases:
        case_data = cases_data[case]
        if isinstance(case_data, dict) and 'legacy' in case_data:
            # New structure
            legacy_times.append(case_data['legacy']['time'])
            docling_times.append(case_data['docling']['time'])
        else:
            # Old structure
            legacy_times.append(case_data.get('legacy_time', 0))
            docling_times.append(case_data.get('docling_time', 0))
    
    plt.bar(x - width/2, legacy_times, width, label='Legacy')
    plt.bar(x + width/2, docling_times, width, label='Docling')
    
    plt.xlabel('Cases')
    plt.ylabel('Processing Time (s)')
    plt.title('Processing Time Comparison')
    plt.xticks(x, cases)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'processing_times.png')
    plt.close()
    
    # Plot Jaccard similarities
    plt.figure(figsize=(10, 6))
    
    # Convert 'N/A' to NaN for plotting
    def safe_float(x):
        try:
            return float(x) if x != 'N/A' else np.nan
        except (ValueError, TypeError):
            return np.nan
    
    similarities = {
        'Plankart': [],
        'Bestemmelser': [],
        'SOSI': []
    }
    
    for case in cases:
        case_data = cases_data[case]
        if isinstance(case_data, dict) and 'similarities' in case_data:
            # New structure
            similarities['Plankart'].append(safe_float(case_data['similarities']['plankart']))
            similarities['Bestemmelser'].append(safe_float(case_data['similarities']['bestemm']))
            similarities['SOSI'].append(safe_float(case_data['similarities']['sosi']))
        else:
            # Old structure
            similarities['Plankart'].append(safe_float(case_data.get('plankart_jaccard')))
            similarities['Bestemmelser'].append(safe_float(case_data.get('bestemm_jaccard')))
            similarities['SOSI'].append(safe_float(case_data.get('sosi_jaccard')))
    
    x = np.arange(len(cases))
    width = 0.25
    multiplier = 0
    
    for attribute, similarity in similarities.items():
        offset = width * multiplier
        plt.bar(x + offset, similarity, width, label=attribute)
        multiplier += 1
    
    plt.xlabel('Cases')
    plt.ylabel('Jaccard Similarity')
    plt.title('Similarity Comparison')
    plt.xticks(x + width, cases)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'similarities.png')
    plt.close()

def analyze_field_context(doc: DoclingDocument, field: str) -> Dict:
    """Analyze the context in which a field appears in the document."""
    context = {
        "surrounding_text": [],
        "section_context": None,
        "proximity_fields": set(),
        "field_type": None
    }
    
    # Determine field type
    if re.match(r'^H\d+', field):
        context["field_type"] = "heading"
    elif re.match(r'^[A-ZÆØÅ]+\d+', field):
        context["field_type"] = "zone"
    elif re.match(r'^#\d+', field):
        context["field_type"] = "reference"
    
    # Find the field in the document
    for item in doc.texts:
        if hasattr(item, 'text') and field in item.text:
            # Get surrounding text
            if hasattr(item, 'parent'):
                parent = item.parent.resolve(doc)
                if parent and hasattr(parent, 'children'):
                    for child_ref in parent.children:
                        child = child_ref.resolve(doc)
                        if hasattr(child, 'text'):
                            context["surrounding_text"].append(child.text)
            
            # Get section context
            current = item
            while hasattr(current, 'parent') and current.parent:
                parent = current.parent.resolve(doc)
                if isinstance(parent, GroupItem) and parent.label == "SECTION":
                    context["section_context"] = parent.name
                    break
                current = parent
            
            # Find nearby fields
            if hasattr(item, 'prov'):
                # Handle case where prov is a list
                prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                for prov in prov_list:
                    if hasattr(prov, 'bbox'):
                        item_bbox = prov.bbox
                        for other_item in doc.texts:
                            if other_item != item and hasattr(other_item, 'prov'):
                                other_prov_list = other_item.prov if isinstance(other_item.prov, list) else [other_item.prov]
                                for other_prov in other_prov_list:
                                    if hasattr(other_prov, 'bbox'):
                                        other_bbox = other_prov.bbox
                                        # Check if items are close (within 100 units)
                                        if (abs(item_bbox.l - other_bbox.l) < 100 and 
                                            abs(item_bbox.t - other_bbox.t) < 100):
                                            if hasattr(other_item, 'text'):
                                                context["proximity_fields"].add(other_item.text)
    
    return context

def analyze_semantic_relationships(legacy_res: Dict, docling_res: Dict, doc: DoclingDocument) -> Dict:
    """Analyze semantic relationships between extracted fields."""
    relationships = {
        "hierarchical": {},  # Parent-child relationships
        "spatial": {},      # Spatial relationships
        "contextual": {},   # Context-based relationships
        "field_patterns": {} # Common patterns in field extraction
    }
    
    # Analyze hierarchical relationships
    for field in legacy_res.get("plankart", set()) | docling_res.get("plankart", set()):
        context = analyze_field_context(doc, field)
        if context["field_type"] == "heading":
            relationships["hierarchical"][field] = {
                "children": set(),
                "parent": None
            }
            # Find potential children (fields that appear after this heading)
            for item in doc.texts:
                if hasattr(item, 'text') and field in item.text:
                    # Look for fields that appear after this heading
                    for other_item in doc.texts:
                        if (hasattr(other_item, 'prov') and 
                            hasattr(item, 'prov')):
                            # Handle case where prov is a list
                            item_prov_list = item.prov if isinstance(item.prov, list) else [item.prov]
                            other_prov_list = other_item.prov if isinstance(other_item.prov, list) else [other_item.prov]
                            for item_prov in item_prov_list:
                                for other_prov in other_prov_list:
                                    if (hasattr(item_prov, 'bbox') and 
                                        hasattr(other_prov, 'bbox') and
                                        other_prov.bbox.t > item_prov.bbox.t):
                                        if hasattr(other_item, 'text'):
                                            relationships["hierarchical"][field]["children"].add(other_item.text)
    
    # Analyze spatial relationships
    for field in legacy_res.get("plankart", set()) | docling_res.get("plankart", set()):
        context = analyze_field_context(doc, field)
        relationships["spatial"][field] = {
            "nearby_fields": context["proximity_fields"],
            "section": context["section_context"]
        }
    
    # Analyze contextual relationships
    for field in legacy_res.get("bestemmelser", set()) | docling_res.get("bestemmelser", set()):
        context = analyze_field_context(doc, field)
        relationships["contextual"][field] = {
            "surrounding_text": context["surrounding_text"],
            "section_context": context["section_context"]
        }
    
    # Analyze field patterns
    for doc_type in ["plankart", "bestemmelser", "sosi"]:
        legacy_fields = legacy_res.get(doc_type, set())
        docling_fields = docling_res.get(doc_type, set())
        
        # Find common patterns in field names
        patterns = {}
        for field in legacy_fields | docling_fields:
            # Extract pattern (e.g., "H1", "H2", etc.)
            match = re.match(r'([A-ZÆØÅ]+)(\d+)', field)
            if match:
                pattern = match.group(1)
                if pattern not in patterns:
                    patterns[pattern] = {
                        "legacy_count": 0,
                        "docling_count": 0,
                        "examples": set()
                    }
                patterns[pattern]["examples"].add(field)
                if field in legacy_fields:
                    patterns[pattern]["legacy_count"] += 1
                if field in docling_fields:
                    patterns[pattern]["docling_count"] += 1
        
        relationships["field_patterns"][doc_type] = patterns
    
    return relationships

def generate_semantic_analysis_report(relationships: Dict, output_dir: Path):
    """Generate a report of semantic relationships analysis."""
    report_path = output_dir / "semantic_analysis.tex"
    
    with open(report_path, 'w') as f:
        f.write("\\section{Semantic Analysis}\n\n")
        
        # Hierarchical Relationships
        f.write("\\subsection{Hierarchical Relationships}\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|l}\n")
        f.write("\\hline\n")
        f.write("Heading & Children \\\\\n")
        f.write("\\hline\n")
        for heading, data in relationships["hierarchical"].items():
            children = ", ".join(sorted(data["children"]))
            f.write(f"{heading} & {children} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Hierarchical Relationships Between Fields}\n")
        f.write("\\end{table}\n\n")
        
        # Field Patterns
        f.write("\\subsection{Field Patterns}\n\n")
        for doc_type, patterns in relationships["field_patterns"].items():
            f.write(f"\\subsubsection{{{doc_type.title()}}}\n\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{l|cc|l}\n")
            f.write("\\hline\n")
            f.write("Pattern & Legacy & Docling & Examples \\\\\n")
            f.write("\\hline\n")
            for pattern, data in patterns.items():
                examples = ", ".join(sorted(data["examples"])[:3])  # Show first 3 examples
                f.write(f"{pattern} & {data['legacy_count']} & {data['docling_count']} & {examples} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Field Patterns in {doc_type.title()}}}\n")
            f.write("\\end{table}\n\n")
        
        # Contextual Relationships
        f.write("\\subsection{Contextual Relationships}\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|l}\n")
        f.write("\\hline\n")
        f.write("Field & Context \\\\\n")
        f.write("\\hline\n")
        for field, data in relationships["contextual"].items():
            context = data["section_context"] or "No section context"
            f.write(f"{field} & {context} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Contextual Relationships}\n")
        f.write("\\end{table}\n\n")

def perform_statistical_analysis(legacy_results: List[Dict], docling_results: List[Dict]) -> Dict:
    """Perform statistical analysis on the results."""
    analysis = {
        "processing_time": {
            "legacy_mean": np.mean([r["time"] for r in legacy_results]),
            "legacy_std": np.std([r["time"] for r in legacy_results]),
            "docling_mean": np.mean([r["time"] for r in docling_results]),
            "docling_std": np.std([r["time"] for r in docling_results]),
            "t_test": None,  # Will be filled with scipy.stats.ttest_ind result
            "effect_size": None  # Will be filled with Cohen's d
        },
        "field_counts": {
            "legacy_mean": np.mean([len(r["plankart"] | r["bestemmelser"] | r["sosi"]) for r in legacy_results]),
            "legacy_std": np.std([len(r["plankart"] | r["bestemmelser"] | r["sosi"]) for r in legacy_results]),
            "docling_mean": np.mean([len(r["plankart"] | r["bestemmelser"] | r["sosi"]) for r in docling_results]),
            "docling_std": np.std([len(r["plankart"] | r["bestemmelser"] | r["sosi"]) for r in docling_results]),
            "t_test": None,
            "effect_size": None
        }
    }
    
    # Perform t-tests
    from scipy import stats
    
    # Processing time t-test
    legacy_times = [r["time"] for r in legacy_results]
    docling_times = [r["time"] for r in docling_results]
    analysis["processing_time"]["t_test"] = stats.ttest_ind(legacy_times, docling_times)
    
    # Field count t-test
    legacy_counts = [len(r["plankart"] | r["bestemmelser"] | r["sosi"]) for r in legacy_results]
    docling_counts = [len(r["plankart"] | r["bestemmelser"] | r["sosi"]) for r in docling_results]
    analysis["field_counts"]["t_test"] = stats.ttest_ind(legacy_counts, docling_counts)
    
    # Calculate effect sizes (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se
    
    analysis["processing_time"]["effect_size"] = cohens_d(legacy_times, docling_times)
    analysis["field_counts"]["effect_size"] = cohens_d(legacy_counts, docling_counts)
    
    return analysis

def generate_statistical_report(statistical_analysis: Dict, output_dir: Path):
    """Generate a LaTeX report of statistical analysis results."""
    report_path = output_dir / "statistical_analysis.tex"
    
    with open(report_path, 'w') as f:
        f.write("\\section{Statistical Analysis}\n\n")
        
        # Processing Time Analysis
        f.write("\\subsection{Processing Time Analysis}\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|cc}\n")
        f.write("\\hline\n")
        f.write("Metric & Legacy & Docling \\\\\n")
        f.write("\\hline\n")
        f.write(f"Mean Time (s) & {statistical_analysis['processing_time']['legacy_mean']:.2f} & {statistical_analysis['processing_time']['docling_mean']:.2f} \\\\\n")
        f.write(f"Std Dev (s) & {statistical_analysis['processing_time']['legacy_std']:.2f} & {statistical_analysis['processing_time']['docling_std']:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Processing Time Statistics}\n")
        f.write("\\end{table}\n\n")
        
        t_test = statistical_analysis['processing_time']['t_test']
        f.write(f"T-test results: t = {t_test.statistic:.3f}, p = {t_test.pvalue:.3f}\n\n")
        f.write(f"Effect size (Cohen's d): {statistical_analysis['processing_time']['effect_size']:.3f}\n\n")
        
        # Field Count Analysis
        f.write("\\subsection{Field Count Analysis}\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l|cc}\n")
        f.write("\\hline\n")
        f.write("Metric & Legacy & Docling \\\\\n")
        f.write("\\hline\n")
        f.write(f"Mean Fields & {statistical_analysis['field_counts']['legacy_mean']:.2f} & {statistical_analysis['field_counts']['docling_mean']:.2f} \\\\\n")
        f.write(f"Std Dev & {statistical_analysis['field_counts']['legacy_std']:.2f} & {statistical_analysis['field_counts']['docling_std']:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Field Count Statistics}\n")
        f.write("\\end{table}\n\n")
        
        t_test = statistical_analysis['field_counts']['t_test']
        f.write(f"T-test results: t = {t_test.statistic:.3f}, p = {t_test.pvalue:.3f}\n\n")
        f.write(f"Effect size (Cohen's d): {statistical_analysis['field_counts']['effect_size']:.3f}\n\n")

async def process_plankart_legacy(plankart_path: Path) -> Set[str]:
    """Process plankart using legacy method."""
    if not FieldExtractor or not model_name:
        logger.warning("Legacy FieldExtractor not available")
        return set()
    
    try:
        extractor_llm = FieldExtractor(model_name)
        with open(plankart_path, 'rb') as f:
            content = f.read()
        extracted_set = await extractor_llm.extract_fields(content)
        return {normalize_field_comparison(f) for f in extracted_set if normalize_field_comparison(f)}
    except Exception as e:
        logger.error(f"Legacy Plankart processing failed: {e}")
        return set()

async def process_bestemmelser_legacy(bestemmelser_path: Path) -> Set[str]:
    """Process bestemmelser using legacy method."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(bestemmelser_path, 'rb') as f:
                files = {'file': (bestemmelser_path.name, f, 'application/pdf')}
                response = await client.post(NER_SERVICE_URL, files=files)
                response.raise_for_status()
                ner_fields = response.json().get('fields', [])
                return {normalize_field_comparison(f) for f in ner_fields if normalize_field_comparison(f)}
    except Exception as e:
        logger.error(f"Legacy Bestemmelser processing failed: {e}")
        return set()

def process_sosi_legacy(sosi_path: Optional[Path]) -> Set[str]:
    """Process SOSI using legacy method."""
    if not LegacySosiParser or not sosi_path or not sosi_path.exists():
        return set()
    
    try:
        sosi_codes_map = load_sosi_purpose_codes(SOSI_CODES_CSV)
        legacy_sosi_parser = LegacySosiParser(purpose_map=sosi_codes_map)
        parsed_sosi = legacy_sosi_parser.parse_file(sosi_path)
        sosi_zones = parsed_sosi.get("fields", {}).get("zone_identifiers", [])
        sosi_hensyn = parsed_sosi.get("fields", {}).get("hensynssoner", [])
        return {normalize_field_comparison(f) for f in sosi_zones if normalize_field_comparison(f)} | \
               {normalize_field_comparison(h) for h in sosi_hensyn if normalize_field_comparison(h)}
    except Exception as e:
        logger.error(f"Legacy SOSI processing failed: {e}")
        return set()

def process_plankart_docling(plankart_path: Path, save_visualization: bool = False) -> Set[str]:
    """Process plankart using Docling method."""
    try:
        doc = parse_plankart(plankart_path, plankart_path.stem)
        if doc:
            if save_visualization:
                doc.save_as_json(DOCLING_OUTPUT_DIR / f"{doc.name}_structure.json", image_mode='placeholder')
            return extract_zones_from_docling_plankart(doc)
        return set()
    except Exception as e:
        logger.error(f"Docling Plankart processing failed: {e}")
        return set()

def process_bestemmelser_docling(bestemmelser_path: Path, save_visualization: bool = False) -> Set[str]:
    """Process bestemmelser using Docling method."""
    try:
        doc = parse_planbestemmelser(bestemmelser_path, bestemmelser_path.stem)
        if doc:
            if save_visualization:
                doc.save_as_json(DOCLING_OUTPUT_DIR / f"{doc.name}_structure.json")
            return extract_zones_from_docling_bestemmelser(doc)
        return set()
    except Exception as e:
        logger.error(f"Docling Bestemmelser processing failed: {e}")
        return set()

def process_sosi_docling(sosi_path: Optional[Path], sosi_codes_map: Dict[str, str], save_visualization: bool = False) -> Set[str]:
    """Process SOSI using Docling method."""
    if not sosi_path or not sosi_path.exists():
        return set()
    
    try:
        doc = parse_sosi_to_docling(sosi_path, sosi_codes_map, sosi_path.stem)
        if doc:
            if save_visualization:
                doc.save_as_json(DOCLING_OUTPUT_DIR / f"{doc.name}_structure.json")
            return extract_zones_from_docling_sosi(doc)
        return set()
    except Exception as e:
        logger.error(f"Docling SOSI processing failed: {e}")
        return set()

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> str:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return "N/A"
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return f"{intersection/union:.2f}" if union > 0 else "0.00"

def save_comparison_results(comparison: Dict, output_path: Path):
    """Save comparison results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

def save_comparison_summary(results: List[Dict], output_path: Path):
    """Save comparison summary to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def save_comparison_csv(results: List[Dict], output_path: Path):
    """Save comparison results to CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

def save_docling_consistency_table(docling_results, cases, output_path):
    """Save a CSV with intersections and differences of docling field sets for each case."""
    rows = []
    for idx, (case_name, case_details) in enumerate(cases.items()):
        P = docling_results[idx]["plankart"]
        B = docling_results[idx]["bestemmelser"]
        S = docling_results[idx]["sosi"]
        matching_all = len(P & B & S)
        only_plankart = len(P - B - S)
        only_bestemmelser = len(B - P - S)
        only_sosi = len(S - P - B)
        rows.append({
            "case": case_name,
            "matching_all": matching_all,
            "only_plankart": only_plankart,
            "only_bestemmelser": only_bestemmelser,
            "only_sosi": only_sosi
        })
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def save_docling_field_sets_json(docling_results, cases, output_dir):
    """Save, for each case, the actual sets and set operations for docling fields as JSON."""
    import json
    for idx, (case_name, case_details) in enumerate(cases.items()):
        P = docling_results[idx]["plankart"]
        B = docling_results[idx]["bestemmelser"]
        S = docling_results[idx]["sosi"]
        matching_all = P & B & S
        only_plankart = P - B - S
        only_bestemmelser = B - P - S
        only_sosi = S - P - B
        data = {
            "case": case_name,
            "plankart_fields": sorted(P),
            "bestemmelser_fields": sorted(B),
            "sosi_fields": sorted(S),
            "matching_all": sorted(matching_all),
            "only_plankart": sorted(only_plankart),
            "only_bestemmelser": sorted(only_bestemmelser),
            "only_sosi": sorted(only_sosi),
            "counts": {
                "matching_all": len(matching_all),
                "only_plankart": len(only_plankart),
                "only_bestemmelser": len(only_bestemmelser),
                "only_sosi": len(only_sosi)
            }
        }
        with open(output_dir / f"docling_field_sets_{case_name}.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

# --- Main Execution ---
async def main():
    """Main function to run the comparison."""
    # Load SOSI codes
    sosi_codes = load_sosi_purpose_codes(SOSI_CODES_CSV)
    if not sosi_codes:
        logger.error("Failed to load SOSI purpose codes. Cannot proceed.")
        return
    
    # Process each case
    case_results = []
    legacy_results = []
    docling_results = []
    
    for case_name, case_details in CASES.items():
        logger.info(f"\n===== Processing Case: {case_name} =====")
        
        # Run legacy method
        logger.info("Running Legacy Method Simulation...")
        legacy_start = time.time()
        legacy_plankart = await process_plankart_legacy(case_details['plankart'])
        legacy_bestemm = await process_bestemmelser_legacy(case_details['bestemmelser'])
        legacy_sosi = process_sosi_legacy(case_details['sosi'])
        legacy_time = time.time() - legacy_start
        
        legacy_result = {
            "plankart": legacy_plankart,
            "bestemmelser": legacy_bestemm,
            "sosi": legacy_sosi,
            "time": legacy_time
        }
        legacy_results.append(legacy_result)
        
        # Run Docling method
        logger.info("Running Docling Method...")
        docling_start = time.time()
        docling_plankart = process_plankart_docling(case_details['plankart'])
        docling_bestemm = process_bestemmelser_docling(case_details['bestemmelser'])
        docling_sosi = process_sosi_docling(case_details['sosi'], sosi_codes)
        docling_time = time.time() - docling_start
        
        docling_result = {
            "plankart": docling_plankart,
            "bestemmelser": docling_bestemm,
            "sosi": docling_sosi,
            "time": docling_time
        }
        docling_results.append(docling_result)
        
        # Compare results
        logger.info(f"Comparing results for case: {case_name}")
        comparison = {
            'case': case_name,
            'legacy_plankart_fields': len(legacy_plankart),
            'legacy_bestemm_fields': len(legacy_bestemm),
            'legacy_sosi_fields': len(legacy_sosi),
            'docling_plankart_fields': len(docling_plankart),
            'docling_bestemm_fields': len(docling_bestemm),
            'docling_sosi_fields': len(docling_sosi),
            'legacy_time': legacy_time,
            'docling_time': docling_time,
            'plankart_jaccard': calculate_jaccard_similarity(legacy_plankart, docling_plankart),
            'bestemm_jaccard': calculate_jaccard_similarity(legacy_bestemm, docling_bestemm),
            'sosi_jaccard': calculate_jaccard_similarity(legacy_sosi, docling_sosi)
        }
        case_results.append(comparison)
        
        # Save individual case results
        save_comparison_results(comparison, OUTPUT_DIRS["comparison"] / f"{case_name}_comparison.json")
        
        # Generate visualizations for the case
        process_plankart_docling(case_details['plankart'], save_visualization=True)
        process_bestemmelser_docling(case_details['bestemmelser'], save_visualization=True)
    
    # Create DataFrame for analysis
    df_summary = pd.DataFrame(case_results)
    df_summary.set_index('case', inplace=True)
    
    # Perform all analyses
    logger.info("Performing comprehensive analysis...")
    
    # 1. Basic comparison plots
    create_comparison_plots(df_summary, OUTPUT_DIRS["plots"])
    
    # 2. Error analysis
    error_analysis = analyze_results(case_results)
    generate_error_analysis_plots(error_analysis, OUTPUT_DIRS["error_analysis"])
    
    # 3. Statistical analysis
    statistical_analysis = perform_statistical_analysis(legacy_results, docling_results)
    generate_statistical_report(statistical_analysis, OUTPUT_DIRS["statistical"])
    
    # 4. Field analysis
    for idx, (case_name, case_details) in enumerate(CASES.items()):
        field_analysis = analyze_field_extraction_errors(
            legacy_results[idx], 
            docling_results[idx], 
            case_name
        )
        confusion_matrices = generate_confusion_matrices(
            legacy_results[idx], 
            docling_results[idx], 
            case_name
        )
        field_comparison = create_field_comparison_table(
            legacy_results[idx], 
            docling_results[idx], 
            case_name
        )
        field_consistency = analyze_field_consistency(
            legacy_results[idx], 
            docling_results[idx], 
            case_name
        )
        # Save field analysis results
        save_comparison_results(field_analysis, OUTPUT_DIRS["field_analysis"] / f"{case_name}_field_analysis.json")
        save_comparison_results(confusion_matrices, OUTPUT_DIRS["field_analysis"] / f"{case_name}_confusion_matrices.json")
        field_comparison.to_csv(OUTPUT_DIRS["field_analysis"] / f"{case_name}_field_comparison.csv")
        save_comparison_results(field_consistency, OUTPUT_DIRS["field_analysis"] / f"{case_name}_field_consistency.json")
    
    # 5. Semantic analysis
    for idx, (case_name, case_details) in enumerate(CASES.items()):
        doc = parse_plankart(case_details['plankart'], case_details['plankart'].stem)
        semantic_relationships = analyze_semantic_relationships(
            legacy_results[idx], 
            docling_results[idx], 
            doc
        )
        generate_semantic_analysis_report(semantic_relationships, OUTPUT_DIRS["semantic"])
    
    # 6. Generate summary report
    generate_summary_report(df_summary, OUTPUT_DIRS["summary"])
    
    # 7. Save docling consistency table for LaTeX/Excel
    save_docling_consistency_table(docling_results, CASES, OUTPUT_DIRS["summary"] / "docling_consistency_table.csv")
    
    # 8. Save docling field sets and set operations as JSON for each case
    save_docling_field_sets_json(docling_results, CASES, OUTPUT_DIRS["summary"])
    
    # Save summary results
    save_comparison_summary(case_results, OUTPUT_DIRS["comparison"] / "comparison_summary.json")
    save_comparison_csv(case_results, OUTPUT_DIRS["comparison"] / "comparison_summary.csv")
    
    logger.info("Analysis complete. Results saved in organized directory structure.")


if __name__ == "__main__":
    import asyncio
    # Ensure the event loop policy is set for Windows if necessary
    if sys.platform == "win32":
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())