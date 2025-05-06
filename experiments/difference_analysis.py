# experiments/analyze_differences.py
import json
import difflib
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any

from docling_core.types.doc import DoclingDocument, TextItem, GroupItem, RefItem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
RESULTS_DIR = Path(__file__).parent / "results"
DOCLING_PARSED_DIR = RESULTS_DIR / "docling_parsed"
DIFF_OUTPUT_DIR = RESULTS_DIR / "difference_analysis"
DIFF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helper Functions ---
def load_docling_doc(json_path: Path) -> DoclingDocument:
    """Loads a DoclingDocument from its JSON representation."""
    logger.info(f"Loading Docling document from: {json_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Docling JSON file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Use DoclingDocument's built-in loading if available, otherwise basic Pydantic parse
    if hasattr(DoclingDocument, 'load_from_json'):
         return DoclingDocument.load_from_json(json_path)
    else:
         # Fallback: Basic Pydantic parsing (might miss some object reconstructions)
         logger.warning("Using basic Pydantic parsing for DoclingDocument. Full functionality might be limited.")
         # This requires DoclingDocument to be a Pydantic model
         # return DoclingDocument.parse_obj(data) # Deprecated in Pydantic V2
         return DoclingDocument.model_validate(data) # Pydantic V2


def get_item_by_ref(doc: DoclingDocument, ref_str: str) -> Any:
    """Resolves a JSON pointer string to an item in the document."""
    try:
        return RefItem(cref=ref_str).resolve(doc)
    except Exception as e:
        logger.warning(f"Could not resolve ref '{ref_str}': {e}")
        return None

def compare_docling_structures(doc1: DoclingDocument, doc2: DoclingDocument) -> Dict:
    """Performs a structured comparison between two DoclingDocuments."""
    diff_results = {
        "added_items": [],
        "removed_items": [],
        "modified_items": [],
        "moved_items": [] # Items with same content but different parent/siblings
    }

    # Create maps of items by their reference string for easier lookup
    items1 = {item.self_ref: item for item in doc1.texts + doc1.groups if hasattr(item, 'self_ref')}
    items2 = {item.self_ref: item for item in doc2.texts + doc2.groups if hasattr(item, 'self_ref')}

    refs1 = set(items1.keys())
    refs2 = set(items2.keys())

    # Identify added and removed items based on references
    added_refs = refs2 - refs1
    removed_refs = refs1 - refs2
    common_refs = refs1 & refs2

    for ref in added_refs:
        item = items2[ref]
        diff_results["added_items"].append({
            "ref": ref,
            "label": str(getattr(item, 'label', 'Group')),
            "text_or_name": getattr(item, 'text', getattr(item, 'name', 'N/A'))[:100],
            "parent_ref": item.parent.cref if hasattr(item, 'parent') and item.parent else 'None'
        })

    for ref in removed_refs:
        item = items1[ref]
        diff_results["removed_items"].append({
            "ref": ref,
            "label": str(getattr(item, 'label', 'Group')),
            "text_or_name": getattr(item, 'text', getattr(item, 'name', 'N/A'))[:100],
            "parent_ref": item.parent.cref if hasattr(item, 'parent') and item.parent else 'None'
        })

    # Identify modified and potentially moved items
    for ref in common_refs:
        item1 = items1[ref]
        item2 = items2[ref]

        mods = {}
        # Compare text/name content
        text1 = getattr(item1, 'text', getattr(item1, 'name', None))
        text2 = getattr(item2, 'text', getattr(item2, 'name', None))
        if text1 != text2:
            mods["content"] = {"from": text1[:100], "to": text2[:100]}

        # Compare parent reference
        parent1_ref = item1.parent.cref if hasattr(item1, 'parent') and item1.parent else None
        parent2_ref = item2.parent.cref if hasattr(item2, 'parent') and item2.parent else None
        if parent1_ref != parent2_ref:
            mods["parent"] = {"from": parent1_ref, "to": parent2_ref}

        # Compare label (less common to change, but possible)
        label1 = getattr(item1, 'label', None)
        label2 = getattr(item2, 'label', None)
        if label1 != label2:
            mods["label"] = {"from": str(label1), "to": str(label2)}

        # Add to appropriate list
        if mods:
            if "content" in mods and len(mods) == 1: # Only content changed
                 diff_results["modified_items"].append({"ref": ref, "changes": mods})
            elif "parent" in mods: # Parent changed, potentially moved
                 diff_results["moved_items"].append({"ref": ref, "changes": mods})
            else: # Other modifications
                 diff_results["modified_items"].append({"ref": ref, "changes": mods})


    logger.info(f"Comparison Summary: Added={len(diff_results['added_items'])}, "
                f"Removed={len(diff_results['removed_items'])}, "
                f"Modified={len(diff_results['modified_items'])}, "
                f"Moved={len(diff_results['moved_items'])}")
    return diff_results

def compare_raw_text(text1: str, text2: str) -> List[str]:
    """Compares two strings using difflib and returns unified diff."""
    d = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile='doc1',
        tofile='doc2',
        n=3 # Context lines
    )
    return list(d)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Define Files to Compare ---
    # Example: Comparing two versions of Planbestemmelser for case 1
    # Assume you create a modified version, e.g., Planbestemmelser_case1_mod.json
    # For now, let's compare case1 and case2 Planbestemmelser as an example of difference
    doc1_path = DOCLING_PARSED_DIR / "Planbestemmelser_case1_structure.json"
    doc2_path = DOCLING_PARSED_DIR / "Planbestemmelser_case2_structure.json" # Using case2 as the "modified" version for demo

    if not doc1_path.exists() or not doc2_path.exists():
        logger.error(f"Required Docling JSON files not found in {DOCLING_PARSED_DIR}. Run parsers first.")
    else:
        try:
            # --- Structured Comparison ---
            logger.info("--- Running Structured Difference Analysis ---")
            doc1 = load_docling_doc(doc1_path)
            doc2 = load_docling_doc(doc2_path)
            structured_diff = compare_docling_structures(doc1, doc2)

            # Save structured diff
            diff_file_struct = DIFF_OUTPUT_DIR / f"diff_{doc1.name}_vs_{doc2.name}_structured.json"
            with open(diff_file_struct, "w", encoding='utf-8') as f:
                json.dump(structured_diff, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved structured diff results to {diff_file_struct}")

            # --- Raw Text Comparison ---
            logger.info("\n--- Running Raw Text Difference Analysis ---")
            # Export text from DoclingDocuments for fair comparison of extracted content
            text1 = doc1.export_to_text()
            text2 = doc2.export_to_text()
            raw_diff = compare_raw_text(text1, text2)

            # Save raw diff
            diff_file_raw = DIFF_OUTPUT_DIR / f"diff_{doc1.name}_vs_{doc2.name}_raw.diff"
            with open(diff_file_raw, "w", encoding='utf-8') as f:
                f.writelines(raw_diff)
            logger.info(f"Saved raw text diff results to {diff_file_raw}")

            print("\nComparison Complete. Check the 'results/difference_analysis' folder.")

        except Exception as e:
            logger.error(f"Error during difference analysis: {e}", exc_info=True)