# experiments/parse_planbestemmelser.py
import fitz  # PyMuPDF
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import sys
sys.path.append('/home/are/.local/lib/python3.10/site-packages')

# Ensure imports from docling_core
try:
    from docling_core.types.doc.document import (
        DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel,
        BoundingBox, ProvenanceItem, GroupLabel, ContentItem, NodeItem,
        SectionHeaderItem, PageItem, Size, ContentLayer
    )
except ImportError as e:
    print(f"ERROR: Failed to import docling-core types: {e}")
    print("Make sure docling-core is installed and in your PYTHONPATH")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Regular expressions for parsing
SECTION_HEADING_REGEX = re.compile(r'^(\d+(?:\.\d+)*)\s*(?:\s+|[–-])\s*(.+?)$')
LIST_ITEM_REGEX = re.compile(r'^(?:[•\-–]\s*|\d+\.\s*|\w\)\s*)(.+)$')

def detect_section_level(section_id: str) -> int:
    """Detect section level based on the number of dot-separated numbers."""
    return len(section_id.split('.'))

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing characters."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('–', '-').strip()
    return text

def is_likely_footer_or_header(text: str, y_pos: float, page_height: float) -> bool:
    """Check if text is likely to be a header or footer based on position and content."""
    if not text: return False
    text_lower = text.lower()
    
    # Position-based checks (top 10% or bottom 10% of page)
    is_top = y_pos < page_height * 0.1
    is_bottom = y_pos > page_height * 0.9
    
    # Content-based checks
    has_page_indicator = bool(re.search(r'side\s*\d+|\d+\s*av\s*\d+', text_lower))
    is_short = len(text) < 30
    looks_like_metadata = bool(re.search(r'vedtatt|dato|plan\s*id|saksnr', text_lower))
    
    return (is_top or is_bottom) and (has_page_indicator or (is_short and looks_like_metadata))

def parse_planbestemmelser(pdf_path: Path, doc_id: str) -> DoclingDocument:
    logger.info(f"Parsing Planbestemmelser: {pdf_path.name}")
    doc = DoclingDocument(id=doc_id, name=pdf_path.stem)
    pdf_doc = None
    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return doc

    # section_stack: Keys=level (int), Values=NodeItem (Heading or Group or Body)
    section_stack: Dict[int, Union[GroupItem, SectionHeaderItem]] = {0: doc.body}
    current_paragraph_buffer = []
    last_added_item_ref: Optional[RefItem] = None

    char_offset = 0

    for page_num, page in enumerate(pdf_doc):
        page_width, page_height = page.rect.width, page.rect.height
        doc.add_page(page_no=page_num, size=Size(width=page_width, height=page_height))

        try:
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT | fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except Exception as e:
            logger.error(f"Error getting text blocks for page {page_num}: {e}")
            continue

        for block in blocks:
            if "lines" not in block: continue

            for line in block["lines"]:
                line_orig = "".join(span["text"] for span in line["spans"]) # Keep original spacing from spans
                line_text_stripped = line_orig.strip()
                if not line_text_stripped: continue

                line_bbox_fitz = fitz.Rect(line["bbox"])
                bbox = BoundingBox(l=line_bbox_fitz.x0, t=line_bbox_fitz.y0, r=line_bbox_fitz.x1, b=line_bbox_fitz.y1)
                start_char = char_offset
                end_char = start_char + len(line_orig)
                char_offset = end_char + 1 # Add 1 for potential newline/space
                prov = ProvenanceItem(page_no=page_num, bbox=bbox, charspan=(start_char, end_char))

                is_furniture = is_likely_footer_or_header(line_text_stripped, line_bbox_fitz.y0, page_height)

                # --- Flush Paragraph if needed ---
                # Flush if current line is furniture, heading, or list item, AND buffer is not empty
                heading_match = SECTION_HEADING_REGEX.match(line_text_stripped)
                list_match = LIST_ITEM_REGEX.match(line_text_stripped)
                if (is_furniture or heading_match or list_match) and current_paragraph_buffer:
                    parent_level = max(section_stack.keys())
                    parent_item = section_stack[parent_level]
                    para_text = clean_text(" ".join(current_paragraph_buffer))
                    if para_text:
                        last_added_item_ref = doc.add_text(
                            label=DocItemLabel.PARAGRAPH, text=para_text,
                            orig=" ".join(current_paragraph_buffer), parent=parent_item,
                            content_layer=ContentLayer.BODY, prov=prov
                            # Missing provenance for multi-line paragraph - harder to add accurately here
                        ).get_ref()
                        logger.debug(f"Flushed paragraph under {parent_item.self_ref}: {para_text[:50]}...")
                    current_paragraph_buffer = [] # Reset buffer

                # --- Process Furniture ---
                if is_furniture:
                    if line_bbox_fitz.y0 < page_height * 0.5:
                        doc.add_text(label=DocItemLabel.PAGE_HEADER, text=line_text_stripped, orig=line_orig,
                                   parent=doc.furniture, content_layer=ContentLayer.FURNITURE, prov=prov)
                    else:
                        doc.add_text(label=DocItemLabel.PAGE_FOOTER, text=line_text_stripped, orig=line_orig,
                                   parent=doc.furniture, content_layer=ContentLayer.FURNITURE, prov=prov)
                    continue # Skip further processing

                # --- Process Heading ---
                elif heading_match:
                    section_id = heading_match.group(1)
                    title_text = heading_match.group(2).strip()
                    level = detect_section_level(section_id)

                    parent_level = level - 1
                    while parent_level >= 0 and parent_level not in section_stack: parent_level -= 1
                    parent_item = section_stack.get(parent_level, doc.body)

                    # Create a section group first
                    section_group = doc.add_group(
                        label=GroupLabel.SECTION, parent=parent_item,
                        content_layer=ContentLayer.BODY
                    )

                    # Then add the header as a child of the section group
                    heading_item = doc.add_text(
                        label=DocItemLabel.SECTION_HEADER, text=f"{section_id} {title_text}",
                        orig=line_orig, parent=section_group, content_layer=ContentLayer.BODY,
                        prov=prov
                    )
                    logger.info(f"Added heading (L{level}, Parent Ref: {parent_item.self_ref}): {heading_item.text}")

                    levels_to_remove = [lvl for lvl in section_stack if lvl >= level]
                    for lvl in levels_to_remove: del section_stack[lvl]
                    section_stack[level] = section_group
                    last_added_item_ref = heading_item.get_ref()

                # --- Process List Item ---
                elif list_match:
                    parent_level = max(section_stack.keys())
                    parent_item = section_stack[parent_level]
                    list_text = clean_text(list_match.group(1))

                    # Logic to find/create list group
                    list_group_parent = parent_item # Default parent
                    last_node = last_added_item_ref.resolve(doc) if last_added_item_ref else None

                    # If the immediate parent item is already a list group, use it
                    if isinstance(parent_item, GroupItem) and parent_item.label == GroupLabel.LIST:
                         list_group_parent = parent_item
                    # Or if the *last added item* was a list item, use *its* parent (the list group)
                    elif isinstance(last_node, TextItem) and last_node.label == DocItemLabel.LIST_ITEM:
                          if last_node.parent:
                               list_group_parent = last_node.parent.resolve(doc)
                          else: # Should not happen if structure is correct
                               logger.warning("List item found without a parent group, creating new list.")
                               list_group_parent = doc.add_group(label=GroupLabel.LIST, parent=parent_item, content_layer=ContentLayer.BODY)
                    # Otherwise, create a new list group under the current section/parent
                    else:
                         list_group_parent = doc.add_group(label=GroupLabel.LIST, parent=parent_item, content_layer=ContentLayer.BODY)

                    list_item = doc.add_text(
                         label=DocItemLabel.LIST_ITEM, text=list_text, orig=line_orig,
                         parent=list_group_parent, content_layer=ContentLayer.BODY,
                         prov=prov
                    )
                    last_added_item_ref = list_item.get_ref()

                # --- Process Paragraph Line ---
                else:
                    # Append to the current paragraph buffer
                    # We handle merging/adding when a non-paragraph line or end-of-page is encountered
                    current_paragraph_buffer.append(line_orig)

    # --- Add Final Paragraph ---
    if current_paragraph_buffer:
        parent_level = max(section_stack.keys())
        parent_item = section_stack[parent_level]
        para_text = clean_text(" ".join(current_paragraph_buffer))
        if para_text:
            doc.add_text(label=DocItemLabel.PARAGRAPH, text=para_text,
                         orig=" ".join(current_paragraph_buffer), parent=parent_item,
                         content_layer=ContentLayer.BODY)
        logger.debug(f"Added final paragraph under L{parent_level}: {para_text[:50]}...")

    if pdf_doc: pdf_doc.close()
    logger.info(f"Finished parsing {pdf_path.name}. Texts: {len(doc.texts)}, Groups: {len(doc.groups)}")
    return doc

# --- (Keep print_docling_structure helper function as before) ---
def print_docling_structure(doc: DoclingDocument, max_depth=10):
    """Prints the hierarchical structure of a DoclingDocument."""
    print(f"\n--- Document Structure: {doc.name} ({len(doc.texts)} text, {len(doc.groups)} groups, {len(doc.pages)} pages) ---")
    visited_items = set()
    visited_groups = set()

    def print_node(node, indent=""):
        if not node or not hasattr(node, 'self_ref'):
            print(f"{indent}Invalid node encountered.")
            return

        is_group = isinstance(node, GroupItem)
        node_id = node.self_ref
        visited_set = visited_groups if is_group else visited_items

        if node_id in visited_set:
             return
        visited_set.add(node_id)

        item_info = f"[{node.label}]" if hasattr(node, 'label') else f"[Group]"
        text_preview = ""
        if hasattr(node, 'text'):
            text_preview = node.text[:80].replace('\n', ' ') + ('...' if len(node.text) > 80 else '')
        elif hasattr(node, 'name'):
             text_preview = f"Name: {node.name}"

        parent_ref = node.parent.cref if hasattr(node, 'parent') and node.parent else 'None'
        print(f"{indent}{item_info} {text_preview} (Ref: {node.self_ref}, Parent: {parent_ref})")

        if hasattr(node, 'children') and len(indent) < max_depth * 2:
            for child_ref in node.children:
                 try:
                    if isinstance(child_ref, RefItem):
                         child_item = child_ref.resolve(doc)
                    elif isinstance(child_ref, str) and child_ref.startswith("#/"):
                         child_item = RefItem(cref=child_ref).resolve(doc)
                    else:
                         child_item = child_ref if hasattr(child_ref, 'self_ref') else None
                         if not child_item: continue
                    print_node(child_item, indent + "  ")
                 except Exception as e:
                     child_ref_str = child_ref.cref if isinstance(child_ref, RefItem) else str(child_ref)
                     print(f"{indent}  Error resolving/printing child {child_ref_str}: {e}")

    if doc.body and hasattr(doc.body, 'children'):
         print("Body Children:")
         for child_ref in doc.body.children:
              try:
                 item = child_ref.resolve(doc)
                 print_node(item)
              except Exception as e:
                   print(f" Error resolving top-level child {child_ref.cref}: {e}")
    else:
         print("Document body has no children or is not defined properly.")
    print("--- End of Structure ---")

# --- Main Execution ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    case1_pdf_path = script_dir / "data/planbestemmelser/Evjetun_leirsted.pdf"
    case2_pdf_path = script_dir / "data/planbestemmelser/Kjetså_massetak.pdf"
    output_dir = script_dir / "results/docling_parsed/planbestemmelser" # Specific output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if case1_pdf_path.exists():
        logger.info(f"Processing {case1_pdf_path}...")
        doc1 = parse_planbestemmelser(case1_pdf_path, "Evjetun_leirsted")
        if doc1:
            print_docling_structure(doc1)
            try:
                doc1.save_as_json(output_dir / f"{doc1.name}_structure.json")
                logger.info(f"Saved {doc1.name} structure to JSON.")
            except Exception as e:
                 logger.error(f"Could not save {doc1.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case1_pdf_path}")

    if case2_pdf_path.exists():
        logger.info(f"Processing {case2_pdf_path}...")
        doc2 = parse_planbestemmelser(case2_pdf_path, "Kjetså_massetak")
        if doc2:
            print_docling_structure(doc2)
            try:
                doc2.save_as_json(output_dir / f"{doc2.name}_structure.json")
                logger.info(f"Saved {doc2.name} structure to JSON.")
            except Exception as e:
                 logger.error(f"Could not save {doc2.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case2_pdf_path}")