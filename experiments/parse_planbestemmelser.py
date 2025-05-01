# parse_planbestemmelser.py
import fitz  # PyMuPDF
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

from docling_core.types.doc import (
    DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel,
    BoundingBox, ProvenanceItem, GroupLabel
)
from docling_core.types.doc.document import ContentLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regex to detect numbered headings (e.g., 1., 1.1, 2.3.4)
SECTION_HEADING_REGEX = re.compile(r"^\s*(\d+(\.\d+)*)[\.\s]*(.*)")

def detect_section_level(section_id: str) -> int:
    """Determines heading level based on dots."""
    return section_id.count('.') + 1

def clean_text(text: str) -> str:
    """Removes extra whitespace and handles hyphenation."""
    text = re.sub(r'-\n', '', text) # Basic de-hyphenation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_likely_footer_or_header(text: str, y_pos: float, page_height: float) -> bool:
    """Check if text is likely a header or footer based on position and content."""
    margin = 72 # Roughly 1 inch margin in points
    if y_pos < margin or y_pos > page_height - margin:
        # Common header/footer patterns
        if re.match(r'Side \d+ av \d+', text) or \
           re.match(r'Side \d+ \| \d+', text) or \
           re.search(r'PlanID \d+', text) or \
           re.search(r'Plan ID: \d+', text) or \
           re.search(r'Detaljregulering for', text) or \
           re.search(r'Vedtatt PS \d+/\d+', text) or \
           re.search(r'Dato: \d+\.\d+\.\d+', text):
           return True
    return False

def parse_planbestemmelser(pdf_path: Path, doc_id: str) -> DoclingDocument:
    """
    Parses a Planbestemmelser PDF into a hierarchical DoclingDocument.
    """
    logger.info(f"Parsing Planbestemmelser: {pdf_path.name}")
    doc = DoclingDocument(id=doc_id, name=pdf_path.stem)
    pdf_doc = fitz.open(pdf_path)

    # Track the current hierarchy of section headers
    # Keys are levels (int), values are the corresponding DocItem (TextItem)
    section_stack: Dict[int, TextItem] = {0: doc.body}
    current_paragraph_text = ""
    last_item = None

    for page_num, page in enumerate(pdf_doc):
        page_height = page.rect.height
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"] # Get detailed block info

        for block in blocks:
            if "lines" not in block: continue

            for line in block["lines"]:
                line_text = ""
                line_bbox = fitz.Rect() # Bounding box for the whole line
                if not line["spans"]: continue

                # Combine spans into a single line text and calculate line bbox
                first_span_bbox = fitz.Rect(line["spans"][0]["bbox"])
                line_bbox = first_span_bbox
                for span in line["spans"]:
                    line_text += span["text"]
                    line_bbox.include_rect(fitz.Rect(span["bbox"]))
                line_text = line_text.strip()

                if not line_text: continue

                # Determine if this is furniture (header/footer)
                is_furniture = is_likely_footer_or_header(line_text, line_bbox.y0, page_height)
                if is_furniture:
                    # Add as furniture text if it's header/footer
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=line_text,
                        orig=line_text,
                        parent=doc.furniture,
                        content_layer=ContentLayer.FURNITURE
                    )
                    logger.debug(f"Added furniture text: '{line_text}'")
                    continue

                heading_match = SECTION_HEADING_REGEX.match(line_text)

                # --- Process potential heading ---
                if heading_match:
                    # First, add any accumulated paragraph text before processing the heading
                    if current_paragraph_text:
                        current_level = max(section_stack.keys())
                        parent_item = section_stack[current_level]
                        # Use add_text for paragraphs, linking to the last active heading
                        para_item = doc.add_text(
                            label=DocItemLabel.PARAGRAPH,
                            text=clean_text(current_paragraph_text),
                            orig=current_paragraph_text,
                            parent=parent_item,
                            content_layer=ContentLayer.BODY
                        )
                        logger.debug(f"Added paragraph under {parent_item.label if hasattr(parent_item, 'label') else 'body'}: {clean_text(current_paragraph_text)[:50]}...")
                        current_paragraph_text = ""
                        last_item = para_item

                    section_id = heading_match.group(1)
                    title_text = heading_match.group(3).strip()
                    level = detect_section_level(section_id)

                    # Find the correct parent in the stack
                    parent_level = level - 1
                    while parent_level >= 0 and parent_level not in section_stack:
                        parent_level -= 1
                    parent_item = section_stack.get(parent_level, doc.body)

                    # Use add_heading, setting the parent explicitly
                    heading_item = doc.add_heading(
                        text=f"{section_id} {title_text}",
                        orig=line_text,
                        level=level,
                        parent=parent_item,
                        content_layer=ContentLayer.BODY
                    )
                    logger.info(f"Added heading (L{level}, Parent: {parent_item.label if hasattr(parent_item, 'label') else 'body'}): {heading_item.text}")

                    # Update the stack: remove deeper levels and add current
                    levels_to_remove = [lvl for lvl in section_stack if lvl >= level]
                    for lvl in levels_to_remove:
                        del section_stack[lvl]
                    section_stack[level] = heading_item
                    last_item = heading_item

                # --- Process paragraph text ---
                else:
                    # Append to the current paragraph buffer
                    if current_paragraph_text and not current_paragraph_text.endswith(' '):
                        current_paragraph_text += " "
                    current_paragraph_text += line_text

        # End of page: Add any remaining paragraph text for this page
        if current_paragraph_text:
            current_level = max(section_stack.keys())
            parent_item = section_stack[current_level]
            para_item = doc.add_text(
                label=DocItemLabel.PARAGRAPH,
                text=clean_text(current_paragraph_text),
                orig=current_paragraph_text,
                parent=parent_item,
                content_layer=ContentLayer.BODY
            )
            logger.debug(f"Added EOP paragraph under {parent_item.label if hasattr(parent_item, 'label') else 'body'}: {clean_text(current_paragraph_text)[:50]}...")
            current_paragraph_text = ""
            last_item = para_item

    # Add any final remaining paragraph text after the last page
    if current_paragraph_text:
         current_level = max(section_stack.keys())
         parent_item = section_stack[current_level]
         final_para = doc.add_text(
             label=DocItemLabel.PARAGRAPH,
             text=clean_text(current_paragraph_text),
             orig=current_paragraph_text,
             parent=parent_item,
             content_layer=ContentLayer.BODY
         )
         logger.debug(f"Added final paragraph under {parent_item.label if hasattr(parent_item, 'label') else 'body'}: {final_para.text[:50]}...")

    logger.info(f"Finished parsing. Total items: {len(doc.texts)}")
    return doc

# --- Helper to print structure ---
def print_docling_structure(doc: DoclingDocument, max_depth=10):
    """Prints the hierarchical structure of a DoclingDocument."""
    print(f"\n--- Document Structure: {doc.name} ({len(doc.texts)} items) ---")
    visited = set()

    def print_item(item, indent=""):
        if not item or item.self_ref in visited:
            return
        visited.add(item.self_ref)

        item_info = f"[{item.label}]" if hasattr(item, 'label') else f"[{type(item).__name__}]"
        text_preview = ""
        if hasattr(item, 'text'):
            text_preview = item.text[:80].replace('\n', ' ') + ('...' if len(item.text) > 80 else '')
        elif hasattr(item, 'name'):
             text_preview = f"Name: {item.name}"

        print(f"{indent}{item_info} {text_preview} (Ref: {item.self_ref}, Parent: {item.parent.cref if item.parent else 'None'})")

        if hasattr(item, 'children') and len(indent) < max_depth * 2: # Limit recursion depth
            for child_ref in item.children:
                 try:
                    child_item = child_ref.resolve(doc)
                    print_item(child_item, indent + "  ")
                 except Exception as e:
                     print(f"{indent}  Error resolving child {child_ref.cref}: {e}")

    # Start printing from the body's children
    if doc.body and hasattr(doc.body, 'children'):
         print("Body Children:")
         for child_ref in doc.body.children:
              try:
                 item = child_ref.resolve(doc)
                 print_item(item)
              except Exception as e:
                   print(f" Error resolving top-level child {child_ref.cref}: {e}")
    else:
         print("Document body has no children or is not defined properly.")
    print("--- End of Structure ---")


# --- Main Execution ---
if __name__ == "__main__":
    case1_pdf_path = Path("experiments/data/planbestemmelser/Planbestemmelser.pdf")
    case2_pdf_path = Path("experiments/data/planbestemmelser/bestemmelser.pdf")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if case1_pdf_path.exists():
        logger.info(f"Processing {case1_pdf_path}...")
        doc1 = parse_planbestemmelser(case1_pdf_path, "Planbestemmelser")
        print_docling_structure(doc1)
        # Save the result
        try:
            doc1.save_as_json(output_dir / f"{doc1.name}_structure.json")
            logger.info(f"Saved {doc1.name} structure to JSON.")
        except Exception as e:
             logger.error(f"Could not save {doc1.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case1_pdf_path}")

    if case2_pdf_path.exists():
        logger.info(f"Processing {case2_pdf_path}...")
        doc2 = parse_planbestemmelser(case2_pdf_path, "bestemmelser")
        print_docling_structure(doc2)
        # Save the result
        try:
            doc2.save_as_json(output_dir / f"{doc2.name}_structure.json")
            logger.info(f"Saved {doc2.name} structure to JSON.")
        except Exception as e:
             logger.error(f"Could not save {doc2.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case2_pdf_path}")