# experiments/parse_plankart.py (REVISED - More Specific Types)
import fitz  # PyMuPDF
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import sys
import io
from PIL import Image
sys.path.append('/home/are/.local/lib/python3.10/site-packages')

# Ensure imports from docling_core
try:
    from docling_core.types.doc.document import (
        DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel,
        BoundingBox, ProvenanceItem, GroupLabel, ContentItem, NodeItem,
        SectionHeaderItem, PageItem, Size, ContentLayer, ImageRef
    )
except ImportError as e:
    print(f"ERROR: Failed to import docling-core types: {e}")
    print("Make sure docling-core is installed and in your PYTHONPATH")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Regexes
ZONE_ID_REGEX = re.compile(r"^(?:[fo]_)?([A-ZÆØÅ]+)\d+(?:-\d+)?$|^#\d+$")
LEGEND_KEYWORD_REGEX = re.compile(r"Tegnforklaring|Reguleringsplan PBL|§\s?12-")
SECTION_REF_REGEX = re.compile(r"^\s*§\s?\d{1,2}-")
MAP_COORDINATE_REGEX = re.compile(r"^[ØN]\d+$")  # For map coordinates like Ø424400, N6489400

def extract_page_image(page, zoom=2):
    """Extract image from PDF page with specified zoom level."""
    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        return ImageRef.from_pil(pil_image, dpi=int(72 * zoom))
    except Exception as e:
        logger.warning(f"Failed to extract page image: {e}")
        return None

def parse_plankart(pdf_path: Path, doc_id: str) -> DoclingDocument:
    logger.info(f"Parsing Plankart: {pdf_path.name}")
    doc = DoclingDocument(id=doc_id, name=pdf_path.stem)
    pdf_doc = None
    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return doc

    # Create main groups for the document
    doc.add_title(text=f"Plankart: {pdf_path.name}", orig=f"Plankart: {pdf_path.name}", parent=doc.body)
    
    # Create groups for different content types
    legend_group = doc.add_group(name="Tegnforklaring", label=GroupLabel.KEY_VALUE_AREA,
                                parent=doc.body, content_layer=ContentLayer.FURNITURE)
    main_content_group = doc.add_group(name="Map Content", label=GroupLabel.CHAPTER,
                                      parent=doc.body, content_layer=ContentLayer.BODY)

    # Create sub-groups for map identifiers and legend entries
    map_identifiers_group = doc.add_group(name="Map Identifiers", label=GroupLabel.SECTION,
                                         parent=main_content_group, content_layer=ContentLayer.BODY)
    legend_entries_group = doc.add_group(name="Legend Entries", label=GroupLabel.SECTION,
                                        parent=legend_group, content_layer=ContentLayer.FURNITURE)

    char_offset = 0

    for page_num, page in enumerate(pdf_doc):
        page_width, page_height = page.rect.width, page.rect.height
        
        # Extract page image
        page_image = extract_page_image(page)
        
        # Create page group
        page_group = doc.add_group(name=f"Page {page_num + 1}", label=GroupLabel.SHEET,
                                  parent=main_content_group, content_layer=ContentLayer.BODY)
        
        # Add page to document with image
        doc.add_page(page_no=page_num, size=Size(width=page_width, height=page_height), image=page_image)

        try:
            # Get text blocks with their positions
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT | 
                                 fitz.TEXT_PRESERVE_LIGATURES | 
                                 fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        except Exception as e:
            logger.error(f"Error getting text blocks for page {page_num}: {e}")
            continue

        current_legend_section_group = legend_entries_group

        for block in blocks:
            if "lines" not in block:
                continue

            # Check if block is part of legend
            first_line_text = "".join(span["text"] for span in block["lines"][0]["spans"]).strip()
            block_is_legend = bool(LEGEND_KEYWORD_REGEX.search(first_line_text))
            block_is_legend_section_header = bool(SECTION_REF_REGEX.match(first_line_text))
            block_is_map_coordinate = bool(MAP_COORDINATE_REGEX.match(first_line_text))

            # Determine parent group based on content type
            parent_group = current_legend_section_group if block_is_legend else page_group
            content_layer = ContentLayer.FURNITURE if block_is_legend else ContentLayer.BODY

            if block_is_legend and block_is_legend_section_header:
                # Create new section group for legend
                current_legend_section_group = doc.add_group(
                    name=first_line_text,
                    label=GroupLabel.SECTION,
                    parent=legend_entries_group,
                    content_layer=ContentLayer.FURNITURE
                )
                parent_group = current_legend_section_group

            for line in block["lines"]:
                line_orig = "".join(span["text"] for span in line["spans"])
                line_text_stripped = line_orig.strip()
                if not line_text_stripped:
                    continue

                try:
                    # Get line bounding box
                    bbox = BoundingBox(
                        l=line["bbox"][0],
                        t=line["bbox"][1],
                        r=line["bbox"][2],
                        b=line["bbox"][3]
                    )
                    
                    # Calculate character span
                    start_char = char_offset
                    end_char = start_char + len(line_orig)
                    char_offset = end_char + 1
                    
                    # Create provenance
                    prov = ProvenanceItem(
                        page_no=page_num,
                        bbox=bbox,
                        charspan=(start_char, end_char)
                    )

                    # Determine item type and add to document
                    if ZONE_ID_REGEX.match(line_text_stripped):
                        # Zone identifier on map
                        doc.add_text(
                            label=DocItemLabel.TEXT,
                            text=line_text_stripped,
                            orig=line_orig,
                            parent=map_identifiers_group,
                            content_layer=ContentLayer.BODY,
                            prov=prov
                        )
                    elif MAP_COORDINATE_REGEX.match(line_text_stripped):
                        # Map coordinate
                        doc.add_text(
                            label=DocItemLabel.TEXT,
                            text=line_text_stripped,
                            orig=line_orig,
                            parent=map_identifiers_group,
                            content_layer=ContentLayer.BODY,
                            prov=prov
                        )
                    elif block_is_legend_section_header and line == block["lines"][0]:
                        # Legend section header
                        doc.add_text(
                            label=DocItemLabel.SECTION_HEADER,
                            text=line_text_stripped,
                            orig=line_orig,
                            parent=parent_group,
                            content_layer=content_layer,
                            prov=prov
                        )
                    elif block_is_legend:
                        # Legend entry
                        doc.add_text(
                            label=DocItemLabel.PARAGRAPH,
                            text=line_text_stripped,
                            orig=line_orig,
                            parent=parent_group,
                            content_layer=content_layer,
                            prov=prov
                        )
                    else:
                        # Regular text on map
                        doc.add_text(
                            label=DocItemLabel.PARAGRAPH,
                            text=line_text_stripped,
                            orig=line_orig,
                            parent=page_group,
                            content_layer=content_layer,
                            prov=prov
                        )

                except Exception as e:
                    logger.error(f"Failed processing line '{line_text_stripped}': {e}")

    if pdf_doc:
        pdf_doc.close()
    
    logger.info(f"Finished parsing {pdf_path.name}. Texts: {len(doc.texts)}, Groups: {len(doc.groups)}, Pages: {len(doc.pages)}")
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

    print("\nPage Information:")
    for page_no, page_item in doc.pages.items():
         img_info = "with image" if page_item.image else "no image"
         print(f"  Page {page_no}: Size({page_item.size.width}x{page_item.size.height}), {img_info}")

    print("--- End of Structure ---")


# --- Main Execution ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    case1_pdf_path = script_dir / "data/plankart/Evjetun_leirsted.pdf"
    case2_pdf_path = script_dir / "data/plankart/Kjetså_massetak.pdf"
    output_dir = script_dir / "results/docling_parsed/plankart" # Specific output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if case1_pdf_path.exists():
        logger.info(f"Processing {case1_pdf_path}...")
        doc1 = parse_plankart(case1_pdf_path, "Evjetun_leirsted")
        if doc1:
            print_docling_structure(doc1)
            try:
                doc1.save_as_json(output_dir / f"{doc1.name}_structure.json", image_mode='placeholder')
                logger.info(f"Saved {doc1.name} structure to JSON.")
            except Exception as e:
                 logger.error(f"Could not save {doc1.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case1_pdf_path}")

    if case2_pdf_path.exists():
        logger.info(f"Processing {case2_pdf_path}...")
        doc2 = parse_plankart(case2_pdf_path, "Kjetså_massetak")
        if doc2:
            print_docling_structure(doc2)
            try:
                doc2.save_as_json(output_dir / f"{doc2.name}_structure.json", image_mode='placeholder')
                logger.info(f"Saved {doc2.name} structure to JSON.")
            except Exception as e:
                 logger.error(f"Could not save {doc2.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case2_pdf_path}")