# parse_plankart.py
import fitz # PyMuPDF
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import io
from PIL import Image # Make sure Pillow is installed: pip install Pillow

from docling_core.types.doc import (
    DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel,
    BoundingBox, ProvenanceItem, GroupLabel, Size, ImageRef
)

from docling_core.types.doc.document import ContentLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Regex to identify potential zone identifiers
ZONE_ID_REGEX = re.compile(r"^(?:[fo]_)?([A-ZÆØÅ]+)\d+(?:-\d+)?$|^#\d+$")
# Regex to identify the legend section
LEGEND_KEYWORD_REGEX = re.compile(r"Tegnforklaring|Reguleringsplan PBL|§12-")

def parse_plankart(pdf_path: Path, doc_id: str) -> DoclingDocument:
    """
    Parses a Plankart PDF, extracting text elements, positions, page images,
    and identifying legend/zones.
    """
    logger.info(f"Parsing Plankart: {pdf_path.name}")
    doc = DoclingDocument(id=doc_id, name=pdf_path.stem)
    pdf_doc = fitz.open(pdf_path)

    # Add a title for the document itself
    doc.add_title(
        text=f"Plankart: {pdf_path.name}", 
        orig=f"Plankart: {pdf_path.name}",
        content_layer=ContentLayer.BODY
    )

    # Create groups for different content types
    legend_group = doc.add_group(
        name="Tegnforklaring", 
        label=GroupLabel.UNSPECIFIED,
        parent=doc.body,
        content_layer=ContentLayer.FURNITURE  # Legends are often considered furniture
    )

    page_groups = {}  # To hold groups for each page's main content
    char_offset = 0  # Global character offset tracking

    for page_num, page in enumerate(pdf_doc):
        page_width = page.rect.width
        page_height = page.rect.height

        # --- Add PageItem with Image ---
        page_image_ref = None
        try:
            zoom = 2  # Use zoom for better image quality
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            
            # Create ImageRef from PIL Image
            page_image_ref = ImageRef.from_pil(pil_image, dpi=int(72 * zoom))
            logger.debug(f"Successfully created ImageRef for page {page_num}")
            
        except Exception as e:
            logger.warning(f"Failed to extract or create ImageRef for page {page_num}: {e}")

        # Add page metadata to the document
        page_item = doc.add_page(
            page_no=page_num,
            size=Size(width=page_width, height=page_height),
            image=page_image_ref
        )

        # Create a group for this page's main content
        page_group = doc.add_group(
            name=f"Page {page_num + 1}",
            label=GroupLabel.UNSPECIFIED,
            parent=doc.body,
            content_layer=ContentLayer.BODY
        )
        page_groups[page_num] = page_group

        # --- Process Text Blocks ---
        try:
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
        except Exception as e:
            logger.error(f"Error getting text blocks for page {page_num}: {e}")
            continue

        for block in blocks:
            if "lines" not in block:
                continue

            block_text = " ".join(
                "".join(span["text"] for span in line["spans"]) 
                for line in block["lines"]
            )
            block_is_legend = bool(LEGEND_KEYWORD_REGEX.search(block_text))

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    try:
                        bbox = BoundingBox(
                            l=span["bbox"][0], 
                            t=span["bbox"][1], 
                            r=span["bbox"][2], 
                            b=span["bbox"][3]
                        )
                        start_char = char_offset
                        end_char = start_char + len(text)
                        char_offset = end_char + 1

                        # Create single provenance item
                        prov = ProvenanceItem(
                            page_no=page_num,
                            bbox=bbox,
                            charspan=(start_char, end_char)
                        )

                        # Determine content type and parent
                        is_legend_item = block_is_legend or bool(LEGEND_KEYWORD_REGEX.search(text))
                        parent_group = legend_group if is_legend_item else page_group
                        content_layer = ContentLayer.FURNITURE if is_legend_item else ContentLayer.BODY

                        # Determine appropriate item type based on content
                        if ZONE_ID_REGEX.match(text):
                            # Zone identifiers get special treatment
                            doc.add_text(
                                label=DocItemLabel.TEXT,
                                text=text,
                                orig=span["text"],
                                prov=prov,  # Single provenance item
                                parent=parent_group,
                                content_layer=content_layer
                            )
                        else:
                            # Regular text gets added as paragraphs
                            doc.add_text(
                                label=DocItemLabel.PARAGRAPH,
                                text=text,
                                orig=span["text"],
                                prov=prov,  # Single provenance item
                                parent=parent_group,
                                content_layer=content_layer
                            )

                    except Exception as e:
                        logger.error(f"Failed to add text item '{text}': {e}")

    logger.info(f"Finished parsing. Total text items: {len(doc.texts)}, Groups: {len(doc.groups)}, Pages: {len(doc.pages)}")
    return doc

# --- Helper to print structure (reuse from previous script) ---
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
             # print(f"{indent}Skipping already visited: {node_id}")
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
                    # Check if child_ref is already a RefItem or needs to be created/resolved
                    if isinstance(child_ref, RefItem):
                         child_item = child_ref.resolve(doc)
                    elif isinstance(child_ref, str) and child_ref.startswith("#/"): # Assuming string refs
                         child_item = RefItem(cref=child_ref).resolve(doc)
                    else:
                         # Attempt to resolve directly if it's maybe an object already (less likely)
                         # This part might need adjustment based on how children are stored
                         child_item = child_ref if hasattr(child_ref, 'self_ref') else None
                         if not child_item:
                              print(f"{indent}  Cannot resolve child: {child_ref}")
                              continue

                    print_node(child_item, indent + "  ")
                 except Exception as e:
                     child_ref_str = child_ref.cref if isinstance(child_ref, RefItem) else str(child_ref)
                     print(f"{indent}  Error resolving/printing child {child_ref_str}: {e}")

    # Start printing from the body's children
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

    # Print page info including images
    print("\nPage Information:")
    for page_no, page_item in doc.pages.items():
         img_info = "with image" if page_item.image else "no image"
         print(f"  Page {page_no}: Size({page_item.size.width}x{page_item.size.height}), {img_info}")

    print("--- End of Structure ---")


# --- Main Execution ---
if __name__ == "__main__":
    case1_pdf_path = Path("experiments/data/plankart/Plankart.pdf")
    case2_pdf_path = Path("experiments/data/plankart/Plankart_20240515.pdf")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if case1_pdf_path.exists():
        logger.info(f"Processing {case1_pdf_path}...")
        doc1 = parse_plankart(case1_pdf_path, "Plankart")
        print_docling_structure(doc1)
        # Save the result
        try:
            doc1.save_as_json(output_dir / f"{doc1.name}_structure.json", image_mode='placeholder') # Save without huge embedded images
            logger.info(f"Saved {doc1.name} structure to JSON.")
        except Exception as e:
             logger.error(f"Could not save {doc1.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case1_pdf_path}")

    if case2_pdf_path.exists():
        logger.info(f"Processing {case2_pdf_path}...")
        doc2 = parse_plankart(case2_pdf_path, "Plankart_20240515")
        print_docling_structure(doc2)
         # Save the result
        try:
            doc2.save_as_json(output_dir / f"{doc2.name}_structure.json", image_mode='placeholder')
            logger.info(f"Saved {doc2.name} structure to JSON.")
        except Exception as e:
             logger.error(f"Could not save {doc2.name} to JSON: {e}")
    else:
        logger.error(f"File not found: {case2_pdf_path}")