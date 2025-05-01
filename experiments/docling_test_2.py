# %%
# 1
# Import required libraries
import os
import re
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from typing import Dict, List, Tuple, Optional

from docling_core.types.doc import (
    DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel, 
    BoundingBox, ProvenanceItem, GroupLabel
)

# Set up basic logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Libraries imported successfully")
# %%
# 2
# Create necessary directories
directories = [
    "data/planbestemmelser",
    "results"
]

for d in directories:
    os.makedirs(d, exist_ok=True)
    print(f"✓ Created: {d}")
# %%
# 3
def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_section_level(section_id: str) -> int:
    """
    Determine the heading level based on the section ID (e.g., '1', '1.1', '1.1.1')
    Returns an integer representing the heading level (1-based).
    """
    if not section_id:
        return 0
    
    # Count the number of dots and add 1
    return section_id.count('.') + 1

def extract_section_info(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract section ID and title from a line of text.
    Returns a tuple of (section_id, title) or (None, None) if not a section heading.
    """
    # Match section headings with pattern like "1.", "1.1", "1.1.1", etc.
    section_match = re.match(r'^(\d+(\.\d+)*)[.\s]+(.+)$', line.strip())
    
    if section_match:
        section_id = section_match.group(1)
        title = section_match.group(3).strip()
        return section_id, title
    
    # Try alternative patterns for numbered lists that aren't section headings
    # For example "1) Item" or "a) Item"
    if re.match(r'^[a-z\d][)]\s', line.strip()):
        return None, None
    
    return None, None

print("Helper functions defined")
def parse_planbestemmelser_with_hierarchy(pdf_path: Path, doc_id: str) -> DoclingDocument:
    """
    Parse a Planbestemmelser PDF and create a DoclingDocument with proper hierarchical structure.
    
    Args:
        pdf_path: Path to the PDF file
        doc_id: Identifier for the created document
        
    Returns:
        A DoclingDocument with hierarchical structure of sections and content
    """
    logger.info(f"Parsing Planbestemmelser document: {pdf_path}")
    
    # Create initial document
    doc = DoclingDocument(id=doc_id, name=doc_id)
    
    try:
        # Extract text from PDF
        pdf_doc = fitz.open(pdf_path)
        
        # Track the hierarchy by level
        section_hierarchy = {}
        current_section = None
        current_level = 0
        
        # Get all text at once for better context
        all_text = ""
        for page_num, page in enumerate(pdf_doc):
            all_text += page.get_text("text")
        
        # Split into lines and clean up
        lines = all_text.split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        
        # Process line by line
        current_paragraph = ""
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip page headers/footers
            if re.match(r'^Side\s+\d+\s+av\s+\d+', line) or "Detaljregulering for Evjetun" in line:
                i += 1
                continue
            
            # Try to extract section heading
            section_id, title = extract_section_info(line)
            
            if section_id:
                # If we have accumulated paragraph text, add it to current section
                if current_paragraph:
                    if current_section:
                        doc.add_text(
                            label=DocItemLabel.PARAGRAPH,
                            text=clean_text(current_paragraph),
                            orig=current_paragraph,
                            parent=current_section
                        )
                    else:
                        doc.add_text(
                            label=DocItemLabel.PARAGRAPH,
                            text=clean_text(current_paragraph),
                            orig=current_paragraph
                        )
                    current_paragraph = ""
                
                # Determine section level
                level = detect_section_level(section_id)
                
                # Find parent section (search backwards through levels)
                parent_section = None
                for l in range(level-1, 0, -1):
                    if l in section_hierarchy:
                        parent_section = section_hierarchy[l]
                        break
                
                # Create new section with full heading text
                full_heading = f"{section_id} {title}"
                
                # Use add_text with SECTION_HEADER label
                new_section = doc.add_text(
                    label=DocItemLabel.SECTION_HEADER,
                    text=full_heading,
                    orig=line,
                    parent=parent_section
                )
                
                # Store this section at its level
                section_hierarchy[level] = new_section
                
                # Set as current section
                current_section = new_section
                current_level = level
            else:
                # Check for potential subsection titles without numbering
                if line.isupper() and len(line.split()) <= 5 and current_section is not None:
                    # This might be a non-numbered subsection title
                    # Add as a subsection of the current section
                    subsection = doc.add_text(
                        label=DocItemLabel.SECTION_HEADER,
                        text=line,
                        orig=line,
                        parent=current_section
                    )
                    # Don't update the section hierarchy since this is a special case
                    current_section = subsection
                else:
                    # Normal paragraph text
                    # Check if this line might be continuing a previous paragraph
                    if current_paragraph and not line[0].isupper() and not line[0].isdigit():
                        # Likely continuation of previous paragraph
                        current_paragraph += " " + line
                    else:
                        # New paragraph
                        if current_paragraph:
                            # Save previous paragraph
                            if current_section:
                                doc.add_text(
                                    label=DocItemLabel.PARAGRAPH,
                                    text=clean_text(current_paragraph),
                                    orig=current_paragraph,
                                    parent=current_section
                                )
                            else:
                                doc.add_text(
                                    label=DocItemLabel.PARAGRAPH,
                                    text=clean_text(current_paragraph),
                                    orig=current_paragraph
                                )
                        # Start new paragraph
                        current_paragraph = line
            
            i += 1
        
        # Add any remaining paragraph
        if current_paragraph:
            if current_section:
                doc.add_text(
                    label=DocItemLabel.PARAGRAPH,
                    text=clean_text(current_paragraph),
                    orig=current_paragraph,
                    parent=current_section
                )
            else:
                doc.add_text(
                    label=DocItemLabel.PARAGRAPH,
                    text=clean_text(current_paragraph),
                    orig=current_paragraph
                )
        
        logger.info(f"Successfully created document with {len(doc.texts)} text items")
        return doc
    
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        raise e

print("Planbestemmelser parser function defined")
# %%
# 4
def print_document_structure(doc: DoclingDocument):
    """
    Print the hierarchical structure of a DoclingDocument for debugging.
    
    Args:
        doc: The DoclingDocument to print
    """
    # Print basic document info safely
    print("\n📄 Document:")
    if hasattr(doc, 'name'):
        print(f"Name: {doc.name}")
    elif hasattr(doc, 'id'):
        print(f"ID: {doc.id}")
    else:
        print("(No name or ID available)")
    
    # Print text items count
    text_items_count = 0
    if hasattr(doc, 'texts'):
        text_items_count = len(doc.texts)
    print(f"Total text items: {text_items_count}")
    
    # Find and print all section headers
    section_headers = []
    if hasattr(doc, 'texts'):
        section_headers = [t for t in doc.texts if hasattr(t, 'label') and t.label == DocItemLabel.SECTION_HEADER]
    
    print("\n🔷 Section Heading Structure:")
    for header in section_headers:
        # Get heading level if available
        level = 1
        if hasattr(header, 'level'):
            level = header.level
        elif hasattr(header, 'get_attr') and callable(header.get_attr):
            try:
                level = header.get_attr('level', 1)
            except:
                level = 1
        
        indent = "  " * (level - 1)
        print(f"{indent}{'#' * level} {header.text}")
    
    # Print full structure with content
    print("\n🔷 Full Document Structure:")
    
    # Get root items (those without parents)
    root_items = []
    if hasattr(doc, 'texts'):
        for item in doc.texts:
            has_parent = False
            if hasattr(item, 'parent') and item.parent is not None:
                has_parent = True
            if not has_parent:
                root_items.append(item)
    
    # Track visited items to avoid duplicates in recursive traversal
    visited = set()
    
    def print_item(item, level=0):
        if id(item) in visited:
            return
        visited.add(id(item))
        
        indent = "  " * level
        
        if hasattr(item, 'label'):
            if item.label == DocItemLabel.SECTION_HEADER:
                hlevel = 1
                if hasattr(item, 'level'):
                    hlevel = item.level
                elif hasattr(item, 'get_attr') and callable(item.get_attr):
                    try:
                        hlevel = item.get_attr('level', 1)
                    except:
                        hlevel = 1
                
                print(f"{indent}{'#' * hlevel} {item.text}")
            elif item.label == DocItemLabel.PARAGRAPH:
                # Truncate long paragraphs for display
                text = item.text
                if len(text) > 100:
                    text = text[:97] + "..."
                print(f"{indent}📝 {text}")
            else:
                print(f"{indent}📄 [{item.label}] {item.text}")
        elif hasattr(item, 'name'):
            # It's a group
            print(f"{indent}📂 {item.name}")
        
        # Process children if any
        children = []
        try:
            if hasattr(doc, 'get_children') and callable(doc.get_children):
                children = doc.get_children(item)
            elif hasattr(item, 'children'):
                children = item.children
        except:
            # If there's any error getting children, just continue
            pass
        
        for child in children:
            print_item(child, level + 1)
    
    # Start with root level items
    for item in root_items:
        print_item(item)

print("Document structure printer function defined")


# %%
# 5
# Path to your Planbestemmelser PDF - adjust as needed
planbestemmelser_path = Path("data/planbestemmelser/Planbestemmelser.pdf")

# Check if the file exists
if not planbestemmelser_path.exists():
    print(f"❌ File not found: {planbestemmelser_path}")
    print("Please make sure the PDF file is in the correct location.")
else:
    print(f"✓ Found PDF file: {planbestemmelser_path}")
    
    try:
        # Parse the document
        doc = parse_planbestemmelser_with_hierarchy(planbestemmelser_path, "planbestemmelser_1")
        
        # Print document structure
        print_document_structure(doc)
        
        # Print some statistics safely
        section_count = 0
        paragraph_count = 0
        if hasattr(doc, 'texts'):
            section_count = len([t for t in doc.texts if hasattr(t, 'label') and t.label == DocItemLabel.SECTION_HEADER])
            paragraph_count = len([t for t in doc.texts if hasattr(t, 'label') and t.label == DocItemLabel.PARAGRAPH])
        
        print(f"\n📊 Document Statistics:")
        print(f"  - Total sections: {section_count}")
        print(f"  - Total paragraphs: {paragraph_count}")
        print(f"  - Total text items: {len(doc.texts) if hasattr(doc, 'texts') else 0}")
        
        # Print first few items for inspection
        if hasattr(doc, 'texts') and len(doc.texts) > 0:
            print("\n📋 First 10 items:")
            for i, item in enumerate(doc.texts[:10]):
                text_preview = item.text[:50] + "..." if len(item.text) > 50 else item.text
                print(f"{i}: [{item.label}] {text_preview}")
        
        print("\n✅ Document parsed successfully")
    except Exception as e:
        print(f"❌ Error parsing document: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
# %%
# 6
def find_sections_by_title_pattern(doc: DoclingDocument, pattern: str):
    """
    Find all sections whose titles match a given pattern.
    
    Args:
        doc: The DoclingDocument to search
        pattern: Regular expression pattern to match against section titles
        
    Returns:
        List of matching section items
    """
    matching_sections = []
    if hasattr(doc, 'texts'):
        for item in doc.texts:
            if (hasattr(item, 'label') and 
                item.label == DocItemLabel.SECTION_HEADER and 
                re.search(pattern, item.text, re.IGNORECASE)):
                matching_sections.append(item)
    
    return matching_sections

def get_section_content(doc: DoclingDocument, section_item):
    """
    Get all content (paragraphs) belonging to a specific section.
    
    Args:
        doc: The DoclingDocument
        section_item: The section item to get content for
        
    Returns:
        List of text items that are children of the section
    """
    content = []
    
    # Check if the document has a method to get children
    try:
        if hasattr(doc, 'get_children') and callable(doc.get_children):
            content = doc.get_children(section_item)
        else:
            # Manual lookup based on parent reference
            if hasattr(doc, 'texts'):
                for item in doc.texts:
                    if hasattr(item, 'parent') and item.parent == section_item:
                        content.append(item)
    except Exception as e:
        print(f"Error getting section content: {e}")
    
    return content

# Try to find some specific sections
if 'doc' in locals():
    print("\n🔍 Finding specific sections:")
    
    # Find sections related to "bestemmelser"
    bestemmelser_sections = find_sections_by_title_pattern(doc, r'bestemmelser')
    print(f"\nFound {len(bestemmelser_sections)} sections related to 'bestemmelser':")
    for section in bestemmelser_sections:
        print(f"  - {section.text}")
    
    # Find sections related to "arealformål"
    arealformål_sections = find_sections_by_title_pattern(doc, r'arealformål')
    print(f"\nFound {len(arealformål_sections)} sections related to 'arealformål':")
    for section in arealformål_sections:
        print(f"  - {section.text}")
    
    # Find sections that might contain zone identifiers 
    zone_pattern = r'([A-Z]+\d+|[fo]_[A-Z]+\d+)'
    zone_sections = find_sections_by_title_pattern(doc, zone_pattern)
    print(f"\nFound {len(zone_sections)} sections potentially containing zone identifiers:")
    for section in zone_sections:
        print(f"  - {section.text}")
        
    # Try to find a section about the plan's purpose ("Planens hensikt")
    target_sections = find_sections_by_title_pattern(doc, r'planens hensikt')
    
    if target_sections:
        section = target_sections[0]
        print(f"\n📑 Content for section '{section.text}':")
        
        content = get_section_content(doc, section)
        
        if content:
            for item in content:
                print(f"\n{item.text}")
        else:
            print("No content found for this section.")
    else:
        print("\nNo section about 'Planens hensikt' found.")

# %%
# 7
# Try to find some specific sections
if 'doc' in locals():
    print("\n🔍 Finding specific sections:")
    
    # Find sections related to "bestemmelser"
    bestemmelser_sections = find_sections_by_title_pattern(doc, r'bestemmelser')
    print(f"\nFound {len(bestemmelser_sections)} sections related to 'bestemmelser':")
    for section in bestemmelser_sections:
        print(f"  - {section.text}")
    
    # Find sections related to "arealformål"
    arealformål_sections = find_sections_by_title_pattern(doc, r'arealformål')
    print(f"\nFound {len(arealformål_sections)} sections related to 'arealformål':")
    for section in arealformål_sections:
        print(f"  - {section.text}")
    
    # Find sections that might contain zone identifiers 
    zone_pattern = r'([A-Z]+\d+|[fo]_[A-Z]+\d+)'
    zone_sections = find_sections_by_title_pattern(doc, zone_pattern)
    print(f"\nFound {len(zone_sections)} sections potentially containing zone identifiers:")
    for section in zone_sections:
        print(f"  - {section.text}")
        
    # Try to find a section about the plan's purpose ("Planens hensikt")
    target_sections = find_sections_by_title_pattern(doc, r'planens hensikt')
    
    if target_sections:
        section = target_sections[0]
        print(f"\n📑 Content for section '{section.text}':")
        
        content = get_section_content(doc, section)
        
        if content:
            for item in content:
                print(f"\n{item.text}")
        else:
            print("No content found for this section.")
    else:
        print("\nNo section about 'Planens hensikt' found.")
        
    # Check for the first section header, if any are available
    section_headers = [t for t in doc.texts if hasattr(t, 'label') and t.label == DocItemLabel.SECTION_HEADER]
    if section_headers:
        print("\n📑 Content for the first section:")
        first_section = section_headers[0]
        content = get_section_content(doc, first_section)
        
        if content:
            for item in content:
                print(f"\n{item.text}")
        else:
            print("No content found for this section.")
        
def get_section_content(doc: DoclingDocument, section_item):
    """
    Get all content (paragraphs) belonging to a specific section.
    
    Args:
        doc: The DoclingDocument
        section_item: The section item to get content for
        
    Returns:
        List of text items that are children of the section
    """
    content = []
    
    # Check if the document has a method to get children
    if hasattr(doc, 'get_children'):
        content = doc.get_children(section_item)
    else:
        # Manual lookup based on parent reference
        for item in doc.texts:
            if hasattr(item, 'parent') and item.parent == section_item:
                content.append(item)
    
    return content





# %%
# 8
def save_document_structure(doc: DoclingDocument, output_path: Path):
    """
    Save the document structure to a text file for later reference.
    
    Args:
        doc: The DoclingDocument to save
        output_path: Path where to save the output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write basic document info
        f.write("Document Structure\n")
        f.write("=================\n\n")
        
        if hasattr(doc, 'name'):
            f.write(f"Name: {doc.name}\n")
        if hasattr(doc, 'id'):
            f.write(f"ID: {doc.id}\n")
        
        text_count = len(doc.texts) if hasattr(doc, 'texts') else 0
        f.write(f"Total text items: {text_count}\n\n")
        
        # Find section headers
        section_headers = []
        if hasattr(doc, 'texts'):
            section_headers = [t for t in doc.texts if hasattr(t, 'label') and t.label == DocItemLabel.SECTION_HEADER]
        
        f.write("SECTION STRUCTURE:\n")
        f.write("=================\n\n")
        
        for header in section_headers:
            # Get heading level if available
            level = 1
            if hasattr(header, 'level'):
                level = header.level
            elif hasattr(header, 'get_attr') and callable(header.get_attr):
                try:
                    level = header.get_attr('level', 1)
                except:
                    level = 1
            
            indent = "  " * (level - 1)
            f.write(f"{indent}{'#' * level} {header.text}\n")
        
        f.write("\nFULL CONTENT:\n")
        f.write("=============\n\n")
        
        # Get root items (those without parents)
        root_items = []
        if hasattr(doc, 'texts'):
            for item in doc.texts:
                has_parent = False
                if hasattr(item, 'parent') and item.parent is not None:
                    has_parent = True
                if not has_parent:
                    root_items.append(item)
        
        # Track visited items to avoid duplicates in recursive traversal
        visited = set()
        
        def write_item(item, level=0):
            if id(item) in visited:
                return
            visited.add(id(item))
            
            indent = "  " * level
            
            if hasattr(item, 'label'):
                if item.label == DocItemLabel.SECTION_HEADER:
                    hlevel = 1
                    if hasattr(item, 'level'):
                        hlevel = item.level
                    elif hasattr(item, 'get_attr') and callable(item.get_attr):
                        try:
                            hlevel = item.get_attr('level', 1)
                        except:
                            hlevel = 1
                    
                    f.write(f"{indent}{'#' * hlevel} {item.text}\n")
                elif item.label == DocItemLabel.PARAGRAPH:
                    f.write(f"{indent}PARAGRAPH: {item.text}\n")
                else:
                    f.write(f"{indent}[{item.label}] {item.text}\n")
            elif hasattr(item, 'name'):
                # It's a group
                f.write(f"{indent}GROUP: {item.name}\n")
            
            # Process children if any
            children = []
            try:
                if hasattr(doc, 'get_children') and callable(doc.get_children):
                    children = doc.get_children(item)
                elif hasattr(item, 'children'):
                    children = item.children
            except:
                # If there's any error getting children, just continue
                pass
            
            for child in children:
                write_item(child, level + 1)
        
        # Start with root level items
        for item in root_items:
            write_item(item)

# Save document structure to file
if 'doc' in locals():
    output_path = Path("results/planbestemmelser_structure.txt")
    try:
        save_document_structure(doc, output_path)
        print(f"\n✅ Document structure saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving document structure: {str(e)}")

# %%
# 9
import json

def document_to_json(doc: DoclingDocument, output_path: Path):
    """
    Convert a DoclingDocument to a JSON representation and save it to a file.
    
    Args:
        doc: The DoclingDocument to convert
        output_path: Path where to save the output JSON
    """
    # Create a dictionary to represent the document structure
    doc_dict = {
        "metadata": {
            "name": doc.name if hasattr(doc, "name") else "",
            "id": doc.id if hasattr(doc, "id") else "",
            "text_count": len(doc.texts) if hasattr(doc, "texts") else 0
        },
        "sections": [],
        "paragraphs": [],
        "relationships": []
    }
    
    # Add sections (headers)
    if hasattr(doc, "texts"):
        # Add all sections
        section_headers = [t for t in doc.texts if hasattr(t, "label") and t.label == DocItemLabel.SECTION_HEADER]
        for header in section_headers:
            # Get level if available
            level = 1
            if hasattr(header, "level"):
                level = header.level
            elif hasattr(header, "get_attr") and callable(header.get_attr):
                try:
                    level = header.get_attr("level", 1)
                except:
                    level = 1
            
            # Add to sections list
            section_info = {
                "id": id(header),  # Use object id as unique identifier
                "text": header.text,
                "level": level,
                "parent_id": id(header.parent) if hasattr(header, "parent") and header.parent is not None else None
            }
            doc_dict["sections"].append(section_info)
        
        # Add all paragraphs
        paragraphs = [t for t in doc.texts if hasattr(t, "label") and t.label == DocItemLabel.PARAGRAPH]
        for para in paragraphs:
            para_info = {
                "id": id(para),
                "text": para.text,
                "parent_id": id(para.parent) if hasattr(para, "parent") and para.parent is not None else None
            }
            doc_dict["paragraphs"].append(para_info)
        
        # Add parent-child relationships
        for item in doc.texts:
            if hasattr(item, "parent") and item.parent is not None:
                rel = {
                    "child_id": id(item),
                    "parent_id": id(item.parent),
                    "child_type": str(item.label) if hasattr(item, "label") else "unknown",
                    "parent_type": str(item.parent.label) if hasattr(item.parent, "label") else "unknown"
                }
                doc_dict["relationships"].append(rel)
    
    # Write to file as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Document structure exported as JSON to {output_path}")
    return doc_dict

# Use the function to export the document structure as JSON
if 'doc' in locals():
    json_output_path = Path("results/planbestemmelser_structure.json")
    try:
        doc_json = document_to_json(doc, json_output_path)
        print("\n✅ Document structure exported as JSON")
    except Exception as e:
        print(f"❌ Error exporting document structure to JSON: {str(e)}")
        import traceback
        traceback.print_exc()
# %%
