# %% 
%pip install --quiet docling docling-core pymupdf pandas matplotlib seaborn scikit-learn
print("OKAY")


# %%
import os

directories = [
    "data/planbestemmelser",
    "data/plankart",
    "data/sosi",
    "results"
]

for d in directories:
    os.makedirs(d, exist_ok=True)
    print("✔ Created:", d)

# %%
import shutil, os

# Mapping for easier movement
case_files = {
    "case1": {
        "planbestemmelser": "data/planbestemmelser/Planbestemmelser.pdf",
        "plankart": "data/plankart/Plankart.pdf",
        "sosi": "data/sosi/Evjetun.sos"
    },
    "case2": {
        "planbestemmelser": "data/planbestemmelser/bestemmelser.pdf",
        "plankart": "data/plankart/Plankart_20240515.pdf",
        "sosi": "data/sosi/Kjetså_massetak.sos"
    }
}

# Copy files into structured mock folders
for case, files in case_files.items():
    for doc_type, source_path in files.items():
        filename = os.path.basename(source_path)
        target_folder = f"data/mock/{doc_type}/{case}"
        os.makedirs(target_folder, exist_ok=True)
        target_path = f"{target_folder}/{filename}"
        shutil.copy2(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")


# %%
import os

# Define document types and cases
doc_types = ["planbestemmelser", "plankart", "sosi"]
cases = ["case1", "case2"]

# Walk through and confirm all files exist
for doc_type in doc_types:
    for case in cases:
        folder = f"data/mock/{doc_type}/{case}"
        print(f"\nContents of {folder}:")
        if os.path.exists(folder):
            for file in os.listdir(folder):
                print(f" - {file}")
        else:
            print(" ❌ Folder does not exist!")
# %%
import pandas as pd

# Load purpose codes from Reguleringsplan.csv
csv_path = "data/Reguleringsplan.csv"  # Make sure you've placed it here
df = pd.read_csv(csv_path, sep=";")

# Clean up the columns
df = df[["SOSI-kode", "Formål"]].dropna()
df["SOSI-kode"] = df["SOSI-kode"].astype(str).str.replace(r"[^\d]", "", regex=True)
df = df[df["SOSI-kode"].str.isnumeric()]  # Only valid codes
df = df.drop_duplicates(subset="SOSI-kode")

# Create mapping dictionary
sosi_purpose_codes = dict(zip(df["SOSI-kode"], df["Formål"]))
print(f"Loaded {len(sosi_purpose_codes)} SOSI purpose codes.")

# %%
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load updated SOSI purpose codes with correct encoding
try:
    df = pd.read_csv("data/Reguleringsplan.csv", sep=";", encoding="utf-8")
    df = df[["SOSI-kode", "Formål"]].dropna()

    df["SOSI-kode"] = (
        df["SOSI-kode"]
        .astype(str)
        .apply(lambda x: re.sub(r"[^\d]", "", x) if "(" in x or ")" in x else x.strip())
        .astype(str)
    )


    df["Formål"] = df["Formål"].astype(str).str.strip()
    purpose_map = dict(zip(df["SOSI-kode"], df["Formål"]))

    print(f"📘 Loaded {len(purpose_map)} SOSI purpose codes.")
    print("🎯 Test lookup for code 6770:")
    print("6770" in purpose_map)
    print(f"Mapped 6770 → {purpose_map.get('6770')}")
except Exception as e:
    print("❌ Failed to load purpose codes:", e)
    purpose_map = {}

print("📋 Columns in CSV:", df.columns.tolist())


# ✅ SOSI Parser class
class SosiParser:
    def __init__(self, purpose_map: Dict[str, str] = None):
        self.purpose_map = purpose_map or {}

    def parse_file(self, file_path: Path) -> Dict:
        logger.info(f"Parsing SOSI file: {file_path}")
        try:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                raw_content = f.read()
                content = raw_content.encode('ISO-8859-1').decode('utf-8', errors='replace')
        except Exception as e:
            logger.error(f"Failed to decode file: {e}")
            raise e
        return self.parse_content(content)


    def parse_content(self, content: str) -> Dict:
        result = {"features": [], "zone_identifiers": []}
        current_feature = None
        current_level = 0
        feature_stack = [result]

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('!'):
                continue

            level_match = re.match(r'^(\.+)', line)
            new_level = len(level_match.group(1)) if level_match else 0
            line = line[new_level:] if level_match else line

            while len(feature_stack) > new_level + 1:
                feature_stack.pop()

            parent_feature = feature_stack[-1] if feature_stack else None

            if ' ' in line:
                key, value = line.split(' ', 1)
                key, value = key.strip(), value.strip()

                if key == "FELTNAVN":
                    result["zone_identifiers"].append(value)
                    if current_feature:
                        current_feature["zone_identifier"] = value

                if key == "RPAREALFORMÅL" and current_feature:
                    current_feature["area_purpose_code"] = value
                    current_feature["area_purpose"] = self.purpose_map.get(value, "ukjent")

                if current_feature and key != "FLATE":
                    current_feature.setdefault("attributes", {})[key] = value
                elif parent_feature and parent_feature != result and key != "FLATE":
                    parent_feature.setdefault("attributes", {})[key] = value

            if line.startswith("FLATE"):
                feature_id = line.split(" ", 1)[1].strip(":")
                current_feature = {
                    "id": feature_id,
                    "type": "FLATE",
                    "attributes": {},
                    "children": []
                }
                result["features"].append(current_feature)
                feature_stack.append(current_feature)

        return result

    def extract_zone_identifiers(self, sosi_data: Dict) -> List[str]:
        zones = sosi_data.get("zone_identifiers", [])
        if not zones:
            for feature in sosi_data.get("features", []):
                if "zone_identifier" in feature:
                    zones.append(feature["zone_identifier"])
                elif "attributes" in feature and "FELTNAVN" in feature["attributes"]:
                    zones.append(feature["attributes"]["FELTNAVN"])
        return list(set(zones))

    def process_file(self, file_path: Path) -> Dict:
        sosi_data = self.parse_file(file_path)
        return {
            "metadata": {
                "filename": file_path.name
            },
            "zones": self.extract_zone_identifiers(sosi_data),
            "raw_data": sosi_data
        }

# ✅ Try it out again!
parser = SosiParser(purpose_map=purpose_map)
sosi_path = Path("data/mock/sosi/case1/Evjetun.sos")
result = parser.process_file(sosi_path)

print(f"\n📌 Found {len(result['zones'])} zone identifiers.")
print(f"🗂️ Zone identifiers: {result['zones'][:5]}")
print("\n📒 Sample feature with purpose mapping:")
for f in result["raw_data"]["features"]:
    if "zone_identifier" in f:
        print(f)
        break
# %%
# Cell 7a: Convert real files into structured Docling-style documents (corrected paths)
from pathlib import Path
import fitz  # PyMuPDF
import re

# 🧠 Extract hierarchy from a PDF based on numbering
def extract_structure_from_pdf(pdf_path: Path, doc_type: str, doc_id: str):
    doc = fitz.open(pdf_path)
    sections = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        lines = text.split("\n")
        for line in lines:
            match = re.match(r"^(\d{1,2}(\.\d{1,2})*)\s+(.+)", line.strip())
            if match:
                section_id = match.group(1)
                title = match.group(3)
                sections.append({
                    "id": section_id,
                    "title": title,
                    "text": "",
                    "zones": []
                })
    return {
        "id": doc_id,
        "type": doc_type,
        "sections": sections
    }

# 🧾 Convert parsed SOSI structure to docling-like format
def sosi_to_docling(sosi_parsed: dict, doc_id: str):
    return {
        "id": doc_id,
        "type": "SOSI",
        "sections": [
            {
                "id": f["id"],
                "title": f.get("zone_identifier", "ukjent"),
                "text": "",
                "zones": [f.get("zone_identifier")] if "zone_identifier" in f else [],
                "purpose": f.get("area_purpose", "ukjent"),
            }
            for f in sosi_parsed["raw_data"]["features"]
        ]
    }

# Define parser and process the SOSI file first
parser = SosiParser(purpose_map=purpose_map)
result = parser.process_file(Path("data/mock/sosi/case1/Evjetun.sos"))

# Then use it in the doc_structures dictionary
doc_structures = {
    "planbestemmelser_1": extract_structure_from_pdf(Path("data/mock/planbestemmelser/case1/Planbestemmelser.pdf"), "Planbestemmelser", "planbestemmelser_1"),
    "planbestemmelser_2": extract_structure_from_pdf(Path("data/mock/planbestemmelser/case2/bestemmelser.pdf"), "Planbestemmelser", "planbestemmelser_2"),
    "plankart_1": extract_structure_from_pdf(Path("data/mock/plankart/case1/Plankart.pdf"), "Plankart", "plankart_1"),
    "plankart_2": extract_structure_from_pdf(Path("data/mock/plankart/case2/Plankart_20240515.pdf"), "Plankart", "plankart_2"),
    "sosi_1": sosi_to_docling(result, "Evjetun"),
    "sosi_2": sosi_to_docling(SosiParser(purpose_map).process_file(Path("data/mock/sosi/case2/Kjetså_massetak.sos")), "Kjetså")
}

# %%
import fitz  # PyMuPDF
import re
from pathlib import Path
from docling_core.types.doc import (
    DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel
)

# Regex for detecting hierarchical headings
section_heading_regex = re.compile(r"^\d+(\.\d+)*")

def parse_planbestemmelser_to_docling(pdf_path: Path, doc_id: str) -> DoclingDocument:
    # Create initial document with required fields
    doc = DoclingDocument(
        id=doc_id,
        name=doc_id
    )
    
    # Extract text from PDF using PyMuPDF
    pdf_doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(pdf_doc):
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) < 5:
                continue
            text = block[4].strip()
            if not text:
                continue
            
            is_heading = bool(section_heading_regex.match(text))
            
            # Use the built-in methods to add content
            if is_heading:
                doc.add_heading(text=text, orig=text, level=1)
            else:
                doc.add_text(label=DocItemLabel.PARAGRAPH, text=text, orig=text)
    
    return doc

# Run it with try/except to catch any errors
try:
    doc1 = parse_planbestemmelser_to_docling(
        Path("data/mock/planbestemmelser/case1/Planbestemmelser.pdf"), 
        "planbestemmelser_1"
    )
    
    # Only proceed with doc2 if doc1 was successful
    doc2 = parse_planbestemmelser_to_docling(
        Path("data/mock/planbestemmelser/case2/bestemmelser.pdf"), 
        "planbestemmelser_2"
    )
    
    # Quick diagnostics
    print(f"📘 Doc 1: {len(doc1.texts)} text items")
    for i, item in enumerate(doc1.texts[:15]):
        print(f"• [{item.label}] {item.text[:200]}...")
        
    print(f"\n📘 Doc 2: {len(doc2.texts)} text items")
    for i, item in enumerate(doc2.texts[:15]):
        print(f"• [{item.label}] {item.text[:200]}...")
    
except Exception as e:
    print(f"Error processing documents: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

# %%
import fitz  # PyMuPDF
import re
from pathlib import Path
from docling_core.types.doc import (
    DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel, GroupLabel
)

def sosi_to_docling_doc(sosi_parsed: dict, doc_id: str) -> DoclingDocument:
    # Create initial document with required fields
    doc = DoclingDocument(
        id=doc_id,
        name=doc_id
    )
    
    # Add features as text items
    for i, feature in enumerate(sosi_parsed["features"]):
        if "zone_identifier" in feature:
            zone = feature.get("zone_identifier", "ukjent")
            purpose = feature.get("area_purpose", "ukjent")
            text = f"{zone} – {purpose}"
            
            # Use the document's built-in method to add the text
            doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text)
    
    return doc

# Try the SOSI document conversion
try:
    sosi_doc1 = sosi_to_docling_doc(result["raw_data"], "sosi_1")
    
    # Get the second SOSI file as well
    sosi_path2 = Path("data/mock/sosi/case2/Kjetså_massetak.sos")
    result2 = parser.process_file(sosi_path2)
    sosi_doc2 = sosi_to_docling_doc(result2["raw_data"], "sosi_2")
    
    # Quick diagnostics
    print(f"📘 SOSI Doc 1: {len(sosi_doc1.texts)} zone items")
    for i, item in enumerate(sosi_doc1.texts[:5]):
        print(f"• [{item.label}] {item.text}")
    
    print(f"\n📘 SOSI Doc 2: {len(sosi_doc2.texts)} zone items")
    for i, item in enumerate(sosi_doc2.texts[:5]):
        print(f"• [{item.label}] {item.text}")
        
except Exception as e:
    print(f"Error processing SOSI documents: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

# %%
import fitz  # PyMuPDF
import re
from pathlib import Path
from docling_core.types.doc import (
    DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel, 
    BoundingBox, ProvenanceItem
)

def parse_plankart_to_docling(pdf_path: Path, doc_id: str) -> DoclingDocument:
    # Create a new document
    doc = DoclingDocument(
        id=doc_id,
        name=doc_id
    )
    
    pdf_doc = fitz.open(pdf_path)
    
    # Add a title for the document
    doc.add_title(
        text=f"Plankart: {pdf_path.name}",
        orig=f"Plankart: {pdf_path.name}"
    )
    
    # Create a group for the legend
    legend_group = doc.add_group(name="Tegnforklaring")
    
    for page_num, page in enumerate(pdf_doc):
        # Add a group for each page
        page_group = doc.add_group(name=f"Page {page_num+1}")
        
        # Extract text with position information
        text_blocks = page.get_text("dict")["blocks"]
        
        # Process text blocks
        text_position = 0  # Track character position for charspan
        for block_idx, block in enumerate(text_blocks):
            if "lines" not in block:
                continue
                
            # Process text with spatial information
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    
                    # Create bounding box for text position
                    bbox = BoundingBox(
                        l=span["bbox"][0],
                        t=span["bbox"][1],
                        r=span["bbox"][2],
                        b=span["bbox"][3]
                    )
                    
                    # Create provenance item with position and charspan
                    char_end = text_position + len(text)
                    prov = ProvenanceItem(
                        page_no=page_num,
                        bbox=bbox,
                        charspan=(text_position, char_end)
                    )
                    text_position = char_end + 1  # +1 for space between spans
                    
                    # Is this part of the legend?
                    is_legend = False
                    if "Tegnforklaring" in text or re.search(r'(Reguleringsplan|§12-5)', text):
                        is_legend = True
                        parent_group = legend_group
                    else:
                        parent_group = page_group
                    
                    # Determine if this is a zone label
                    is_zone = bool(re.match(r'[A-Z]+\d+', text) or 
                                  re.match(r'[fo]_[A-Z]+\d+', text) or
                                  re.match(r'#\d+', text))
                    
                    # Add text with spatial information
                    doc.add_text(
                        label=DocItemLabel.TEXT if is_zone else DocItemLabel.PARAGRAPH,
                        text=text,
                        orig=text,
                        prov=[prov],
                        parent=parent_group
                    )
    
    return doc

# Process both plankart files
try:
    plankart_doc1 = parse_plankart_to_docling(
        Path("data/mock/plankart/case1/Plankart.pdf"), 
        "plankart_1"
    )
    
    plankart_doc2 = parse_plankart_to_docling(
        Path("data/mock/plankart/case2/Plankart_20240515.pdf"), 
        "plankart_2"
    )
    
    # Print summary
    print(f"📘 Plankart Doc 1: {len(plankart_doc1.texts)} text items, {len(plankart_doc1.groups)} groups")
    
    # Print zones (specifically labeled elements)
    zones = [item for item in plankart_doc1.texts if item.label == DocItemLabel.TEXT]
    print(f"\nIdentified {len(zones)} potential zone labels:")
    for i, zone in enumerate(zones[:10]):
        print(f"• Zone: {zone.text}")

    print(f"\nIdentified {len(plankart_doc1.groups)} potential groups:")
    for i, zone in enumerate(plankart_doc1.groups[:10]):
        print(f"• Group: {plankart_doc1.groups[i].name}")
    # Print the first 5 groups
    for i, group in enumerate(plankart_doc1.groups[:5]):
        print(f"• Group: {group.name} with {len(group.texts)} texts")
    # Print the first 5 texts in the group
    for i, text in enumerate(plankart_doc1.groups[0].texts[:5]):
        print(f"  • Text: {text.text} (label: {text.label})")
    
    print(f"\n📘 Plankart Doc 2: {len(plankart_doc2.texts)} text items, {len(plankart_doc2.groups)} groups")

    # Print zones (specifically labeled elements)
    zones = [item for item in plankart_doc2.texts if item.label == DocItemLabel.TEXT]
    print(f"\nIdentified {len(zones)} potential zone labels:")
    for i, zone in enumerate(zones[:10]):
        print(f"• Zone: {zone.text}")
    
except Exception as e:
    print(f"Error processing plankart documents: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

# %%

def create_unified_structure(planbestemmelser_doc, plankart_doc, sosi_doc, case_id):
    """
    Create a unified structure that maps information across all three document types.
    
    Returns a dictionary with zones as keys, and information from all three sources.
    """
    unified_data = {}
    
    # First, collect zone identifiers from plankart
    zone_labels = [item for item in plankart_doc.texts if item.label == DocItemLabel.TEXT]
    for zone_item in zone_labels:
        zone_id = zone_item.text.strip()
        # Skip if not actually a zone ID
        if not re.match(r'([fo]_)?[A-Z]+\d*', zone_id) and not re.match(r'#\d+', zone_id):
            continue
            
        if zone_id not in unified_data:
            unified_data[zone_id] = {
                "case_id": case_id,
                "plankart_info": [],
                "planbestemmelser_sections": [],
                "sosi_details": []
            }
        
        # Correctly access position information if available
        position_info = None
        if zone_item.prov and len(zone_item.prov) > 0:
            prov_item = zone_item.prov[0]
            if hasattr(prov_item, 'bbox'):
                position_info = {
                    "left": prov_item.bbox.l if hasattr(prov_item.bbox, 'l') else None,
                    "top": prov_item.bbox.t if hasattr(prov_item.bbox, 't') else None,
                    "right": prov_item.bbox.r if hasattr(prov_item.bbox, 'r') else None,
                    "bottom": prov_item.bbox.b if hasattr(prov_item.bbox, 'b') else None
                }
        
        unified_data[zone_id]["plankart_info"].append({
            "text": zone_item.text,
            "position": position_info
        })
    
    # Match text from SOSI to zones
    for text_item in sosi_doc.texts:
        if " – " in text_item.text:
            sosi_zone, sosi_type = text_item.text.split(" – ", 1)
            if sosi_zone in unified_data:
                unified_data[sosi_zone]["sosi_details"].append({
                    "type": sosi_type,
                    "full_text": text_item.text
                })
            elif sosi_zone.upper() in unified_data:  # Try case-insensitive match
                unified_data[sosi_zone.upper()]["sosi_details"].append({
                    "type": sosi_type,
                    "full_text": text_item.text
                })
    
    # Match content from planbestemmelser to zones
    relevant_sections = []
    current_section = None
    
    # Find sections with heading levels
    for text_item in planbestemmelser_doc.texts:
        text = text_item.text.strip()
        
        # Check if this is a section heading
        if text_item.label == DocItemLabel.SECTION_HEADER or re.match(r'^\d+(\.\d+)*\s+', text):
            if current_section:
                relevant_sections.append(current_section)
            current_section = {
                "heading": text,
                "content": []
            }
        elif current_section:
            current_section["content"].append(text)
    
    # Add the last section if exists
    if current_section:
        relevant_sections.append(current_section)
    
    # Match sections to zones
    for section in relevant_sections:
        for zone_id in unified_data:
            # Check if zone is mentioned in section heading or content
            if zone_id in section["heading"]:
                unified_data[zone_id]["planbestemmelser_sections"].append(section)
            else:
                # Check in content
                for content_item in section["content"]:
                    if zone_id in content_item:
                        unified_data[zone_id]["planbestemmelser_sections"].append(section)
                        break
    
    return unified_data

# Create unified structures for both cases
try:
    unified_case1 = create_unified_structure(doc1, plankart_doc1, sosi_doc1, "case1")
    unified_case2 = create_unified_structure(doc2, plankart_doc2, sosi_doc2, "case2")

    # Print a more comprehensive summary
    print(f"Case 1: Found {len(unified_case1)} zones with cross-referenced information")
    for zone_id, data in list(unified_case1.items())[:5]:  # Show first 5 zones
        print(f"\nZone: {zone_id}")
        # Plankart information
        print(f"  Plankart appearances: {len(data['plankart_info'])} instances")
        if data['plankart_info']:
            position = data['plankart_info'][0].get('position')
            if position:
                print(f"  Position in plankart: (left={position.get('left')}, top={position.get('top')})")
            
        # SOSI information
        print(f"  SOSI details: {len(data['sosi_details'])} items")
        if data['sosi_details']:
            print(f"  Zone type from SOSI: {data['sosi_details'][0].get('type', 'Unknown')}")
        
        # Planbestemmelser
        print(f"  Planbestemmelser sections: {len(data['planbestemmelser_sections'])} sections")
        if data['planbestemmelser_sections']:
            print(f"  First relevant section: {data['planbestemmelser_sections'][0]['heading']}")
            if data['planbestemmelser_sections'][0]['content']:
                print(f"    Content sample: {data['planbestemmelser_sections'][0]['content'][0][:80]}...")

    print(f"\nCase 2: Found {len(unified_case2)} zones with cross-referenced information")
    for zone_id, data in list(unified_case2.items())[:5]:  # Show first 5 zones
        print(f"\nZone: {zone_id}")
        # Plankart information
        print(f"  Plankart appearances: {len(data['plankart_info'])} instances")
        if data['plankart_info']:
            position = data['plankart_info'][0].get('position')
            if position:
                print(f"  Position in plankart: (left={position.get('left')}, top={position.get('top')})")
            
        # SOSI information
        print(f"  SOSI details: {len(data['sosi_details'])} items")
        if data['sosi_details']:
            print(f"  Zone type from SOSI: {data['sosi_details'][0].get('type', 'Unknown')}")
        
        # Planbestemmelser
        print(f"  Planbestemmelser sections: {len(data['planbestemmelser_sections'])} sections")
        if data['planbestemmelser_sections']:
            print(f"  First relevant section: {data['planbestemmelser_sections'][0]['heading']}")
            if data['planbestemmelser_sections'][0]['content']:
                print(f"    Content sample: {data['planbestemmelser_sections'][0]['content'][0][:80]}...")
except Exception as e:
    print(f"Error creating unified structure: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
# %%
