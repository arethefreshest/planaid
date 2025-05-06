# experiments/parse_sosi.py (Corrected Hierarchy Logic)
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union # Added Union
import json

script_dir = Path(__file__).parent
output_dir = script_dir / "results/docling_parsed/sosi"
output_dir.mkdir(parents=True, exist_ok=True)

# Set up logging to show debug info
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(output_dir / 'parse_debug.log')  # Save to file
    ]
)
logger = logging.getLogger(__name__)

# Ensure imports from docling_core
try:
    from docling_core.types.doc import (
        DoclingDocument, TextItem, GroupItem, RefItem, DocItemLabel, GroupLabel, ContentLayer, TitleItem
    )
except ImportError:
    print("ERROR: docling-core types not found. Make sure docling-core is installed.")
    # Dummy classes with proper initialization
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
                
        def to_dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    class DoclingDocument:
        def __init__(self, id="", name="", **kwargs):
            self.id = id
            self.name = name
            self.body = GroupItem(name="body")
            self.texts = []
            self.groups = [self.body]

        def add_text(self, label, text, parent=None):
            text_item = TextItem(label=label, text=text, parent=parent)
            self.texts.append(text_item)
            if parent:
                parent.children.append(text_item)
            return text_item

        def add_group(self, name, parent=None):
            group = GroupItem(name=name)
            self.groups.append(group)
            if parent:
                group.parent = parent
                parent.children.append(group)
            return group

        def save_as_json(self, path):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        def to_dict(self):
            return {
                'id': self.id,
                'name': self.name,
                'structure': self.body.to_dict()
            }

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                return super().default(obj)

    class GroupItem(BaseModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.name = kwargs.get('name', '')
            self.parent = kwargs.get('parent', None)
            self.children = []
            self.value = kwargs.get('value', None)

        def to_dict(self):
            result = {
                'name': self.name,
                'children': []
            }
            if self.value is not None:
                result['value'] = self.value
            for child in self.children:
                if isinstance(child, GroupItem):
                    result['children'].append(child.to_dict())
            return result

    class NodeItem(BaseModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.self_ref = kwargs.get('self_ref', '')
            self.children = kwargs.get('children', [])
            self.parent = kwargs.get('parent', None)
            self.content_layer = kwargs.get('content_layer', '')
            
        def to_dict(self):
            d = super().to_dict()
            d['children'] = [child.to_dict() if hasattr(child, 'to_dict') else str(child) 
                           for child in self.children]
            return d

    class DocItem(NodeItem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.label = kwargs.get('label', '')
            self.prov = kwargs.get('prov', [])

    class TextItem(DocItem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.text = kwargs.get('text', '')
            self.orig = kwargs.get('orig', '')
            self.label = kwargs.get('label', '')
            
        def get_ref(self):
            return RefItem()

        def to_dict(self):
            return {
                'label': self.label,
                'text': self.text,
                'parent': self.parent.name if self.parent else None
            }

    class TitleItem(TextItem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.label = 'title'

    class RefItem(BaseModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cref = kwargs.get('cref', '')
            
        def resolve(self, doc):
            return None

    class GroupLabel:
        UNSPECIFIED = "unspecified"

    class ContentLayer:
        BODY = "body"

def load_sosi_purpose_codes(csv_path: Path) -> Dict[str, str]:
    purpose_map = {}
    try:
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8", dtype=str)
        code_col, purpose_col = "SOSI-kode", "Formål"
        if code_col not in df.columns or purpose_col not in df.columns:
            logger.error(f"Cols '{code_col}'/'{purpose_col}' not in {csv_path}")
            return {}
        df_clean = df[[code_col, purpose_col]].dropna()
        df_clean[code_col] = df_clean[code_col].str.replace(r'[() ]', '', regex=True).str.strip()
        # Ensure we only keep rows where the cleaned code is purely numeric
        df_clean = df_clean[df_clean[code_col].str.match(r'^\d+$')]
        df_clean[purpose_col] = df_clean[purpose_col].str.strip()
        purpose_map = dict(zip(df_clean[code_col], df_clean[purpose_col]))
        logger.info(f"Loaded {len(purpose_map)} SOSI purpose codes from {csv_path.name}.")
    except Exception as e: logger.error(f"Error loading SOSI codes: {e}")
    return purpose_map

def parse_value(value_str):
    if value_str is None:
        return None
    value_str = value_str.strip().strip('"')
    try:
        if ' ' in value_str:
            # Handle coordinates with KP
            if '...' in value_str:
                coords, kp = value_str.split('...')
                coords = coords.strip().split()
                return {
                    'coordinates': [int(x) for x in coords],
                    'kp': kp.strip()
                }
            # Handle plain coordinates - convert to integers if possible
            coords = value_str.split()
            try:
                # For NØ coordinates, always treat as coordinates
                if len(coords) == 2:
                    return {
                        'coordinates': [int(x) for x in coords]
                    }
                return [int(x) if x.isdigit() else float(x) for x in coords]
            except ValueError:
                return value_str
        if value_str.isdigit():
            return int(value_str)
        try:
            return float(value_str)
        except ValueError:
            return value_str
    except ValueError:
        return value_str

def collect_no_coordinates(lines, start_idx, level):
    """Collect all coordinates that belong to the same NØ sequence"""
    coordinates = []
    kp = None
    i = start_idx
    
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('!'):
            i += 1
            continue
            
        # If we hit a line starting with dots, we're done with coordinates
        if line.startswith('.'):
            break
            
        # Try to parse coordinates
        parts = line.split()
        if len(parts) >= 2:
            try:
                coord = [int(parts[0]), int(parts[1])]
                coordinates.append(coord)
                # Check for KP value
                if '...KP' in line:
                    kp_idx = parts.index('...KP')
                    if kp_idx + 1 < len(parts):
                        kp = int(parts[kp_idx + 1])
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse coordinates from line: {line} - {str(e)}")
        i += 1
    
    return {'coordinates': coordinates, 'kp': kp}, i

def parse_no_coordinates(line, level):
    """Parse a single line of NØ coordinates"""
    if not line:
        return None
        
    content = line[level:].strip()
    if content.startswith('NØ'):
        return {'coordinates': [], 'kp': None}
        
    parts = content.split()
    if len(parts) < 2:
        return None
        
    try:
        coords = [int(parts[0]), int(parts[1])]
        kp = None
        
        # Check for KP value
        if len(parts) > 2 and '...KP' in content:
            kp_idx = parts.index('...KP')
            if kp_idx + 1 < len(parts):
                kp = int(parts[kp_idx + 1])
                coords.append(kp)
        
        return {'coordinates': coords, 'kp': kp}
    except (ValueError, IndexError):
        logger.warning(f"Failed to parse coordinates from line: {content}")
        return None

def parse_sosi_to_docling(file_path: Path, purpose_map: Dict[str, str], doc_id: str) -> Optional[DoclingDocument]:
    logger.info(f"Parsing SOSI file: {file_path.name}")
    doc = DoclingDocument(id=doc_id, name=file_path.stem)
    
    # Initialize the document structure with body as root
    body_group = doc.body
    
    # Track main category groups
    main_categories = {
        'HODE': doc.add_group(name='HODE', parent=body_group),
        'KURVE': doc.add_group(name='KURVE', parent=body_group),
        'BUEP': doc.add_group(name='BUEP', parent=body_group),
        'FLATE': doc.add_group(name='FLATE', parent=body_group),
        'SYMBOL': doc.add_group(name='SYMBOL', parent=body_group),
        'TEKST': doc.add_group(name='TEKST', parent=body_group)
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            content = f.read()

    def parse_group_name(text):
        match = re.match(r'^(\w+)\s+(\d+):?$', text)
        if match:
            base_name, number = match.groups()
            return base_name, int(number)
        return text.rstrip(':'), None

    lines = content.splitlines()
    current_category = None
    current_instance = None
    level_groups = {0: body_group}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('!') or line == '.SLUTT':
            i += 1
            continue

        # Count dots to determine level
        level_match = re.match(r'^(\.+)', line)
        if not level_match:
            i += 1
            continue

        level = len(level_match.group(1))
        line_content = line[level:].strip()

        # Check if this is a numbered group
        base_name, number = parse_group_name(line_content)

        # Handle main categories and their numbered instances
        if level == 1 and base_name in main_categories:
            current_category = base_name
            if number is not None:
                # Create numbered instance
                current_instance = doc.add_group(name=str(number), parent=main_categories[base_name])
                level_groups = {1: current_instance}  # Reset level groups for new instance
                
                # Look ahead to process all lines until next numbered instance or different category
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if not next_line or next_line.startswith('!'):
                        i += 1
                        continue
                        
                    next_level_match = re.match(r'^(\.+)', next_line)
                    if not next_level_match:
                        i += 1
                        continue
                        
                    next_level = len(next_level_match.group(1))
                    next_content = next_line[next_level:].strip()
                    
                    # Check if this is a new numbered instance or category
                    next_base, next_number = parse_group_name(next_content)
                    if next_level == 1 and (next_base in main_categories):
                        break
                        
                    # Process this line under the current instance
                    parts = next_content.split(' ', 1)
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else None
                    
                    # Find parent for this level
                    parent_level = next_level - 1
                    parent = level_groups.get(parent_level, current_instance)
                    
                    if key == 'NØ':
                        logger.debug(f"Found NØ section at line {i}: {next_content}")
                        # Create a new NØ group
                        no_group = doc.add_group(name=key, parent=parent)
                        
                        # Move to next line after NØ
                        i += 1
                        result, i = collect_no_coordinates(lines, i, next_level)
                        
                        # Set the coordinates and KP on the NØ group
                        no_group.value = result
                        logger.debug(f"Setting NØ group value with {len(result['coordinates'])} coordinates and KP={result['kp']}")
                        continue
                    else:
                        if value is None:
                            # This is a group
                            new_group = doc.add_group(name=key, parent=parent)
                            level_groups[next_level] = new_group
                        else:
                            # This is a value
                            parsed_value = parse_value(value)
                            value_group = doc.add_group(name=key, parent=parent)
                            value_group.value = parsed_value
                    
                    i += 1
                continue
            else:
                # Non-numbered category (like HODE)
                current_instance = main_categories[base_name]
                level_groups = {1: current_instance}
        
        # Handle regular groups and values under non-numbered categories
        if current_instance and level > 1:
            parent_level = level - 1
            parent = level_groups.get(parent_level, current_instance)
            
            parts = line_content.split(' ', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else None
            
            if key == 'NØ':
                # Special handling for NØ coordinates
                coords, skip_lines = collect_no_coordinates(lines, i, level)
                value_group = doc.add_group(name=key, parent=parent)
                if coords:
                    value_group.value = {'coordinates': coords}
                else:
                    value_group.value = {'coordinates': []}
            else:
                if value is None:
                    new_group = doc.add_group(name=key, parent=parent)
                    level_groups[level] = new_group
                else:
                    parsed_value = parse_value(value)
                    value_group = doc.add_group(name=key, parent=parent)
                    value_group.value = parsed_value
        
        i += 1

    logger.info(f"Finished parsing SOSI. Texts: {len(doc.texts)}, Groups: {len(doc.groups)}")
    return doc

def print_docling_structure(doc: DoclingDocument, max_depth=10):
    """Print and return the document structure"""
    output = []
    output.append(f"\n--- Document Structure: {doc.name} ({len(doc.texts)} text, {len(doc.groups)} groups) ---")

    def print_node(node, indent=""):
        if not node:
            return

        # Print group info
        if isinstance(node, GroupItem):
            if node.value is not None:
                if isinstance(node.value, dict) and 'coordinates' in node.value:
                    output.append(f"{indent}[Group] {node.name}: coords={node.value['coordinates']}")
                else:
                    output.append(f"{indent}[Group] {node.name}: {node.value}")
            else:
                output.append(f"{indent}[Group] {node.name}")
            for child in node.children:
                print_node(child, indent + "  ")
    
    # Start with body
    if doc.body:
        print_node(doc.body)
    output.append("--- End of Structure ---")
    
    # Print to console and return as string
    result = "\n".join(output)
    print(result)  # Still print to console for immediate feedback
    return result

# --- Main Execution ---
if __name__ == "__main__":
    csv_path = script_dir / "data/Reguleringsplan.csv"
    case1_sosi_path = script_dir / "data/sosi/Evjetun_leirsted.sos"
    case2_sosi_path = script_dir / "data/sosi/Kjetså_massetak.sos"

    sosi_codes = load_sosi_purpose_codes(csv_path)

    if not sosi_codes:
        logger.error("Cannot proceed without SOSI purpose codes.")
    else:
        if case1_sosi_path.exists():
            logger.info(f"Processing {case1_sosi_path}...")
            doc1 = parse_sosi_to_docling(case1_sosi_path, sosi_codes, "Evjetun_leirsted")
            if doc1:
                # Write structure to file instead of printing
                structure_output = print_docling_structure(doc1)
                with open(output_dir / f"{doc1.name}_structure.txt", 'w', encoding='utf-8') as f:
                    f.write(structure_output)
                logger.info(f"Wrote structure to {doc1.name}_structure.txt")
                
                try:
                    doc1.save_as_json(output_dir / f"{doc1.name}_structure.json")
                    logger.info(f"Saved {doc1.name} structure to JSON.")
                except Exception as e:
                    logger.error(f"Could not save {doc1.name} to JSON: {e}")
        else:
            logger.error(f"File not found: {case1_sosi_path}")

        if case2_sosi_path.exists():
            logger.info(f"Processing {case2_sosi_path}...")
            doc2 = parse_sosi_to_docling(case2_sosi_path, sosi_codes, "Kjetså_massetak")
            if doc2:
                # Write structure to file instead of printing
                structure_output = print_docling_structure(doc2)
                with open(output_dir / f"{doc2.name}_structure.txt", 'w', encoding='utf-8') as f:
                    f.write(structure_output)
                logger.info(f"Wrote structure to {doc2.name}_structure.txt")
                
                try:
                    doc2.save_as_json(output_dir / f"{doc2.name}_structure.json")
                    logger.info(f"Saved {doc2.name} structure to JSON.")
                except Exception as e:
                    logger.error(f"Could not save {doc2.name} to JSON: {e}")
        else:
            logger.error(f"File not found: {case2_sosi_path}")