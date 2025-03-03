"""
SOSI File Processing Benchmark

This script evaluates different approaches to parsing and processing SOSI files:
1. Direct text-based parsing
2. GDAL/OGR conversion
3. Custom hierarchical parser
4. Field extraction and normalization

The goal is to find the most effective method for extracting and normalizing
field names and regulation codes from SOSI files.
"""

import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import subprocess
from osgeo import gdal, ogr
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SOSIField:
    """Represents a field extracted from SOSI file."""
    name: str
    regulation_code: Optional[str]
    owner_type: Optional[str]  # e.g., 'o_' for public, 'f_' for shared
    attributes: Dict[str, str]

@dataclass
class ParsingResult:
    """Stores results of SOSI parsing process."""
    method: str
    processing_time: float
    fields_extracted: List[SOSIField]
    metrics: Dict[str, float]

class SOSIParser:
    def __init__(self, sosi_codes_file: Optional[str] = None):
        """Initialize parser with optional SOSI codes reference file."""
        self.sosi_codes = {}
        if sosi_codes_file:
            self.load_sosi_codes(sosi_codes_file)

    def load_sosi_codes(self, csv_path: str):
        """Load SOSI codes and their meanings from CSV file."""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sosi_codes[row['code']] = row['description']

    def direct_parse(self, sosi_path: str) -> ParsingResult:
        """
        Parse SOSI file directly using text processing.
        This is the simplest approach, using regex to find field names.
        """
        start_time = time.time()
        fields = []
        
        try:
            with open(sosi_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find all field name declarations
            field_matches = re.finditer(r'\.\.FELTNAVN\s+([^\n]+)', content)
            
            for match in field_matches:
                field_name = match.group(1).strip()
                
                # Look for associated attributes
                context = content[max(0, match.start() - 500):match.end() + 500]
                
                # Extract regulation code if present
                reg_code = None
                reg_match = re.search(r'\.\.RPAREALFORMÅL\s+(\d+)', context)
                if reg_match:
                    reg_code = reg_match.group(1)
                
                # Determine owner type
                owner_type = None
                if field_name.startswith('o_'):
                    owner_type = 'public'
                elif field_name.startswith('f_'):
                    owner_type = 'shared'
                
                # Extract other attributes
                attrs = {}
                for attr_match in re.finditer(r'\.\.[A-ZÆØÅ]+\s+([^\n]+)', context):
                    attr_name = attr_match.group(0).split()[0][2:]  # Remove '..'
                    attrs[attr_name] = attr_match.group(1).strip()
                
                fields.append(SOSIField(
                    name=field_name,
                    regulation_code=reg_code,
                    owner_type=owner_type,
                    attributes=attrs
                ))
        
        except Exception as e:
            logger.error(f"Error in direct parsing: {str(e)}")
            
        processing_time = time.time() - start_time
        
        return ParsingResult(
            method='direct',
            processing_time=processing_time,
            fields_extracted=fields,
            metrics={
                'field_count': len(fields),
                'fields_with_regcode': sum(1 for f in fields if f.regulation_code),
                'public_fields': sum(1 for f in fields if f.owner_type == 'public'),
                'shared_fields': sum(1 for f in fields if f.owner_type == 'shared')
            }
        )

    def gdal_parse(self, sosi_path: str) -> ParsingResult:
        """
        Parse SOSI file using GDAL/OGR.
        This approach converts SOSI to GeoJSON first.
        """
        start_time = time.time()
        fields = []
        
        try:
            # Convert SOSI to GeoJSON using ogr2ogr
            temp_geojson = Path(sosi_path).with_suffix('.geojson')
            subprocess.run([
                'ogr2ogr',
                '-f', 'GeoJSON',
                str(temp_geojson),
                sosi_path
            ], check=True)
            
            # Read GeoJSON using OGR
            driver = ogr.GetDriverByName('GeoJSON')
            dataset = driver.Open(str(temp_geojson))
            
            if dataset:
                layer = dataset.GetLayer(0)
                for feature in layer:
                    field_name = feature.GetField('FELTNAVN')
                    if field_name:
                        reg_code = feature.GetField('RPAREALFORMÅL')
                        
                        # Extract all fields as attributes
                        attrs = {}
                        for i in range(feature.GetFieldCount()):
                            field_def = feature.GetFieldDefnRef(i)
                            field = field_def.GetName()
                            value = feature.GetField(i)
                            if value and field not in ['FELTNAVN', 'RPAREALFORMÅL']:
                                attrs[field] = str(value)
                        
                        fields.append(SOSIField(
                            name=field_name,
                            regulation_code=reg_code,
                            owner_type='public' if field_name.startswith('o_') else (
                                'shared' if field_name.startswith('f_') else None
                            ),
                            attributes=attrs
                        ))
            
            # Cleanup
            dataset = None
            temp_geojson.unlink()
            
        except Exception as e:
            logger.error(f"Error in GDAL parsing: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return ParsingResult(
            method='gdal',
            processing_time=processing_time,
            fields_extracted=fields,
            metrics={
                'field_count': len(fields),
                'fields_with_regcode': sum(1 for f in fields if f.regulation_code),
                'public_fields': sum(1 for f in fields if f.owner_type == 'public'),
                'shared_fields': sum(1 for f in fields if f.owner_type == 'shared')
            }
        )

    def hierarchical_parse(self, sosi_path: str) -> ParsingResult:
        """
        Parse SOSI file using a custom hierarchical parser.
        This approach maintains the SOSI file's structure.
        """
        start_time = time.time()
        fields = []
        
        try:
            with open(sosi_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_object = {}
            current_level = 0
            
            for line in lines:
                # Count leading dots to determine hierarchy level
                level = 0
                for char in line:
                    if char == '.':
                        level += 1
                    else:
                        break
                
                line = line.strip()
                if not line:
                    continue
                
                # Remove dots and split into key-value
                parts = line[level:].strip().split(maxsplit=1)
                key = parts[0]
                value = parts[1] if len(parts) > 1 else None
                
                if level == 0:
                    # New top-level object
                    if current_object and 'FELTNAVN' in current_object:
                        fields.append(self._create_field_from_object(current_object))
                    current_object = {}
                    current_level = 0
                elif level > current_level:
                    # Going deeper in hierarchy
                    current_object[key] = value
                elif level <= current_level:
                    # Same level or coming back up
                    if current_object and 'FELTNAVN' in current_object:
                        fields.append(self._create_field_from_object(current_object))
                    current_object = {key: value}
                
                current_level = level
            
            # Don't forget the last object
            if current_object and 'FELTNAVN' in current_object:
                fields.append(self._create_field_from_object(current_object))
                
        except Exception as e:
            logger.error(f"Error in hierarchical parsing: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return ParsingResult(
            method='hierarchical',
            processing_time=processing_time,
            fields_extracted=fields,
            metrics={
                'field_count': len(fields),
                'fields_with_regcode': sum(1 for f in fields if f.regulation_code),
                'public_fields': sum(1 for f in fields if f.owner_type == 'public'),
                'shared_fields': sum(1 for f in fields if f.owner_type == 'shared')
            }
        )

    def _create_field_from_object(self, obj: Dict[str, str]) -> SOSIField:
        """Create a SOSIField from a parsed object dictionary."""
        field_name = obj.get('FELTNAVN', '')
        return SOSIField(
            name=field_name,
            regulation_code=obj.get('RPAREALFORMÅL'),
            owner_type='public' if field_name.startswith('o_') else (
                'shared' if field_name.startswith('f_') else None
            ),
            attributes={k: v for k, v in obj.items() if k not in ['FELTNAVN', 'RPAREALFORMÅL']}
        )

    def normalize_field_name(self, field: SOSIField) -> str:
        """
        Normalize a field name for consistency checking.
        Handles prefixes, case, and special characters.
        """
        name = field.name
        
        # Remove owner prefix if present
        if name.startswith(('o_', 'f_')):
            name = name[2:]
        
        # Convert to uppercase for consistency
        name = name.upper()
        
        # Remove any non-alphanumeric characters except underscore
        name = re.sub(r'[^\w\s]', '', name)
        
        return name

    def analyze_results(self, results: List[ParsingResult], output_dir: Path):
        """Analyze and visualize parsing results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'method': result.method,
                'processing_time': result.processing_time,
                **result.metrics
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save detailed results
        df.to_csv(output_dir / 'parsing_results.csv', index=False)
        
        # Create visualizations
        self._plot_timing_comparison(df, output_dir)
        self._plot_field_counts(df, output_dir)
        
        return df

    def _plot_timing_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Create timing comparison plot."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar', x='method', y='processing_time')
        plt.title('Processing Time by Parsing Method')
        plt.xlabel('Method')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'timing_comparison.png')
        plt.close()

    def _plot_field_counts(self, df: pd.DataFrame, output_dir: Path):
        """Create field count comparison plot."""
        import matplotlib.pyplot as plt
        
        metrics = ['field_count', 'fields_with_regcode', 'public_fields', 'shared_fields']
        
        plt.figure(figsize=(12, 6))
        df[['method'] + metrics].set_index('method').plot(kind='bar')
        plt.title('Field Counts by Parsing Method')
        plt.xlabel('Method')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'field_counts.png')
        plt.close()

def main():
    # Create output directory
    output_dir = Path("sosi_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize parser
    parser = SOSIParser("sosi_codes.csv")  # You would create this CSV with regulation codes
    
    # Test with sample SOSI file (you would replace this with real test files)
    test_dir = Path("test_files")
    if not test_dir.exists():
        logger.error("Test files directory not found. Please add test SOSI files first.")
        return
    
    # Process each SOSI file
    results = []
    for sosi_file in test_dir.glob("*.sos"):
        logger.info(f"Processing {sosi_file}")
        
        # Try each parsing method
        results.extend([
            parser.direct_parse(str(sosi_file)),
            parser.gdal_parse(str(sosi_file)),
            parser.hierarchical_parse(str(sosi_file))
        ])
    
    # Analyze results
    df = parser.analyze_results(results, output_dir)
    
    # Print summary
    print("\nParsing Results Summary:")
    print("\nProcessing Time (seconds):")
    print(df.groupby('method')['processing_time'].agg(['mean', 'std']))
    
    print("\nField Counts:")
    print(df.groupby('method')[['field_count', 'fields_with_regcode']].mean())

if __name__ == "__main__":
    main() 