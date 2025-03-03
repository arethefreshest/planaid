"""
Test SOSI File Generator

This script generates test SOSI files with various structures and content to evaluate
parsing approaches. It creates:
1. Simple SOSI file with basic fields
2. Complex SOSI file with nested structures
3. SOSI file with various field types and prefixes
4. SOSI file with regulation codes and attributes
"""

import random
from pathlib import Path
from typing import List, Dict
import csv

class SOSIGenerator:
    def __init__(self):
        """Initialize generator with common SOSI elements."""
        self.field_prefixes = ['', 'o_', 'f_']
        self.field_types = ['BRA', 'SPP', 'BKS', 'BR', 'BE']
        self.regulation_codes = {
            '1110': 'Boligbebyggelse',
            '1120': 'Fritidsbebyggelse',
            '1130': 'Sentrumsformål',
            '1150': 'Forretninger',
            '1160': 'Offentlig eller privat tjenesteyting',
            '1170': 'Fritids- og turistformål',
            '2010': 'Veg',
            '2019': 'Annen veggrunn - grøntareal',
            '3040': 'Turdrag',
            '3050': 'Park'
        }

    def generate_simple_sosi(self, output_path: Path):
        """Generate a simple SOSI file with basic fields."""
        content = [
            "..OBJTYPE RpArealformålOmråde",
            "..EIERFORM 1",
            "..RPAREALFORMÅL 1110",
            "..FELTNAVN BRA1",
            "..NASJONALAREALPLANID",
            "...PLANID 2024001",
            "...KOMM 4223",
            "..VERTNIV 2",
            "..NØ",
            "645811623 43868995",
            ".FLATE 393:",
            "..OBJTYPE RpArealformålOmråde",
            "..EIERFORM 1",
            "..RPAREALFORMÅL 1120",
            "..FELTNAVN o_BRA2",
            "..NASJONALAREALPLANID",
            "...PLANID 2024001",
            "...KOMM 4223"
        ]
        
        self._write_sosi_file(output_path, content)

    def generate_complex_sosi(self, output_path: Path):
        """Generate a complex SOSI file with nested structures."""
        content = []
        
        # Add header
        content.extend([
            ".HODE",
            "..TEGNSETT UTF-8",
            "..SOSI-VERSJON 4.5",
            "..SOSI-NIVÅ 4",
            "..TRANSPAR",
            "...KOORDSYS 22",
            "...ORIGO-NØ 0 0",
            "...ENHET 0.01"
        ])
        
        # Add multiple features
        for i in range(5):
            feature = [
                f".FLATE {i+1}:",
                "..OBJTYPE RpArealformålOmråde",
                f"..EIERFORM {random.randint(1, 3)}",
                f"..RPAREALFORMÅL {random.choice(list(self.regulation_codes.keys()))}",
                f"..FELTNAVN {self._generate_field_name()}",
                "..NASJONALAREALPLANID",
                "...PLANID 2024001",
                "...KOMM 4223",
                "..VERTNIV 2",
                "..IDENT",
                f"...LOKALID {self._generate_uuid()}",
                "...NAVNEROM http://data.geonorge.no/4223/Reguleringsplaner/so",
                f"...VERSJONID {self._generate_date()}",
                "..NØ"
            ]
            
            # Add some coordinates
            for _ in range(random.randint(3, 6)):
                feature.append(f"{random.randint(600000000, 700000000)} {random.randint(40000000, 50000000)}")
            
            content.extend(feature)
        
        self._write_sosi_file(output_path, content)

    def generate_field_types_sosi(self, output_path: Path):
        """Generate a SOSI file with various field types and prefixes."""
        content = []
        
        # Create features with different combinations
        for prefix in self.field_prefixes:
            for field_type in self.field_types:
                for i in range(1, 3):
                    feature = [
                        f".FLATE {len(content)//10 + 1}:",
                        "..OBJTYPE RpArealformålOmråde",
                        "..EIERFORM 1",
                        f"..RPAREALFORMÅL {random.choice(list(self.regulation_codes.keys()))}",
                        f"..FELTNAVN {prefix}{field_type}{i}",
                        "..NASJONALAREALPLANID",
                        "...PLANID 2024001",
                        "...KOMM 4223",
                        "..NØ",
                        f"{random.randint(600000000, 700000000)} {random.randint(40000000, 50000000)}"
                    ]
                    content.extend(feature)
        
        self._write_sosi_file(output_path, content)

    def generate_regulation_codes_sosi(self, output_path: Path):
        """Generate a SOSI file focusing on regulation codes and attributes."""
        content = []
        
        # Create features for each regulation code
        for code, description in self.regulation_codes.items():
            feature = [
                f".FLATE {len(content)//15 + 1}:",
                "..OBJTYPE RpArealformålOmråde",
                f"..EIERFORM {random.randint(1, 3)}",
                f"..RPAREALFORMÅL {code}",
                f"..FELTNAVN {self._generate_field_name()}",
                "..BESKRIVELSE",
                f"...BESKRIVELSE {description}",
                "..NASJONALAREALPLANID",
                "...PLANID 2024001",
                "...KOMM 4223",
                "..VERTNIV 2",
                "..IDENT",
                f"...LOKALID {self._generate_uuid()}",
                "...NAVNEROM http://data.geonorge.no/4223/Reguleringsplaner/so",
                f"...VERSJONID {self._generate_date()}"
            ]
            content.extend(feature)
        
        self._write_sosi_file(output_path, content)

    def _generate_field_name(self) -> str:
        """Generate a random field name."""
        prefix = random.choice(self.field_prefixes)
        field_type = random.choice(self.field_types)
        number = random.randint(1, 5)
        return f"{prefix}{field_type}{number}"

    def _generate_uuid(self) -> str:
        """Generate a UUID-like string."""
        import uuid
        return str(uuid.uuid4())

    def _generate_date(self) -> str:
        """Generate a date string."""
        year = random.randint(2020, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{year}-{month:02d}-{day:02d}"

    def _write_sosi_file(self, path: Path, content: List[str]):
        """Write content to a SOSI file."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

    def generate_sosi_codes_csv(self, output_path: Path):
        """Generate CSV file with SOSI regulation codes."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['code', 'description'])
            for code, desc in self.regulation_codes.items():
                writer.writerow([code, desc])

def main():
    # Create test directory
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = SOSIGenerator()
    
    # Generate test files
    generator.generate_simple_sosi(test_dir / "simple.sos")
    generator.generate_complex_sosi(test_dir / "complex.sos")
    generator.generate_field_types_sosi(test_dir / "field_types.sos")
    generator.generate_regulation_codes_sosi(test_dir / "regulation_codes.sos")
    
    # Generate SOSI codes reference file
    generator.generate_sosi_codes_csv(Path("sosi_codes.csv"))
    
    print("Generated test SOSI files:")
    for file in test_dir.glob("*.sos"):
        print(f"- {file.name}")
    print("\nGenerated SOSI codes reference file: sosi_codes.csv")

if __name__ == "__main__":
    main() 