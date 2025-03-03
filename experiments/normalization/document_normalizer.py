"""
Document Normalization Experiment

This script implements and evaluates different normalization techniques for document text,
focusing on preparing text for accurate diff computation and consistency checking.

Techniques tested:
1. Basic normalization (whitespace, case, punctuation)
2. Layout-aware normalization (handling columns, headers, footers)
3. Structure-preserving normalization (maintaining document hierarchy)
4. Language-specific normalization (Norwegian characters, abbreviations)
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NormalizationResult:
    """Stores results of normalization process."""
    original_text: str
    normalized_text: str
    processing_time: float
    technique: str
    metrics: Dict[str, float]

class DocumentNormalizer:
    def __init__(self):
        """Initialize the normalizer with required models and resources."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Load spaCy model for Norwegian
        self.nlp = spacy.load('nb_core_news_sm')
        
        # Common Norwegian abbreviations and their expansions
        self.no_abbreviations = {
            'bl.a.': 'blant annet',
            'dvs.': 'det vil si',
            'evt.': 'eventuelt',
            'f.eks.': 'for eksempel',
            'mht.': 'med hensyn til',
            'osv.': 'og så videre',
            'vs.': 'versus'
        }
        
        # Regex patterns for common elements
        self.header_patterns = [
            r'^\s*Side \d+\s*$',
            r'^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$',  # Dates
            r'^\s*[A-ZÆØÅ\s]{10,}\s*$'  # All caps headers
        ]
        
        self.footer_patterns = [
            r'^\s*\d+\s*$',  # Page numbers
            r'.*[Kk]ommune\s*$'  # Municipality names
        ]

    def basic_normalize(self, text: str) -> NormalizationResult:
        """
        Perform basic text normalization:
        - Convert to lowercase
        - Standardize whitespace
        - Remove redundant punctuation
        - Standardize line endings
        """
        start_time = time.time()
        
        # Store original for metrics
        original = text
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Standardize whitespace
        text = re.sub(r'[^\w\s\-æøåÆØÅ]', '', text)  # Keep only alphanumeric and Norwegian chars
        text = text.strip()
        
        # Calculate metrics
        processing_time = time.time() - start_time
        metrics = {
            'char_reduction': 1 - (len(text) / len(original)),
            'whitespace_reduction': 1 - (text.count(' ') / original.count(' '))
        }
        
        return NormalizationResult(
            original_text=original,
            normalized_text=text,
            processing_time=processing_time,
            technique='basic',
            metrics=metrics
        )

    def layout_aware_normalize(self, text: str) -> NormalizationResult:
        """
        Normalize text while preserving meaningful layout information:
        - Remove headers/footers
        - Handle multi-column text
        - Preserve paragraph breaks
        """
        start_time = time.time()
        original = text
        
        # Split into lines
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Skip headers and footers
            if any(re.match(pattern, line) for pattern in self.header_patterns + self.footer_patterns):
                continue
            
            # Handle potential column merging
            if re.match(r'^\s*\w+', line):  # Line starts with word
                if normalized_lines and re.match(r'\w+\s*$', normalized_lines[-1]):  # Previous line ends with word
                    # Potential column break - join with space
                    normalized_lines[-1] = f"{normalized_lines[-1]} {line.strip()}"
                    continue
            
            normalized_lines.append(line.strip())
        
        # Join lines, preserving paragraph breaks
        text = '\n\n'.join(' '.join(group) for group in self._group_paragraphs(normalized_lines))
        
        # Calculate metrics
        processing_time = time.time() - start_time
        metrics = {
            'lines_removed': 1 - (len(normalized_lines) / len(lines)),
            'paragraphs': text.count('\n\n') + 1
        }
        
        return NormalizationResult(
            original_text=original,
            normalized_text=text,
            processing_time=processing_time,
            technique='layout_aware',
            metrics=metrics
        )

    def structure_preserving_normalize(self, text: str) -> NormalizationResult:
        """
        Normalize while maintaining document structure:
        - Preserve section hierarchy
        - Maintain list formatting
        - Keep table-like structures
        """
        start_time = time.time()
        original = text
        
        # Convert to structured format (HTML-like)
        soup = BeautifulSoup(f"<doc>{text}</doc>", 'html.parser')
        
        # Identify and mark sections
        for line in soup.find_all(text=True):
            if re.match(r'^\s*\d+\.\s+', str(line)):  # Section numbers
                new_tag = soup.new_tag('section')
                line.wrap(new_tag)
            elif re.match(r'^\s*[a-zæøå]\)\s+', str(line)):  # List items
                new_tag = soup.new_tag('li')
                line.wrap(new_tag)
        
        # Extract text while preserving structure
        text = self._extract_structured_text(soup)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        metrics = {
            'sections': len(soup.find_all('section')),
            'list_items': len(soup.find_all('li'))
        }
        
        return NormalizationResult(
            original_text=original,
            normalized_text=text,
            processing_time=processing_time,
            technique='structure_preserving',
            metrics=metrics
        )

    def norwegian_specific_normalize(self, text: str) -> NormalizationResult:
        """
        Apply Norwegian-specific normalization:
        - Handle special characters (æ, ø, å)
        - Expand common abbreviations
        - Normalize date formats
        - Handle Norwegian-specific tokens
        """
        start_time = time.time()
        original = text
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Normalize tokens
        normalized_tokens = []
        for token in doc:
            # Expand abbreviations
            if token.text.lower() in self.no_abbreviations:
                normalized_tokens.append(self.no_abbreviations[token.text.lower()])
            # Normalize dates
            elif re.match(r'\d{2}\.\d{2}\.\d{4}', token.text):
                normalized_tokens.append(self._normalize_date(token.text))
            else:
                normalized_tokens.append(token.text)
        
        text = ' '.join(normalized_tokens)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        metrics = {
            'abbreviations_expanded': sum(1 for t in doc if t.text.lower() in self.no_abbreviations),
            'dates_normalized': sum(1 for t in doc if re.match(r'\d{2}\.\d{2}\.\d{4}', t.text))
        }
        
        return NormalizationResult(
            original_text=original,
            normalized_text=text,
            processing_time=processing_time,
            technique='norwegian_specific',
            metrics=metrics
        )

    def _group_paragraphs(self, lines: List[str]) -> List[List[str]]:
        """Group lines into paragraphs based on empty lines."""
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            if not line.strip():
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs

    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """Extract text while maintaining structure markers."""
        text = []
        for element in soup.descendants:
            if element.name == 'section':
                text.append(f"§{element.get_text().strip()}")
            elif element.name == 'li':
                text.append(f"• {element.get_text().strip()}")
            elif isinstance(element, str) and element.strip():
                text.append(element.strip())
        return '\n'.join(text)

    def _normalize_date(self, date_str: str) -> str:
        """Normalize Norwegian date format."""
        try:
            day, month, year = date_str.split('.')
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except:
            return date_str

    def normalize_all(self, text: str) -> Dict[str, NormalizationResult]:
        """Apply all normalization techniques and return results."""
        return {
            'basic': self.basic_normalize(text),
            'layout_aware': self.layout_aware_normalize(text),
            'structure_preserving': self.structure_preserving_normalize(text),
            'norwegian_specific': self.norwegian_specific_normalize(text)
        }

def main():
    # Create output directory
    output_dir = Path("normalization_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize normalizer
    normalizer = DocumentNormalizer()
    
    # Test with sample texts (you would replace these with real documents)
    test_texts = {
        'simple': """
        Dette er en test.
        Med flere linjer.
        Og noen tall: 123.
        """,
        'with_layout': """
        Side 1
        
        Kolonne 1    Kolonne 2
        Text her     Mer text
        Fortsetter   Også her
        
        Side 2
        """,
        'with_structure': """
        1. Hovedseksjon
           a) Første punkt
           b) Andre punkt
        
        2. Andre seksjon
           Dette er noe tekst.
        """,
        'with_norwegian': """
        Møte den 01.02.2024
        
        Bl.a. følgende ble diskutert:
        - Punkt 1
        - Osv.
        """
    }
    
    # Process each test text
    all_results = []
    for text_type, text in test_texts.items():
        results = normalizer.normalize_all(text)
        
        # Save individual results
        for technique, result in results.items():
            result_dict = {
                'text_type': text_type,
                'technique': technique,
                'processing_time': result.processing_time,
                **result.metrics
            }
            all_results.append(result_dict)
            
            # Save normalized text
            output_file = output_dir / f"{text_type}_{technique}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.normalized_text)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save metrics
    results_df.to_csv(output_dir / 'normalization_metrics.csv', index=False)
    
    # Generate summary plots
    import matplotlib.pyplot as plt
    
    # Processing time comparison
    plt.figure(figsize=(10, 6))
    results_df.pivot(index='text_type', columns='technique', values='processing_time').plot(kind='bar')
    plt.title('Processing Time by Technique and Text Type')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'processing_time_comparison.png')
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(results_df.groupby('technique').agg({
        'processing_time': ['mean', 'std']
    }))

if __name__ == "__main__":
    main() 