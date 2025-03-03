"""
Field Name Matching Benchmark

This script tests different approaches to matching field names:
1. String similarity algorithms
2. Normalization strategies
3. Norwegian-specific handling
4. Prefix handling

The goal is to find the most accurate method for matching field names
across different documents.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
import jellyfish
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Stores results of a field matching test."""
    method: str
    true_positives: int
    false_positives: int
    false_negatives: int
    processing_time: float
    metrics: Dict[str, float]

class FieldMatcher:
    """Tests different field matching strategies."""
    
    def __init__(self):
        """Initialize matcher with test data."""
        # Common prefixes in Norwegian regulatory documents
        self.prefixes = ['o_', 'f_']
        
        # Test data: tuples of (field1, field2, should_match)
        self.test_cases = [
            # Exact matches
            ('BRA1', 'BRA1', True),
            ('o_BRA1', 'o_BRA1', True),
            ('f_BRA1', 'f_BRA1', True),
            
            # Prefix variations
            ('BRA1', 'o_BRA1', True),
            ('BRA1', 'f_BRA1', True),
            ('o_BRA1', 'f_BRA1', False),
            
            # Case variations
            ('bra1', 'BRA1', True),
            ('O_BRA1', 'o_BRA1', True),
            
            # Number variations
            ('BRA1', 'BRA2', False),
            ('BRA10', 'BRA1', False),
            
            # Special characters
            ('BRA-1', 'BRA1', True),
            ('BRA_1', 'BRA1', True),
            ('BRA.1', 'BRA1', True),
            
            # Norwegian characters
            ('BRÅ1', 'BRA1', False),
            ('BRÆ1', 'BRE1', False),
            ('BRØ1', 'BRO1', False),
            
            # Common typos
            ('BRA1', 'BRA!', False),
            ('BRA1', 'BRA11', False),
            ('BRA1', 'BRA_11', False)
        ]
        
    def basic_normalize(self, field: str) -> str:
        """Basic normalization: lowercase and remove special characters."""
        return re.sub(r'[^\w\s]', '', field.lower())
        
    def advanced_normalize(self, field: str) -> str:
        """Advanced normalization with prefix handling."""
        # Remove known prefixes
        for prefix in self.prefixes:
            if field.lower().startswith(prefix):
                field = field[len(prefix):]
                break
                
        # Convert to uppercase for consistency
        field = field.upper()
        
        # Remove special characters but keep numbers
        field = re.sub(r'[^\w\d]', '', field)
        
        return field
        
    def norwegian_normalize(self, field: str) -> str:
        """Normalization with Norwegian character handling."""
        # Map Norwegian characters to basic Latin
        replacements = {
            'æ': 'ae',
            'ø': 'o',
            'å': 'a',
            'Æ': 'AE',
            'Ø': 'O',
            'Å': 'A'
        }
        
        for nor, lat in replacements.items():
            field = field.replace(nor, lat)
            
        return self.advanced_normalize(field)
        
    def exact_match(self, field1: str, field2: str) -> bool:
        """Simple exact matching after basic normalization."""
        return self.basic_normalize(field1) == self.basic_normalize(field2)
        
    def similarity_match(self, field1: str, field2: str, threshold: float = 0.8) -> bool:
        """Match using string similarity with threshold."""
        norm1 = self.advanced_normalize(field1)
        norm2 = self.advanced_normalize(field2)
        return SequenceMatcher(None, norm1, norm2).ratio() >= threshold
        
    def levenshtein_match(self, field1: str, field2: str, max_distance: int = 2) -> bool:
        """Match using Levenshtein distance."""
        norm1 = self.advanced_normalize(field1)
        norm2 = self.advanced_normalize(field2)
        return levenshtein_distance(norm1, norm2) <= max_distance
        
    def soundex_match(self, field1: str, field2: str) -> bool:
        """Match using Soundex phonetic algorithm."""
        norm1 = self.norwegian_normalize(field1)
        norm2 = self.norwegian_normalize(field2)
        return jellyfish.soundex(norm1) == jellyfish.soundex(norm2)
        
    def test_matcher(self, match_func, name: str) -> MatchResult:
        """Test a matching function against test cases."""
        start_time = time.time()
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for field1, field2, should_match in self.test_cases:
            does_match = match_func(field1, field2)
            
            if does_match and should_match:
                true_positives += 1
            elif does_match and not should_match:
                false_positives += 1
            elif not does_match and should_match:
                false_negatives += 1
                
        processing_time = time.time() - start_time
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return MatchResult(
            method=name,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            processing_time=processing_time,
            metrics={
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': (true_positives + len(self.test_cases) - (true_positives + false_positives + false_negatives)) / len(self.test_cases)
            }
        )
        
    def run_experiments(self) -> pd.DataFrame:
        """Run all field matching experiments."""
        results = []
        
        # Test each matching strategy
        matchers = [
            (self.exact_match, 'exact'),
            (lambda x, y: self.similarity_match(x, y, 0.8), 'similarity_0.8'),
            (lambda x, y: self.similarity_match(x, y, 0.9), 'similarity_0.9'),
            (lambda x, y: self.levenshtein_match(x, y, 1), 'levenshtein_1'),
            (lambda x, y: self.levenshtein_match(x, y, 2), 'levenshtein_2'),
            (self.soundex_match, 'soundex')
        ]
        
        for matcher, name in tqdm(matchers, desc="Testing matching strategies"):
            result = self.test_matcher(matcher, name)
            results.append(result)
            
        return self._analyze_results(results)
        
    def _analyze_results(self, results: List[MatchResult]) -> pd.DataFrame:
        """Analyze and visualize experiment results."""
        # Convert results to DataFrame
        rows = []
        for result in results:
            row = {
                'method': result.method,
                'true_positives': result.true_positives,
                'false_positives': result.false_positives,
                'false_negatives': result.false_negatives,
                'processing_time': result.processing_time,
                **result.metrics
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Create visualizations
        self._plot_accuracy_metrics(df)
        self._plot_confusion_matrix(df)
        
        return df
        
    def _plot_accuracy_metrics(self, df: pd.DataFrame):
        """Plot accuracy metrics for each method."""
        plt.figure(figsize=(12, 6))
        
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        df[['method'] + metrics].set_index('method').plot(kind='bar')
        
        plt.title('Accuracy Metrics by Method')
        plt.xlabel('Method')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('accuracy_metrics.png')
        plt.close()
        
    def _plot_confusion_matrix(self, df: pd.DataFrame):
        """Plot confusion matrix-like metrics."""
        plt.figure(figsize=(10, 6))
        
        metrics = ['true_positives', 'false_positives', 'false_negatives']
        df[['method'] + metrics].set_index('method').plot(kind='bar', stacked=True)
        
        plt.title('Error Analysis by Method')
        plt.xlabel('Method')
        plt.ylabel('Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('error_analysis.png')
        plt.close()

def main():
    # Initialize matcher
    matcher = FieldMatcher()
    
    # Run experiments
    df = matcher.run_experiments()
    
    # Save results
    df.to_csv('field_matching_results.csv', index=False)
    
    # Print summary
    print("\nField Matching Results:")
    print("\nAccuracy Metrics:")
    print(df[['method', 'precision', 'recall', 'f1_score', 'accuracy']])
    
    print("\nProcessing Times:")
    print(df[['method', 'processing_time']])
    
    print("\nResults saved to field_matching_results.csv")

if __name__ == "__main__":
    main() 