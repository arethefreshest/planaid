"""
Pipeline Test Script

This script tests the complete document processing pipeline:
1. PDF text extraction
2. Document normalization
3. Diff computation

It evaluates different combinations of:
- PDF extraction libraries
- Normalization techniques
- Diff algorithms

The goal is to find the most effective combination for regulatory document comparison.
"""

import sys
from pathlib import Path
import logging
import time
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher, unified_diff
import numpy as np
from dataclasses import dataclass

# Add experiments directory to path
experiments_dir = Path(__file__).parent
sys.path.append(str(experiments_dir))

# Import our experimental modules
from pdf_extraction.pdf_extractor_benchmark import PDFExtractorBenchmark
from normalization.document_normalizer import DocumentNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Stores results of a complete pipeline run."""
    extractor: str
    normalizer: str
    diff_algorithm: str
    extraction_time: float
    normalization_time: float
    diff_time: float
    diff_ratio: float
    metrics: Dict[str, float]

class DiffComputer:
    """Implements different diff algorithms for text comparison."""
    
    @staticmethod
    def sequence_matcher_diff(text1: str, text2: str) -> Tuple[float, List[str]]:
        """Use SequenceMatcher for diff computation."""
        start_time = time.time()
        
        # Compute similarity ratio
        matcher = SequenceMatcher(None, text1, text2)
        ratio = matcher.ratio()
        
        # Get detailed diff
        diff = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                diff.append(f' {text1[i1:i2]}')
            elif tag == 'delete':
                diff.append(f'- {text1[i1:i2]}')
            elif tag == 'insert':
                diff.append(f'+ {text2[j1:j2]}')
            elif tag == 'replace':
                diff.append(f'- {text1[i1:i2]}')
                diff.append(f'+ {text2[j1:j2]}')
        
        diff_time = time.time() - start_time
        return ratio, diff, diff_time

    @staticmethod
    def unified_diff(text1: str, text2: str) -> Tuple[float, List[str]]:
        """Use unified_diff for comparison."""
        start_time = time.time()
        
        # Split into lines
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)
        
        # Compute unified diff
        diff = list(unified_diff(lines1, lines2, fromfile='version1', tofile='version2'))
        
        # Calculate similarity ratio (percentage of unchanged lines)
        total_lines = len(lines1) + len(lines2)
        changed_lines = len([line for line in diff if line.startswith(('+', '-'))])
        ratio = 1 - (changed_lines / total_lines if total_lines > 0 else 0)
        
        diff_time = time.time() - start_time
        return ratio, diff, diff_time

class PipelineTester:
    def __init__(self, test_dir: str):
        """Initialize pipeline tester with test directory."""
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pdf_extractor = PDFExtractorBenchmark(test_dir)
        self.normalizer = DocumentNormalizer()
        self.diff_computer = DiffComputer()
        
        # Available techniques
        self.extractors = ['pdfplumber', 'pymupdf', 'tika']
        self.normalizers = ['basic', 'layout_aware', 'structure_preserving', 'norwegian_specific']
        self.diff_algorithms = ['sequence_matcher', 'unified']
        
        self.results = []

    def run_pipeline(self, pdf_path1: str, pdf_path2: str) -> List[PipelineResult]:
        """Run complete pipeline with all combinations of techniques."""
        results = []
        
        for extractor in self.extractors:
            # Extract text from both PDFs
            logger.info(f"Testing extractor: {extractor}")
            
            try:
                # Get extraction function
                extract_func = getattr(self.pdf_extractor, f"extract_with_{extractor}")
                
                # Extract text from both PDFs
                text1, extraction_time1, _ = self.pdf_extractor.measure_time_and_memory(
                    lambda: extract_func(pdf_path1)
                )
                text2, extraction_time2, _ = self.pdf_extractor.measure_time_and_memory(
                    lambda: extract_func(pdf_path2)
                )
                
                total_extraction_time = extraction_time1 + extraction_time2
                
                # Try each normalization technique
                for norm_technique in self.normalizers:
                    logger.info(f"Testing normalizer: {norm_technique}")
                    
                    try:
                        # Normalize both texts
                        normalize_func = getattr(self.normalizer, f"{norm_technique}_normalize")
                        
                        result1 = normalize_func(text1)
                        result2 = normalize_func(text2)
                        
                        norm_time = result1.processing_time + result2.processing_time
                        
                        # Try each diff algorithm
                        for diff_algo in self.diff_algorithms:
                            logger.info(f"Testing diff algorithm: {diff_algo}")
                            
                            try:
                                # Compute diff
                                if diff_algo == 'sequence_matcher':
                                    ratio, diff, diff_time = self.diff_computer.sequence_matcher_diff(
                                        result1.normalized_text,
                                        result2.normalized_text
                                    )
                                else:
                                    ratio, diff, diff_time = self.diff_computer.unified_diff(
                                        result1.normalized_text,
                                        result2.normalized_text
                                    )
                                
                                # Combine metrics
                                metrics = {
                                    **result1.metrics,
                                    'diff_lines': len(diff)
                                }
                                
                                # Store results
                                pipeline_result = PipelineResult(
                                    extractor=extractor,
                                    normalizer=norm_technique,
                                    diff_algorithm=diff_algo,
                                    extraction_time=total_extraction_time,
                                    normalization_time=norm_time,
                                    diff_time=diff_time,
                                    diff_ratio=ratio,
                                    metrics=metrics
                                )
                                
                                results.append(pipeline_result)
                                
                            except Exception as e:
                                logger.error(f"Error in diff computation: {str(e)}")
                                
                    except Exception as e:
                        logger.error(f"Error in normalization: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error in extraction: {str(e)}")
        
        return results

    def analyze_results(self, results: List[PipelineResult]):
        """Analyze and visualize pipeline results."""
        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                'extractor': r.extractor,
                'normalizer': r.normalizer,
                'diff_algorithm': r.diff_algorithm,
                'extraction_time': r.extraction_time,
                'normalization_time': r.normalization_time,
                'diff_time': r.diff_time,
                'total_time': r.extraction_time + r.normalization_time + r.diff_time,
                'diff_ratio': r.diff_ratio,
                **r.metrics
            }
            for r in results
        ])
        
        # Save detailed results
        df.to_csv(self.test_dir / 'pipeline_results.csv', index=False)
        
        # Create visualizations
        self._plot_timing_comparison(df)
        self._plot_accuracy_comparison(df)
        
        return df

    def _plot_timing_comparison(self, df: pd.DataFrame):
        """Create timing comparison plots."""
        plt.figure(figsize=(15, 8))
        
        # Stacked bar chart of processing times
        timing_data = df.groupby(['extractor', 'normalizer', 'diff_algorithm'])[
            ['extraction_time', 'normalization_time', 'diff_time']
        ].mean()
        
        timing_data.plot(kind='bar', stacked=True)
        plt.title('Processing Time Breakdown by Pipeline Configuration')
        plt.xlabel('(Extractor, Normalizer, Diff Algorithm)')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.test_dir / 'timing_comparison.png')
        plt.close()

    def _plot_accuracy_comparison(self, df: pd.DataFrame):
        """Create accuracy comparison plots."""
        plt.figure(figsize=(15, 8))
        
        # Box plot of diff ratios
        df.boxplot(column='diff_ratio', by=['extractor', 'normalizer'])
        plt.title('Diff Ratio by Extraction and Normalization Method')
        plt.xlabel('(Extractor, Normalizer)')
        plt.ylabel('Diff Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.test_dir / 'accuracy_comparison.png')
        plt.close()

def main():
    # Create test directory
    test_dir = Path("pipeline_test_results")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize pipeline tester
    tester = PipelineTester(str(test_dir))
    
    # Get test PDFs (you would replace these with actual test files)
    pdf_dir = Path("test_pdfs")
    if not pdf_dir.exists():
        logger.error("Test PDF directory not found. Please generate test PDFs first.")
        return
    
    # Find PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if len(pdf_files) < 2:
        logger.error("Need at least 2 PDF files for comparison.")
        return
    
    # Run pipeline tests
    results = tester.run_pipeline(str(pdf_files[0]), str(pdf_files[1]))
    
    # Analyze results
    df = tester.analyze_results(results)
    
    # Print summary
    print("\nPipeline Test Summary:")
    print("\nBest Configurations by Total Time:")
    print(df.nsmallest(3, 'total_time')[
        ['extractor', 'normalizer', 'diff_algorithm', 'total_time', 'diff_ratio']
    ])
    
    print("\nBest Configurations by Diff Ratio:")
    print(df.nlargest(3, 'diff_ratio')[
        ['extractor', 'normalizer', 'diff_algorithm', 'total_time', 'diff_ratio']
    ])

if __name__ == "__main__":
    main() 