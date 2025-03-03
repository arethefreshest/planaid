"""
PDF Extraction Benchmark Script

This script compares different PDF text extraction libraries on various metrics:
- Extraction accuracy
- Processing time
- Memory usage
- Handling of multi-column layouts
- OCR capabilities
"""

import time
import os
import psutil
import logging
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF
from tika import parser
import pytesseract
from PIL import Image
import numpy as np
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFExtractorBenchmark:
    def __init__(self, pdf_dir: str):
        """Initialize benchmark with directory containing test PDFs."""
        self.pdf_dir = Path(pdf_dir)
        self.results = []
        
    def measure_time_and_memory(self, func) -> Tuple[float, float]:
        """Measure execution time and peak memory usage of a function."""
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        result = func()
        
        end_time = time.time()
        end_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        return result, end_time - start_time, end_mem - start_mem

    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using PDFPlumber."""
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages)

    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def extract_with_tika(self, pdf_path: str) -> str:
        """Extract text using Apache Tika."""
        parsed = parser.from_file(pdf_path)
        return parsed["content"]

    def extract_with_ocr(self, pdf_path: str) -> str:
        """Extract text using Tesseract OCR."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img, lang='nor')
        doc.close()
        return text

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        return SequenceMatcher(None, text1, text2).ratio()

    def benchmark_file(self, pdf_path: str, ground_truth: str = None) -> Dict:
        """Run benchmarks for a single PDF file."""
        results = {}
        
        # Test each extractor
        extractors = {
            'pdfplumber': self.extract_with_pdfplumber,
            'pymupdf': self.extract_with_pymupdf,
            'tika': self.extract_with_tika,
            'tesseract': self.extract_with_ocr
        }
        
        for name, extractor in extractors.items():
            try:
                text, time_taken, memory_used = self.measure_time_and_memory(
                    lambda: extractor(pdf_path)
                )
                
                results[name] = {
                    'time': time_taken,
                    'memory': memory_used,
                    'success': True,
                    'text_length': len(text)
                }
                
                # Calculate similarity to ground truth if provided
                if ground_truth:
                    results[name]['similarity'] = self.calculate_similarity(text, ground_truth)
                
            except Exception as e:
                logger.error(f"Error with {name}: {str(e)}")
                results[name] = {
                    'time': 0,
                    'memory': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results

    def run_benchmarks(self) -> pd.DataFrame:
        """Run benchmarks on all PDF files in directory."""
        all_results = []
        
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            logger.info(f"Processing {pdf_file}")
            results = self.benchmark_file(str(pdf_file))
            
            # Format results for DataFrame
            for extractor, metrics in results.items():
                row = {
                    'file': pdf_file.name,
                    'extractor': extractor,
                    **metrics
                }
                all_results.append(row)
        
        return pd.DataFrame(all_results)

    def plot_results(self, df: pd.DataFrame):
        """Generate plots comparing the different extractors."""
        # Time comparison
        plt.figure(figsize=(10, 6))
        df.boxplot(column='time', by='extractor')
        plt.title('Extraction Time by Library')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('extraction_time_comparison.png')
        
        # Memory usage comparison
        plt.figure(figsize=(10, 6))
        df.boxplot(column='memory', by='extractor')
        plt.title('Memory Usage by Library')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('memory_usage_comparison.png')
        
        if 'similarity' in df.columns:
            plt.figure(figsize=(10, 6))
            df.boxplot(column='similarity', by='extractor')
            plt.title('Text Similarity to Ground Truth')
            plt.ylabel('Similarity Ratio')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('similarity_comparison.png')

def main():
    # Create test directory if it doesn't exist
    test_dir = Path("test_pdfs")
    test_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = PDFExtractorBenchmark(str(test_dir))
    
    # Run benchmarks
    results_df = benchmark.run_benchmarks()
    
    # Save results
    results_df.to_csv('pdf_extraction_results.csv', index=False)
    logger.info(f"Results saved to pdf_extraction_results.csv")
    
    # Generate plots
    benchmark.plot_results(results_df)
    logger.info("Plots generated")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(results_df.groupby('extractor').agg({
        'time': ['mean', 'std'],
        'memory': ['mean', 'std'],
        'success': 'mean'
    }))

if __name__ == "__main__":
    main() 