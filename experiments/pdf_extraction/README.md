# PDF Extraction Experiments

This directory contains experiments to evaluate different PDF text extraction libraries and approaches. The goal is to determine the most effective method for extracting text from regulatory documents, particularly handling different layouts and content types.

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
- On macOS: `brew install tesseract`
- On Ubuntu: `sudo apt-get install tesseract-ocr`
- On Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki

## Running Experiments

1. Generate test PDFs:
```bash
python generate_test_pdfs.py
```

2. Run benchmarks:
```bash
python pdf_extractor_benchmark.py
```

## Experiment Details

The experiments test different PDF extraction libraries:
- PDFPlumber
- PyMuPDF
- Apache Tika
- Tesseract OCR

On various document types:
1. Simple single-column text
2. Multi-column text
3. Mixed content (text + tables)
4. Scanned-like content

## Metrics

The benchmark measures:
- Extraction accuracy (compared to ground truth)
- Processing time
- Memory usage
- Success rate
- Text length

## Results

Results are saved as:
- CSV file: `pdf_extraction_results.csv`
- Plots:
  - `extraction_time_comparison.png`
  - `memory_usage_comparison.png`
  - `similarity_comparison.png`

## Analysis

The benchmark provides insights into:
- Which library performs best for different document types
- Trade-offs between accuracy and performance
- Memory efficiency
- Handling of complex layouts

Use these results to choose the most appropriate extraction method for your specific use case. 