# SOSI File Processing Experiments

This directory contains experiments to evaluate different approaches to parsing and processing SOSI files, focusing on field extraction and normalization for regulatory document consistency checking.

## Parsing Approaches

1. **Direct Text-Based Parsing**
   - Simple regex-based extraction
   - Line-by-line processing
   - Basic field name and attribute extraction

2. **GDAL/OGR Conversion**
   - Convert SOSI to GeoJSON
   - Use GDAL's structured parsing
   - Maintain geospatial information

3. **Custom Hierarchical Parser**
   - Preserve SOSI file structure
   - Handle nested elements
   - Track hierarchy levels

4. **Field Extraction and Normalization**
   - Extract field names and regulation codes
   - Handle prefixes (o_, f_)
   - Normalize field names for comparison

## Setup

1. Install GDAL system dependencies:
   ```bash
   # macOS
   brew install gdal
   
   # Ubuntu
   sudo apt-get install gdal-bin python3-gdal
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

1. Generate test SOSI files:
   ```bash
   python generate_test_sosi.py
   ```

2. Run parsing benchmarks:
   ```bash
   python sosi_parser_benchmark.py
   ```

## Test Files Generated

1. **simple.sos**
   - Basic SOSI structure
   - Minimal fields and attributes

2. **complex.sos**
   - Nested structures
   - Multiple features
   - Full SOSI header

3. **field_types.sos**
   - Various field name combinations
   - Different prefixes
   - Multiple field types

4. **regulation_codes.sos**
   - All regulation codes
   - Associated attributes
   - Detailed descriptions

## Results

The experiments generate several outputs:

1. **Parsing Results**
   - CSV file with detailed metrics
   - Processing time comparisons
   - Field extraction accuracy
   - Memory usage statistics

2. **Visualizations**
   - Processing time comparison plots
   - Field count comparisons
   - Success rate analysis

3. **Metrics Measured**
   - Processing time
   - Field extraction accuracy
   - Memory usage
   - Success rate for different field types

## Analysis

The results help determine:
- Most efficient parsing approach
- Trade-offs between methods
- Best approach for different use cases
- Integration recommendations

## Integration with Main Project

These experiments inform:

1. **Field Extraction**
   ```
   SOSI File → Parse → Extract Fields → Normalize → Compare
   ```

2. **Consistency Checking**
   ```
   SOSI Fields ↘
   PDF Fields  → Normalize → Compare → Report Differences
   ```

3. **Performance Optimization**
   ```
   Choose fastest parser → Implement caching → Optimize memory
   ```

## Dependencies

- Python 3.10+
- GDAL/OGR libraries
- Pandas for analysis
- Matplotlib for visualization

## Future Work

Areas for further experimentation:
1. Additional parsing libraries
2. Caching strategies
3. Parallel processing
4. Memory optimization
5. Error handling improvements 