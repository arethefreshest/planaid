# Document Processing Experiments

This directory contains experiments to evaluate and optimize different components of the document processing pipeline for regulatory documents.

## Experiment Structure

1. **PDF Extraction (`pdf_extraction/`)**
   - Comparison of different PDF text extraction libraries
   - Handling of various document layouts
   - OCR integration for scanned documents
   - Performance metrics (speed, accuracy, memory usage)

2. **Document Normalization (`normalization/`)**
   - Text normalization techniques
   - Layout-aware processing
   - Structure preservation
   - Norwegian-specific handling

3. **Pipeline Testing (`pipeline_test.py`)**
   - End-to-end pipeline evaluation
   - Combination of extraction and normalization techniques
   - Diff computation methods
   - Performance and accuracy metrics

## Running the Experiments

1. First, generate test PDFs:
```bash
cd pdf_extraction
python generate_test_pdfs.py
```

2. Run PDF extraction experiments:
```bash
cd pdf_extraction
pip install -r requirements.txt
python pdf_extractor_benchmark.py
```

3. Run normalization experiments:
```bash
cd normalization
pip install -r requirements.txt
python document_normalizer.py
```

4. Run complete pipeline test:
```bash
python pipeline_test.py
```

## Results and Analysis

Each experiment generates its own results:

1. **PDF Extraction Results**
   - CSV file with detailed metrics
   - Plots comparing library performance
   - Analysis of accuracy and speed trade-offs

2. **Normalization Results**
   - Normalized text samples
   - Performance metrics
   - Comparison of different techniques

3. **Pipeline Results**
   - End-to-end performance metrics
   - Best performing combinations
   - Trade-off analysis

## Metrics Collected

1. **Performance Metrics**
   - Processing time
   - Memory usage
   - CPU utilization

2. **Accuracy Metrics**
   - Text extraction accuracy
   - Normalization effectiveness
   - Diff accuracy

3. **Quality Metrics**
   - Structure preservation
   - Layout handling
   - Norwegian language support

## Visualization

Results are visualized through:
- Bar charts comparing processing times
- Box plots of accuracy metrics
- Line graphs showing trade-offs
- Detailed CSV files for further analysis

## Integration with Main Project

These experiments inform the following aspects of the main project:

1. **PDF Processing**
   - Choice of extraction library
   - Normalization pipeline configuration
   - Performance optimization

2. **Consistency Checking**
   - Field extraction approach
   - Text normalization strategy
   - Cross-document comparison

3. **Version Control**
   - Diff computation method
   - Change detection sensitivity
   - Performance considerations

## Future Work

Areas identified for further experimentation:
1. Additional PDF extraction libraries
2. Advanced normalization techniques
3. Machine learning-based approaches
4. Performance optimization strategies
5. Integration with cloud services

## Dependencies

Each experiment directory contains its own `requirements.txt` file. Main dependencies include:
- PDF processing libraries
- Text processing tools
- Visualization packages
- Norwegian language support

## Contributing

To add new experiments:
1. Create a new directory with descriptive name
2. Include a README.md explaining the experiment
3. Add requirements.txt for dependencies
4. Implement the experiment
5. Document results and analysis 