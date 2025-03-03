# Document Normalization Experiments

This directory contains experiments to evaluate different document normalization techniques, focusing on preparing text for accurate diff computation and consistency checking between regulatory documents.

## Normalization Techniques

1. **Basic Normalization**
   - Whitespace standardization
   - Case normalization
   - Punctuation handling
   - Line ending standardization

2. **Layout-Aware Normalization**
   - Header/footer removal
   - Multi-column text handling
   - Paragraph structure preservation
   - Table and list formatting

3. **Structure-Preserving Normalization**
   - Section hierarchy maintenance
   - List item preservation
   - Table structure retention
   - Document structure markers

4. **Norwegian-Specific Normalization**
   - Special character handling (æ, ø, å)
   - Abbreviation expansion
   - Date format standardization
   - Norwegian token handling

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

3. Install the Norwegian language model for spaCy:
```bash
python -m spacy download nb_core_news_sm
```

## Running Experiments

Run the normalization experiments:
```bash
python document_normalizer.py
```

## Results

The experiments generate several outputs in the `normalization_results` directory:

1. **Normalized Text Files**
   - `{text_type}_{technique}.txt` for each combination of text type and normalization technique

2. **Metrics**
   - `normalization_metrics.csv`: Detailed metrics for each normalization technique
   - `processing_time_comparison.png`: Visual comparison of processing times

3. **Metrics Measured**
   - Processing time
   - Character reduction ratio
   - Whitespace reduction
   - Lines removed (for layout-aware)
   - Structure preservation (sections, list items)
   - Norwegian-specific metrics (abbreviations expanded, dates normalized)

## Analysis

The results help determine:
- Most effective normalization technique for different document types
- Performance implications of each technique
- Trade-offs between normalization aggressiveness and structure preservation
- Impact on downstream tasks (diff computation, consistency checking)

## Example Usage

```python
from document_normalizer import DocumentNormalizer

normalizer = DocumentNormalizer()

# Apply all normalization techniques
results = normalizer.normalize_all(text)

# Or use specific techniques
basic_result = normalizer.basic_normalize(text)
layout_result = normalizer.layout_aware_normalize(text)
structure_result = normalizer.structure_preserving_normalize(text)
norwegian_result = normalizer.norwegian_specific_normalize(text)
```

## Integration with Main Project

These normalization techniques can be integrated into the main project's workflow:

1. **PDF Processing Pipeline**
   ```
   Raw PDF → Text Extraction → Normalization → Diff/Consistency Check
   ```

2. **Consistency Checking**
   ```
   Planbestemmelser PDF → Normalize → Extract Fields
   Plankart PDF → Normalize → Extract Fields
   SOSI File → Parse → Normalize → Extract Fields
   Compare Normalized Fields
   ```

3. **Version Comparison**
   ```
   Original Version → Normalize
   New Version → Normalize
   Compute Diff on Normalized Texts
   ``` 