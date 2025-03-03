# Algorithm Optimization Experiments

This directory contains experiments to evaluate different algorithmic optimizations and data structures for improving the performance and efficiency of the document processing pipeline.

## Algorithms Tested

1. **MinHash and LSH**
   - Fast document similarity computation
   - Efficient field matching
   - Probabilistic similarity search
   - Configurable accuracy/performance trade-off

2. **Trie-based Field Indexing**
   - Efficient prefix matching
   - Memory-optimized field storage
   - Fast field lookup
   - Support for hierarchical fields

3. **Suffix Arrays**
   - Fast text pattern matching
   - Efficient substring search
   - Support for multiple search patterns
   - Memory-efficient implementation

4. **Bloom Filters**
   - Fast field existence checking
   - Memory-efficient set representation
   - Configurable false positive rate
   - Support for large field sets

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

## Running Experiments

Run the algorithm benchmarks:
```bash
python algorithm_benchmark.py
```

## Test Data

The experiments use:
- Field name patterns from real regulatory documents
- Various document structures and layouts
- Different field naming conventions
- Norwegian-specific patterns

## Results

The experiments generate:

1. **Performance Metrics**
   - Preprocessing time
   - Query time
   - Memory usage
   - Accuracy scores

2. **Visualizations**
   - Timing comparisons
   - Accuracy analysis
   - Memory usage plots
   - Trade-off analysis

3. **CSV Data**
   - Detailed metrics
   - Raw timing data
   - Memory profiles
   - Accuracy measurements

## Analysis

Results help determine:
- Best algorithm for each use case
- Performance bottlenecks
- Memory/speed trade-offs
- Scalability characteristics

## Integration

These algorithms can be integrated into:

1. **Field Extraction**
   ```
   Document → Extract → Index → Search
   ```

2. **Field Matching**
   ```
   Fields → Hash → LSH → Match
   ```

3. **Text Search**
   ```
   Content → Suffix Array → Search → Results
   ```

4. **Field Validation**
   ```
   Field → Bloom Filter → Quick Check
   ```

## Dependencies

- Python 3.10+
- NumPy for computations
- Pandas for analysis
- Specialized algorithm libraries
- Visualization tools

## Future Work

Areas for further optimization:
1. Distributed algorithms
2. GPU acceleration
3. Custom data structures
4. Hybrid approaches 